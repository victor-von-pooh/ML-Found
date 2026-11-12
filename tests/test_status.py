import json
from pathlib import Path
from typing import Any

import pytest
from torch import nn

from ml_found.training.trainer import Trainer
from test_trainer import make_experiment, make_loader, regression_step


def read_json(path: Path) -> dict:
    """
    JSON のファイルを読み込む関数

    Parameters
    ----------
    path: Path
        読み込むファイル

    Returns
    ----------
    data: dict
        ファイルの内容
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    return data


def test_completed_status(tmp_path: Path) -> None:
    """
    学習を終えた後の run.json の内容を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 3\n")
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader(), make_loader())

    status = read_json(experiment.run_dir / "run.json")
    assert status["status"] == "completed"
    assert status["planned_epochs"] == 3
    assert status["completed_epochs"] == 3
    assert status["monitor"] == "valid/loss"
    assert status["best_epoch"] == trainer.checkpoint.best_epoch
    assert status["best_value"] == trainer.checkpoint.best_value
    assert status["final_weights"] == "best"
    assert status["seed"] == 0
    assert status["device"] == "cpu"
    assert status["duration_sec"] >= 0
    assert status["error"] is None

    # 終わった学習の進み具合のファイルは残らない
    assert not (experiment.run_dir / "progress.json").exists()


def test_progress_while_training(tmp_path: Path) -> None:
    """
    学習中に progress.json と run.json が読めることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 2\n")
    seen: list[dict[str, Any]] = []

    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        学習中のファイルの内容を記録してから誤差を求める関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: tuple
            x と y
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            平均二乗誤差と平均絶対誤差
        """
        # 学習中のファイルの内容を記録
        seen.append({
            "progress": read_json(experiment.run_dir / "progress.json"),
            "run": read_json(experiment.run_dir / "run.json")
        })
        outputs = regression_step(model, batch, phase)

        return outputs

    trainer = Trainer(nn.Linear(1, 1), experiment, step)
    trainer.status.progress_interval = 0.0
    trainer.fit(make_loader(n=64, batch_size=16), make_loader())

    # エポック 2 の学習の 3 つ目のバッチの時点では 2 バッチ終えている
    record = seen[4 + 4 + 2]
    assert record["run"]["status"] == "running"
    assert record["run"]["completed_epochs"] == 1
    progress = record["progress"]
    assert progress["phase"] == "train"
    assert progress["epoch"] == 2
    assert progress["total_epochs"] == 2
    assert progress["batch"] == 2
    assert progress["total_batches"] == 4


def test_stop_request(tmp_path: Path) -> None:
    """
    STOP のファイルを置くと, そのエポックで止まり中断として残ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(
        tmp_path,
        "trainer:\n  epochs: 10\n  checkpoint:\n    every_n_epochs: 5\n"
    )

    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        エポック 2 で止める要求を出してから誤差を求める関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: tuple
            x と y
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            平均二乗誤差と平均絶対誤差
        """
        # エポック 2 の途中で止める要求を出す
        if trainer.epoch == 2:
            (experiment.run_dir / "STOP").touch()
        outputs = regression_step(model, batch, phase)

        return outputs

    trainer = Trainer(nn.Linear(1, 1), experiment, step)
    trainer.fit(make_loader(), make_loader())

    # エポック 2 を終えてから止まり, 間隔に関係なく last.pt を保存
    assert len(trainer.history.records) == 2
    status = read_json(experiment.run_dir / "run.json")
    assert status["status"] == "stopped"
    assert status["completed_epochs"] == 2
    assert (experiment.run_dir / "final_model.pt").exists()
    last = experiment.run_dir / "last.pt"
    assert last.exists()


def test_failed_status(tmp_path: Path) -> None:
    """
    エラーで止まった場合に失敗として残ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 3\n")

    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        エポック 2 でエラーを起こす関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: tuple
            x と y
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            平均二乗誤差と平均絶対誤差
        """
        # エポック 2 でエラーを起こす
        if trainer.epoch == 2:
            raise RuntimeError("壊れたバッチ")
        outputs = regression_step(model, batch, phase)

        return outputs

    trainer = Trainer(nn.Linear(1, 1), experiment, step)
    with pytest.raises(RuntimeError):
        trainer.fit(make_loader(), make_loader())

    status = read_json(experiment.run_dir / "run.json")
    assert status["status"] == "failed"
    assert status["completed_epochs"] == 1
    assert status["error"] == "RuntimeError: 壊れたバッチ"
    assert not (experiment.run_dir / "progress.json").exists()


def test_interrupted_status(tmp_path: Path) -> None:
    """
    キーボードの割り込みで止まった場合に中断として残ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 3\n")

    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        エポック 2 でキーボードの割り込みを起こす関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: tuple
            x と y
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            平均二乗誤差と平均絶対誤差
        """
        # エポック 2 で Ctrl-C を押したことにする
        if trainer.epoch == 2:
            raise KeyboardInterrupt
        outputs = regression_step(model, batch, phase)

        return outputs

    trainer = Trainer(nn.Linear(1, 1), experiment, step)
    with pytest.raises(KeyboardInterrupt):
        trainer.fit(make_loader(), make_loader())

    status = read_json(experiment.run_dir / "run.json")
    assert status["status"] == "stopped"
    assert status["error"] == "KeyboardInterrupt"


def test_early_stopping_status(tmp_path: Path) -> None:
    """
    Early Stopping で止めたエポックが run.json に残ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 損失が大きいほど良いとすると, すぐに改善しなくなって止まる
    experiment = make_experiment(
        tmp_path,
        "trainer:\n  epochs: 20\n  mode: max\n"
        "  early_stopping:\n    use: true\n    patience: 2\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader(), make_loader())

    status = read_json(experiment.run_dir / "run.json")
    assert status["status"] == "completed"
    assert status["early_stopped_epoch"] == len(trainer.history.records)
    assert status["early_stopped_epoch"] < 20
