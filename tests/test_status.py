import json
from pathlib import Path
from typing import Any

import pytest
from torch import nn

from ml_found.training.callbacks import load_checkpoint
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

    # 終わった学習の進み具合のファイルの削除
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
        # 学習中のファイルの内容の記録
        seen.append({
            "progress": read_json(experiment.run_dir / "progress.json"),
            "run": read_json(experiment.run_dir / "run.json")
        })
        outputs = regression_step(model, batch, phase)

        return outputs

    trainer = Trainer(nn.Linear(1, 1), experiment, step)
    trainer.run_status.progress_interval = 0.0
    train_loader = make_loader(n=64, batch_size=16)
    valid_loader = make_loader(n=32, batch_size=16)
    trainer.fit(train_loader, valid_loader)

    # エポック 2 の学習の 3 つ目のバッチの時点で, 終えたバッチは 2 つ
    per_epoch = len(train_loader) + len(valid_loader)
    record = seen[per_epoch + 2]
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
    requested: list[bool] = []

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
        # 最初の学習のエポック 2 の途中で, 1 回だけ止める要求の作成
        if trainer.epoch == 2 and not requested:
            (experiment.run_dir / "STOP").touch()
            requested.append(True)
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
    last = load_checkpoint(str(experiment.run_dir / "last.pt"), "cpu")
    assert last["epoch"] == 2

    # 同じ Trainer で STOP を消して学習し直すと完了
    (experiment.run_dir / "STOP").unlink()
    trainer.epoch = 0
    trainer.history.records.clear()
    trainer.fit(make_loader(), make_loader())
    assert read_json(experiment.run_dir / "run.json")["status"] == "completed"


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
        # エポック 2 でエラーの発生
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
        # エポック 2 で Ctrl-C を押した場合の再現
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
    assert not (experiment.run_dir / "progress.json").exists()


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
    # 損失が大きいほど良いとして, すぐに改善しなくなる設定
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


def test_stop_on_last_epoch_and_with_early_stopping(tmp_path: Path) -> None:
    """
    最後のエポックと Early Stopping と同時の STOP の扱いを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 予定の最後より前のエポックで止めた場合は中断
    last = make_experiment(tmp_path / "last", "trainer:\n  epochs: 2\n")
    (last.run_dir / "STOP").touch()
    trainer = Trainer(nn.Linear(1, 1), last, regression_step)
    trainer.fit(make_loader(), make_loader())
    assert len(trainer.history.records) == 1
    assert read_json(last.run_dir / "run.json")["status"] == "stopped"

    # 予定の最後のエポックで STOP を置いても, 全て学習したので完了
    one = make_experiment(tmp_path / "one", "trainer:\n  epochs: 1\n")
    (one.run_dir / "STOP").touch()
    trainer = Trainer(nn.Linear(1, 1), one, regression_step)
    trainer.fit(make_loader(), make_loader())
    assert read_json(one.run_dir / "run.json")["status"] == "completed"

    # 損失が大きいほど良いとすると, エポック 2 で改善せず Early Stopping
    # 同じエポック 2 の途中で STOP を置いても, Early Stopping を優先して完了
    both = make_experiment(
        tmp_path / "both",
        "trainer:\n  epochs: 20\n  mode: max\n"
        "  early_stopping:\n    use: true\n    patience: 1\n"
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
        # エポック 2 の途中で止める要求の作成
        if trainer.epoch == 2:
            (both.run_dir / "STOP").touch()
        outputs = regression_step(model, batch, phase)

        return outputs

    trainer = Trainer(nn.Linear(1, 1), both, step)
    trainer.fit(make_loader(), make_loader())
    status = read_json(both.run_dir / "run.json")
    assert status["status"] == "completed"
    assert status["early_stopped_epoch"] == 2
    assert trainer.stopped_by_request is False


def test_write_failure_does_not_stop_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    状態のファイルを書けなくても学習が続き, 警告が 1 回だけ残ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        関数を差し替える仕組み

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 3\n")

    def fail(path: Path, data: dict) -> None:
        """
        ディスクの不足を再現する関数

        Parameters
        ----------
        path: Path
            書き込むファイル
        data: dict
            書き込む内容

        Returns
        ----------
        None
        """
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("ml_found.training.status.write_json", fail)
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader(), make_loader())
    assert len(trainer.history.records) == 3

    log = (experiment.run_dir / "train.log").read_text(encoding="utf-8")
    assert log.count("実験の状態を書き込めませんでした") == 1


def test_progress_throttle(tmp_path: Path) -> None:
    """
    間隔の中ではバッチごとに progress.json を書かないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 1\n")
    writes: list[str] = []
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    original = trainer.run_status.write

    def record(name: str, data: dict) -> None:
        """
        書き込んだファイル名を記録してから書き込む関数

        Parameters
        ----------
        name: str
            ファイル名
        data: dict
            書き込む内容

        Returns
        ----------
        None
        """
        writes.append(name)
        original(name, data)

    trainer.run_status.write = record  # type: ignore[method-assign]
    trainer.run_status.progress_interval = 3600.0
    trainer.fit(make_loader(n=64, batch_size=4), make_loader())

    # 学習と検証の始めと終わりの強制の 4 回だけ (バッチは 16 + 4 個)
    assert writes.count("progress.json") == 4


def test_resume_status(tmp_path: Path) -> None:
    """
    再開した学習の run.json の値を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    first = make_experiment(tmp_path / "first", "trainer:\n  epochs: 2\n")
    trainer = Trainer(nn.Linear(1, 1), first, regression_step)
    trainer.fit(make_loader(), make_loader())

    second = make_experiment(
        tmp_path / "second",
        "trainer:\n  epochs: 4\n"
        f"  resume_from: {first.run_dir / 'last.pt'}\n"
    )
    resumed = Trainer(nn.Linear(1, 1), second, regression_step)
    resumed.fit(make_loader(), make_loader())

    status = read_json(second.run_dir / "run.json")
    assert status["status"] == "completed"
    assert status["start_epoch"] == 3
    assert status["completed_epochs"] == 4
    assert status["best_epoch"] == resumed.checkpoint.best_epoch


def test_write_failure_keeps_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    状態を書けない場合も, 学習のエラーと割り込みがそのまま出ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        関数を差し替える仕組み

    Returns
    ----------
    None
    """

    def fail(path: Path, data: dict) -> None:
        """
        ディスクの不足を再現する関数

        Parameters
        ----------
        path: Path
            書き込むファイル
        data: dict
            書き込む内容

        Returns
        ----------
        None
        """
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("ml_found.training.status.write_json", fail)
    for error in (RuntimeError("壊れたバッチ"), KeyboardInterrupt()):
        experiment = make_experiment(
            tmp_path / type(error).__name__, "trainer:\n  epochs: 3\n"
        )

        def step(model: nn.Module, batch: tuple, phase: str) -> dict:
            """
            エポック 2 でエラーか割り込みを起こす関数

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
            # エポック 2 でエラーの発生
            if trainer.epoch == 2:
                raise error
            outputs = regression_step(model, batch, phase)

            return outputs

        trainer = Trainer(nn.Linear(1, 1), experiment, step)
        with pytest.raises(type(error)):
            trainer.fit(make_loader(), make_loader())


def test_resume_early_stopped_status(tmp_path: Path) -> None:
    """
    Early Stopping で止めた学習を再開した場合の run.json を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 損失が大きいほど良いとして, エポック 2 で止まる学習
    text = (
        "trainer:\n  epochs: 20\n  mode: max\n"
        "  early_stopping:\n    use: true\n    patience: 1\n"
    )
    first = make_experiment(tmp_path / "first", text)
    trainer = Trainer(nn.Linear(1, 1), first, regression_step)
    trainer.fit(make_loader(), make_loader())
    assert len(trainer.history.records) == 2

    # 再開しても学習はせず, 止めたエポックとベストを引き継いで完了
    second = make_experiment(
        tmp_path / "second",
        text + f"  resume_from: {first.run_dir / 'last.pt'}\n"
    )
    resumed = Trainer(nn.Linear(1, 1), second, regression_step)
    resumed.fit(make_loader(), make_loader())
    status = read_json(second.run_dir / "run.json")
    assert status["status"] == "completed"
    assert status["early_stopped_epoch"] == 2
    assert status["completed_epochs"] == 2
    assert status["best_epoch"] == trainer.checkpoint.best_epoch
    assert status["best_value"] == trainer.checkpoint.best_value
