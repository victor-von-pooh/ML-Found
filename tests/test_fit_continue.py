import csv
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from ml_found.training.trainer import Trainer

from test_trainer import (
    make_experiment, make_loader, regression_step, scripted_step
)


def read_history(path: Path) -> list[dict]:
    """
    history.csv を読む関数

    Parameters
    ----------
    path: Path
        実験のディレクトリ

    Returns
    ----------
    rows: list[dict]
        エポックごとの記録
    """
    # CSV を辞書のリストとして読み込み
    with (path / "history.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    return rows


def test_fit_again_continues(tmp_path: Path) -> None:
    """
    fit を呼び直すと, エポックの番号と記録が続き, 通して学習した場合と
    同じ結果になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 3 エポックを 2 回と, 6 エポックを 1 回
    text = (
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
        "scheduler:\n  name: StepLR\n  params:\n    step_size: 2\n"
        "    gamma: 0.5\n"
    )
    torch.manual_seed(0)
    twice = make_experiment(
        tmp_path / "twice", text + "trainer:\n  epochs: 3\n"
    )
    trainer = Trainer(nn.Linear(1, 1), twice, regression_step)
    trainer.fit(make_loader(), make_loader())
    trainer.fit(make_loader(), make_loader())

    torch.manual_seed(0)
    once = make_experiment(tmp_path / "once", text + "trainer:\n  epochs: 6\n")
    Trainer(nn.Linear(1, 1), once, regression_step).fit(
        make_loader(), make_loader()
    )

    # エポックの番号が続き, 値も学習率も通した場合と一致
    rows = read_history(twice.run_dir)
    assert [row["epoch"] for row in rows] == ["1", "2", "3", "4", "5", "6"]
    expected = read_history(once.run_dir)
    for row, other in zip(rows, expected):
        assert float(row["valid/loss"]) == pytest.approx(
            float(other["valid/loss"])
        )
        assert float(row["lr"]) == pytest.approx(float(other["lr"]))

    # 外から読める状態も続きの範囲
    status = json.loads((twice.run_dir / "run.json").read_text())
    assert status["planned_epochs"] == 6
    assert status["start_epoch"] == 4
    assert status["completed_epochs"] == 6
    last = torch.load(twice.run_dir / "last.pt", weights_only=True)
    assert last["epoch"] == 6


def test_fit_again_from_last_weights(tmp_path: Path) -> None:
    """
    fit の終わりにベストモデルの重みに戻した場合も, 呼び直した fit は
    最後のエポックの重みから続けることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 検証の損失がエポック 1 で最小になり, その後は悪くなる設定
    # (学習の損失は train と valid の順に 1 つずつ消費)
    losses = [1.0, 0.1, 1.0, 0.5, 1.0, 0.9] + [1.0, 1.0] * 3
    experiment = make_experiment(
        tmp_path,
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
        "trainer:\n  epochs: 3\n"
    )
    model = nn.Linear(1, 1)
    trainer = Trainer(model, experiment, scripted_step(losses))
    loader = make_loader(n=4, batch_size=4)
    trainer.fit(loader, loader)

    # ベストモデルの重みと区別できるように, 最後のエポックの重みを変更
    assert trainer.continue_weights is not None
    with torch.no_grad():
        for value in trainer.continue_weights.values():
            value.fill_(7.0)

    # 呼び直すと, 最後のエポックの重み (7) から学習を開始
    # (学習のフェーズを始める時点の重みを記録)
    seen = []
    original = trainer.on_epoch_start

    def record_weight(phase: str) -> None:
        """
        フェーズを始める時点のモデルの重みを記録する関数

        Parameters
        ----------
        phase: str
            train または valid

        Returns
        ----------
        None
        """
        if phase == "train":
            seen.append(model.weight.item())
        original(phase)

    trainer.on_epoch_start = record_weight  # type: ignore[method-assign]
    trainer.fit(loader, loader)
    assert seen[0] == pytest.approx(7.0)
    assert trainer.checkpoint.best_epoch == 1


def test_fit_again_after_early_stopping(tmp_path: Path) -> None:
    """
    Early Stopping で止めた後に fit を呼び直すと, 我慢の回数を数え直して
    学習を続けることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 学習率 0 で全く改善しない設定
    experiment = make_experiment(
        tmp_path,
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.0\n"
        "trainer:\n  epochs: 50\n"
        "  early_stopping:\n    use: true\n    patience: 3\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader())
    history = trainer.fit(make_loader())

    # 1 回目はエポック 4 で停止, 2 回目は 5 から 3 エポックで停止
    epochs = [record["epoch"] for record in history.records]
    assert epochs == [1, 2, 3, 4, 5, 6, 7]


def test_fit_again_warns_auto(tmp_path: Path) -> None:
    """
    auto で補った Scheduler の引数がある場合に, fit を呼び直すと
    予定の長さを超えると警告することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # T_max を auto にした設定
    experiment = make_experiment(
        tmp_path,
        "trainer:\n  epochs: 2\n"
        "scheduler:\n  name: CosineAnnealingLR\n  params:\n    T_max: auto\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader())
    with pytest.warns(UserWarning, match="最初の fit の長さ"):
        trainer.fit(make_loader())


def test_fit_again_onecycle_error(tmp_path: Path) -> None:
    """
    OneCycleLR の総ステップ数を超える続きの学習は, 始める前にエラーに
    することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 学習用データから総ステップ数を補う OneCycleLR
    experiment = make_experiment(
        tmp_path,
        "trainer:\n  epochs: 2\n"
        "scheduler:\n  name: OneCycleLR\n  interval: step\n"
        "  params:\n    max_lr: 0.01\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader())
    with pytest.raises(ValueError, match="fit を呼び直した"):
        trainer.fit(make_loader())


def test_fit_again_error_keeps_state(tmp_path: Path) -> None:
    """
    呼び直した fit が学習の前の確認でエラーになった場合は, モデルの重みと
    run.json の終えたエポック数とベストモデルを変えないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 総ステップ数を超えられない OneCycleLR で, 1 回目の後に呼び直す
    experiment = make_experiment(
        tmp_path,
        "trainer:\n  epochs: 2\n"
        "scheduler:\n  name: OneCycleLR\n  interval: step\n"
        "  params:\n    max_lr: 0.01\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader(), make_loader())
    before = {
        key: value.clone() for key, value in trainer.model.state_dict().items()
    }
    with pytest.raises(ValueError, match="fit を呼び直した"):
        trainer.fit(make_loader(), make_loader())

    # 重みはベストモデルのまま, run.json の記録も残る
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, before[key])
    status = json.loads((experiment.run_dir / "run.json").read_text())
    assert status["status"] == "failed"
    assert status["completed_epochs"] == 2
    assert status["best_epoch"] is not None


def test_fit_again_onecycle_counts_steps(tmp_path: Path) -> None:
    """
    呼び直した fit のバッチ数が前の fit と違っても, OneCycleLR の総ステップ数
    を進めた回数から確かめることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 1 回目は 4 バッチを 2 エポックで総ステップ数 8, 2 回目は 2 バッチ
    # (エポック数とバッチ数の掛け算なら 4 × 2 = 8 で, 超えることを見逃す)
    experiment = make_experiment(
        tmp_path,
        "trainer:\n  epochs: 2\n"
        "scheduler:\n  name: OneCycleLR\n  interval: step\n"
        "  params:\n    max_lr: 0.01\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader(n=64, batch_size=16))
    with pytest.raises(ValueError, match="fit を呼び直した"):
        trainer.fit(make_loader(n=32, batch_size=16))


def test_fit_again_after_interrupt(tmp_path: Path) -> None:
    """
    エポックの途中の割り込みで止まった後に fit を呼び直すと, 途中のエポック
    を捨てて続け, 通して学習した場合と同じ結果になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 学習のバッチの 6 回目 (エポック 2 の 2 バッチ目) で 1 回だけ割り込む step
    calls = {"train": 0}

    def interrupted_step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        決めた回で 1 回だけ割り込む平均二乗誤差の step

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
        if phase == "train":
            calls["train"] += 1
            if calls["train"] == 6:
                raise KeyboardInterrupt
        return regression_step(model, batch, phase)

    text = (
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
        "    momentum: 0.9\n"
        "trainer:\n  epochs: 3\n"
    )
    torch.manual_seed(0)
    stopped = make_experiment(tmp_path / "stopped", text)
    trainer = Trainer(nn.Linear(1, 1), stopped, interrupted_step)
    with pytest.raises(KeyboardInterrupt):
        trainer.fit(make_loader(), make_loader())
    trainer.fit(make_loader(), make_loader())

    torch.manual_seed(0)
    once = make_experiment(tmp_path / "once", text.replace("3", "4"))
    Trainer(nn.Linear(1, 1), once, regression_step).fit(
        make_loader(), make_loader()
    )

    # 1 回目はエポック 1 だけ終え, 続きはエポック 2 から 4
    rows = read_history(stopped.run_dir)
    assert [row["epoch"] for row in rows] == ["1", "2", "3", "4"]
    for row, other in zip(rows, read_history(once.run_dir)):
        assert float(row["valid/loss"]) == pytest.approx(
            float(other["valid/loss"])
        )


def test_fit_retry_before_training_is_first(tmp_path: Path) -> None:
    """
    最初の fit が学習を始める前に止まった場合は, 呼び直した fit を続きでは
    なく最初の fit として扱うことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 総ステップ数が 4 の OneCycleLR に, 4 バッチ × 2 エポックで学習の前に
    # エラー
    experiment = make_experiment(
        tmp_path,
        "trainer:\n  epochs: 2\n"
        "scheduler:\n  name: OneCycleLR\n  interval: step\n"
        "  params:\n    max_lr: 0.01\n    total_steps: 4\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    with pytest.raises(ValueError, match="総ステップ数"):
        trainer.fit(make_loader(n=64, batch_size=16))
    assert not trainer.loop_started

    # 2 バッチで呼び直すと, 最初の fit としてエポック 1 から 2 を学習
    history = trainer.fit(make_loader(n=32, batch_size=16))
    assert [record["epoch"] for record in history.records] == [1, 2]
    status = json.loads((experiment.run_dir / "run.json").read_text())
    assert status["start_epoch"] == 1
    assert status["planned_epochs"] == 2


def test_fit_again_removes_stop_and_sums_duration(tmp_path: Path) -> None:
    """
    呼び直した fit は前の fit で置いた STOP を消して学習を続け, run.json の
    所要時間を合計することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 1 回目は STOP でエポック 1 で止める
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 3\n")
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    stop = experiment.run_dir / "STOP"
    trainer.add_callback(lambda result: stop.touch() and None)
    trainer.fit(make_loader())
    first = json.loads((experiment.run_dir / "run.json").read_text())
    assert first["status"] == "stopped"
    trainer.callbacks.clear()

    # 呼び直すと STOP を消して予定の 3 エポックを学習
    history = trainer.fit(make_loader())
    assert not stop.exists()
    assert [record["epoch"] for record in history.records] == [1, 2, 3, 4]
    second = json.loads((experiment.run_dir / "run.json").read_text())
    assert second["status"] == "completed"
    assert second["started_at"] == first["started_at"]
    assert second["duration_sec"] >= first["duration_sec"]
