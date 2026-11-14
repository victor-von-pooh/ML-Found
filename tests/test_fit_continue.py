import csv
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from test_trainer import (
    make_experiment, make_loader, regression_step, scripted_step
)

from ml_found.training.trainer import Trainer


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
    twice = make_experiment(tmp_path / "twice", text + "trainer:\n  epochs: 3\n")
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

    # 最後のエポックの重みを変えて, ベストモデルの重みと区別できるようにする
    assert trainer.continue_weights is not None
    with torch.no_grad():
        for value in trainer.continue_weights.values():
            value.fill_(7.0)

    # 呼び直すと, 最後のエポックの重み (7) から学習を始める
    seen = []
    trainer.add_callback(
        lambda result: seen.append(model.weight.item()) and None
    )
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

    # 1 回目はエポック 4 で止まり, 2 回目は 5 から 3 エポックで止まる
    epochs = [record["epoch"] for record in history.records]
    assert epochs == [1, 2, 3, 4, 5, 6, 7]


def test_fit_again_warns_auto(tmp_path: Path) -> None:
    """
    auto で補った Scheduler の引数がある場合に, fit を呼び直すと
    予定の長さを超えることを警告する関数

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
