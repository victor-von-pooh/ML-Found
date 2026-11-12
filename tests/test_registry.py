from pathlib import Path

import pytest

from ml_found.training.registry import (
    describe_optimizers, describe_schedulers
)
from test_trainer import make_experiment


def test_describe_optimizers() -> None:
    """
    Optimizer の仕様に名前, 引数, 使えるかが含まれることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    specs = {spec["name"]: spec for spec in describe_optimizers()}

    # Adam の学習率は既定値と説明と範囲を持ち, 実装の都合の引数は詳細設定
    adam = {param["name"]: param for param in specs["Adam"]["params"]}
    assert adam["lr"]["kind"] == "float"
    assert adam["lr"]["default"] == 0.001
    assert adam["lr"]["minimum"] == 0.0
    assert adam["betas"]["kind"] == "list"
    assert adam["betas"]["default"] == [0.9, 0.999]
    assert adam["foreach"]["advanced"] is True
    assert "params" not in adam

    # ML-Found でそのままでは使えないものは理由付き
    assert specs["SGD"]["supported"] is True
    assert specs["SparseAdam"]["supported"] is False
    assert specs["SparseAdam"]["note"]
    assert "Optimizer" not in specs


def test_describe_schedulers() -> None:
    """
    Scheduler の仕様に必須の引数が含まれることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    specs = {spec["name"]: spec for spec in describe_schedulers()}
    step = {param["name"]: param for param in specs["StepLR"]["params"]}
    assert step["step_size"]["required"] is True
    assert step["step_size"]["kind"] == "required"
    assert "optimizer" not in step
    assert "ReduceLROnPlateau" in specs
    assert "SequentialLR" not in specs


def test_param_typos_are_errors(tmp_path: Path) -> None:
    """
    引数の名前の誤りと必須の引数の不足が, 出力先を作る前にエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    with pytest.raises(ValueError, match="無い引数です: lerning_rate"):
        make_experiment(
            tmp_path / "typo",
            "optimizer:\n  name: Adam\n  params:\n    lerning_rate: 0.1\n"
        )
    with pytest.raises(ValueError, match="必須の引数がありません: step_size"):
        make_experiment(
            tmp_path / "missing",
            "scheduler:\n  name: StepLR\n  params: {}\n"
        )
    assert not (tmp_path / "typo" / "outputs").exists()

    # 正しい引数と, ReduceLROnPlateau の mode の省略は通る
    make_experiment(
        tmp_path / "ok",
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n    momentum: 0.9\n"
        "scheduler:\n  name: ReduceLROnPlateau\n  params:\n    patience: 2\n"
    )
