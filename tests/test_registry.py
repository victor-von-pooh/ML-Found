import inspect
from pathlib import Path
from typing import Any, Optional, Union

import pytest
from torch import Tensor, optim

from ml_found.training.registry import (
    describe_optimizers, describe_schedulers, refine_kind
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

    # 既定値が 0 や None でも, 型の注釈から float や int と判定
    sgd = {param["name"]: param for param in specs["SGD"]["params"]}
    assert sgd["momentum"]["kind"] == "float"
    assert sgd["momentum"]["default"] == 0
    assert adam["foreach"]["kind"] == "none"
    lbfgs = {param["name"]: param for param in specs["LBFGS"]["params"]}
    assert lbfgs["lr"]["kind"] == "float"
    assert lbfgs["max_iter"]["kind"] == "int"
    assert lbfgs["max_eval"]["kind"] == "int"

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
    # StepLR の step_size は必須で, 組み合わせる Scheduler は除外
    specs = {spec["name"]: spec for spec in describe_schedulers()}
    step = {param["name"]: param for param in specs["StepLR"]["params"]}
    assert step["step_size"]["required"] is True
    assert step["step_size"]["kind"] == "required"
    assert "optimizer" not in step
    assert "ReduceLROnPlateau" in specs
    assert "SequentialLR" not in specs

    # 注釈が文字列の場合も判定
    one_cycle = {
        param["name"]: param for param in specs["OneCycleLR"]["params"]
    }
    assert one_cycle["total_steps"]["kind"] == "int"
    assert one_cycle["total_steps"]["default"] is None
    plateau = {
        param["name"]: param
        for param in specs["ReduceLROnPlateau"]["params"]
    }
    assert plateau["min_lr"]["kind"] == "float"
    cyclic = {param["name"]: param for param in specs["CyclicLR"]["params"]}

    # PyTorch が初期化で確かめる範囲をクラスごとに持つ
    constant = {
        param["name"]: param for param in specs["ConstantLR"]["params"]
    }
    assert constant["factor"]["maximum"] == 1.0
    restarts = {
        param["name"]: param
        for param in specs["CosineAnnealingWarmRestarts"]["params"]
    }
    assert restarts["T_mult"]["minimum"] == 1
    assert plateau["factor"]["maximum"] == 1.0
    assert cyclic["scale_fn"]["kind"] == "none"


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


class RenamedSGD(optim.SGD):
    def __init__(self, model_params: Any, lr: float = 0.1):
        # 親クラスのコンストラクタを呼び出す
        super(RenamedSGD, self).__init__(model_params, lr=lr)


class FlexibleSGD(optim.SGD):
    def __init__(self, params: Any, **kwargs: Any):
        # 親クラスのコンストラクタを呼び出す
        super(FlexibleSGD, self).__init__(params, lr=kwargs.get("lr", 0.1))


def test_custom_classes(tmp_path: Path) -> None:
    """
    自作のクラスの引数の名前と, 任意の名前の引数を受け取るクラスを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # ML-Found が最初に渡す引数は名前に関係なく対象外
    make_experiment(
        tmp_path / "renamed",
        "optimizer:\n  name: test_registry.RenamedSGD\n  params:\n"
        "    lr: 0.2\n"
    )
    with pytest.raises(ValueError, match="無い引数です: momentum"):
        make_experiment(
            tmp_path / "renamed_typo",
            "optimizer:\n  name: test_registry.RenamedSGD\n  params:\n"
            "    momentum: 0.9\n"
        )

    # 任意の名前の引数を受け取るクラスは確認しない
    make_experiment(
        tmp_path / "flexible",
        "optimizer:\n  name: test_registry.FlexibleSGD\n  params:\n"
        "    anything: 1\n"
    )


def test_one_cycle_steps(tmp_path: Path) -> None:
    """
    OneCycleLR の総ステップ数の指定が無い場合にエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    base = "scheduler:\n  name: OneCycleLR\n  interval: step\n  params:\n"
    with pytest.raises(ValueError, match="total_steps"):
        make_experiment(tmp_path / "none", base + "    max_lr: 0.1\n")
    make_experiment(
        tmp_path / "steps", base + "    max_lr: 0.1\n    total_steps: 10\n"
    )
    make_experiment(
        tmp_path / "epochs",
        base + "    max_lr: 0.1\n    epochs: 2\n    steps_per_epoch: 5\n"
    )


@pytest.mark.parametrize(
    ("kind", "annotation", "expected"),
    [
        ("none", Optional[int], "int"),
        ("int", Union[float, Tensor], "float"),
        ("none", int | None, "int"),
        ("int", "list[float] | float", "float"),
        ("none", "Optional[float]", "float"),
        ("int", "float", "float"),
        ("none", "Callable[[float], float] | None", "none"),
        ("none", list[int] | None, "none"),
        ("none", "dict[str, float] | None", "none"),
        ("none", bool | None, "none"),
        ("int", inspect.Parameter.empty, "int"),
    ],
)
def test_refine_kind(kind: str, annotation: Any, expected: str) -> None:
    """
    型の注釈で引数の種類を補う規則を確かめる関数

    Parameters
    ----------
    kind: str
        既定値から決めた種類
    annotation: Any
        型の注釈
    expected: str
        期待する種類

    Returns
    ----------
    None
    """
    assert refine_kind(kind, annotation) == expected
