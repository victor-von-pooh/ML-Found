import inspect
import json
from pathlib import Path
from typing import Any, Optional, Union

import pytest
import torch
from torch import Tensor, optim

from ml_found.training.registry import (
    ParamSpec,
    describe_optimizers,
    describe_range,
    describe_schedulers,
    is_in_range,
    refine_kind,
    register_optimizer,
    unregister_optimizer,
)
from ml_found.training.trainer import Trainer
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
    OneCycleLR の総ステップ数の指定は, 無くても学習を始められることを確かめる関数
    (Trainer が学習用データのバッチ数から補う)

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    base = "scheduler:\n  name: OneCycleLR\n  interval: step\n  params:\n"
    make_experiment(tmp_path / "none", base + "    max_lr: 0.1\n")
    with pytest.raises(ValueError, match="必須の引数がありません: max_lr"):
        make_experiment(tmp_path / "no_max", base + "    total_steps: 10\n")
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


# 範囲の端を含まない引数. 端の値では PyTorch が例外を出す
# 登録簿の exclusive_minimum と exclusive_maximum がこれと一致することも確認
EXCLUSIVE_BOUNDS = {
    ("LinearLR", "start_factor", "minimum"),
    ("ReduceLROnPlateau", "factor", "maximum"),
    ("OneCycleLR", "div_factor", "minimum"),
    ("OneCycleLR", "final_div_factor", "minimum"),
}

# Scheduler を作るのに必要な引数
SCHEDULER_KWARGS: dict[str, dict[str, Any]] = {
    "StepLR": {"step_size": 1},
    "MultiStepLR": {"milestones": [1]},
    "CosineAnnealingLR": {"T_max": 2},
    "ConstantLR": {},
    "LinearLR": {},
    "ReduceLROnPlateau": {},
    "CosineAnnealingWarmRestarts": {"T_0": 1},
    "OneCycleLR": {"max_lr": 0.1, "total_steps": 10},
    "CyclicLR": {"base_lr": 0.01, "max_lr": 0.1},
    "PolynomialLR": {}
}


def bound_cases() -> list[tuple[str, str, str, Any]]:
    """
    Scheduler の範囲を持つ引数の, 端の値の組を列挙する関数

    Parameters
    ----------
    None

    Returns
    ----------
    cases: list[tuple[str, str, str, Any]]
        クラスの名前, 引数の名前, minimum か maximum, 端の値
    """
    # 登録簿の Scheduler の仕様から, 範囲を持つ引数を集める
    cases = []
    for spec in describe_schedulers():
        for param in spec["params"]:
            for side in ("minimum", "maximum"):
                if param[side] is not None:
                    cases.append(
                        (spec["name"], param["name"], side, param[side])
                    )

    return cases


def test_exclusive_flags() -> None:
    """
    登録簿の端を含まない範囲の指定が, 想定の一覧と一致することを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    flagged = {
        (spec["name"], param["name"], side)
        for spec in describe_schedulers()
        for param in spec["params"]
        for side in ("minimum", "maximum")
        if param[f"exclusive_{side}"]
    }
    assert flagged == EXCLUSIVE_BOUNDS

    # Optimizer では betas の上限だけが端を含まない (PyTorch も 1 未満を要求)
    optimizer_flags = {
        (param["name"], side)
        for spec in describe_optimizers()
        for param in spec["params"]
        for side in ("minimum", "maximum")
        if param[f"exclusive_{side}"]
    }
    assert optimizer_flags == {("betas", "maximum")}


@pytest.mark.parametrize(("name", "param", "side", "value"), bound_cases())
def test_scheduler_bounds_match_torch(
    name: str, param: str, side: str, value: Any
) -> None:
    """
    登録簿の範囲の端の値で, PyTorch が Scheduler を作れるかを確かめる関数

    端を含む範囲は作れること, 端を含まない範囲は例外になることを確かめ,
    PyTorch を更新したときの範囲のずれを検出する

    Parameters
    ----------
    name: str
        Scheduler のクラスの名前
    param: str
        引数の名前
    side: str
        minimum か maximum
    value: Any
        端の値

    Returns
    ----------
    None
    """
    # 端の値を入れて Scheduler を作成 (epochs などは total_steps の代わり)
    kwargs = dict(SCHEDULER_KWARGS.get(name, {}))
    if name == "OneCycleLR" and param in ("epochs", "steps_per_epoch"):
        kwargs.update(total_steps=None, epochs=1, steps_per_epoch=1)
    kwargs[param] = value
    model = torch.nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    cls = getattr(optim.lr_scheduler, name)

    # 作るだけでなく数回進め, 学習中の 0 除算も検出 (OneCycleLR は全ステップ数まで)
    total = kwargs.get("total_steps") or (
        kwargs.get("epochs", 3) * kwargs.get("steps_per_epoch", 1)
    )
    steps = min(3, total - 1) if name == "OneCycleLR" else 3
    metric = [0.0] if name == "ReduceLROnPlateau" else []

    def run() -> None:
        scheduler = cls(optimizer, **kwargs)
        for _ in range(steps):
            optimizer.step()
            scheduler.step(*metric)

    if (name, param, side) in EXCLUSIVE_BOUNDS:
        with pytest.raises((ValueError, ZeroDivisionError)):
            run()
    else:
        run()


def test_param_values_out_of_range(tmp_path: Path) -> None:
    """
    範囲の外の値が, 出力先を作る前にエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    with pytest.raises(ValueError, match=r"lr は 0.0 以上 \(今は -0.1\)"):
        make_experiment(
            tmp_path / "negative",
            "optimizer:\n  name: Adam\n  params:\n    lr: -0.1\n"
        )
    assert not (tmp_path / "negative" / "outputs").exists()

    # 端を含まない範囲は, 端の値もエラー
    with pytest.raises(ValueError, match="factor は 0.0 以上 1.0 未満"):
        make_experiment(
            tmp_path / "factor",
            "scheduler:\n  name: ReduceLROnPlateau\n  params:\n"
            "    factor: 1.0\n"
        )
    with pytest.raises(ValueError, match="start_factor は 0.0 より大きく"):
        make_experiment(
            tmp_path / "start",
            "scheduler:\n  name: LinearLR\n  params:\n"
            "    start_factor: 0\n"
        )

    # 複数の範囲外はまとめて表示
    with pytest.raises(ValueError, match="momentum .*weight_decay"):
        make_experiment(
            tmp_path / "many",
            "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
            "    momentum: 1.5\n    weight_decay: -1\n"
        )

    # 補間は解決した値で確認
    with pytest.raises(ValueError, match=r"patience は 0 以上 \(今は -3\)"):
        make_experiment(
            tmp_path / "interpolation",
            "wait: -3\n"
            "scheduler:\n  name: ReduceLROnPlateau\n  params:\n"
            "    patience: ${wait}\n"
        )

    # 引用符で囲んだ数, 真偽値, 無限大は誤り
    with pytest.raises(ValueError, match="lr は数で書いてください"):
        make_experiment(
            tmp_path / "quoted",
            "optimizer:\n  name: Adam\n  params:\n    lr: '0.001'\n"
        )
    with pytest.raises(ValueError, match="lr は数で書いてください"):
        make_experiment(
            tmp_path / "bool",
            "optimizer:\n  name: Adam\n  params:\n    lr: true\n"
        )
    with pytest.raises(ValueError, match=r"lr は 0.0 以上 \(今は inf\)"):
        make_experiment(
            tmp_path / "inf",
            "optimizer:\n  name: Adam\n  params:\n    lr: .inf\n"
        )

    # 端を含む範囲の端の値, 範囲を持たない引数, 範囲内の補間の値, 配列は通る
    make_experiment(
        tmp_path / "ok",
        "optimizer:\n  name: SGD\n  params:\n    lr: 0\n    momentum: 1.0\n"
        "    dampening: -5\n"
        "scheduler:\n  name: ReduceLROnPlateau\n  params:\n"
        "    factor: 0.0\n    patience: ${trainer.epochs}\n"
        "    min_lr: [0.0]\n"
    )


class CustomLinearLR(optim.lr_scheduler.LinearLR):
    def __init__(
        self, optimizer: optim.Optimizer, start_factor: float = 0.5,
        end_factor: float = 1.0, total_iters: int = 5
    ):
        # 親クラスのコンストラクタを呼び出す (start_factor は 0 より大きく補正)
        super(CustomLinearLR, self).__init__(
            optimizer, max(start_factor, 1e-8), end_factor, total_iters
        )


def test_custom_classes_skip_ranges(tmp_path: Path) -> None:
    """
    自作のクラスには, 登録簿の範囲を当てはめないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 同じ名前の引数 (momentum) や同じ親クラスでも, 自作のものは対象外
    # (SGD の範囲では momentum は 1 以下)
    make_experiment(
        tmp_path / "optimizer",
        "optimizer:\n  name: test_registry.FlexibleSGD\n  params:\n"
        "    momentum: 1.5\n"
    )
    make_experiment(
        tmp_path / "scheduler",
        "scheduler:\n  name: test_registry.CustomLinearLR\n  params:\n"
        "    start_factor: 0.0\n"
    )


def test_is_in_range() -> None:
    """
    範囲の判定と説明の文を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    spec = ParamSpec(
        name="x", kind="float", default=0.5, required=False, advanced=False,
        minimum=0.0, maximum=1.0, exclusive_minimum=True
    )
    assert not is_in_range(spec, 0.0)
    assert is_in_range(spec, 1e-9)
    assert is_in_range(spec, 1.0)
    assert not is_in_range(spec, 1.0001)
    assert not is_in_range(spec, float("nan"))
    assert describe_range(spec) == "0.0 より大きく 1.0 以下"
    spec.maximum = None
    assert describe_range(spec) == "0.0 より大きい"

    unbounded = ParamSpec(
        name="y", kind="float", default=0.0, required=False, advanced=False
    )
    assert is_in_range(unbounded, -1e9)


def test_array_values_and_init_errors(tmp_path: Path) -> None:
    """
    配列の値も要素ごとに範囲を確かめ, Trainer の作成のエラーを記録する
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # betas の要素が範囲の外なら, 出力先を作る前にエラー
    message = r"betas は 0.0 以上 1.0 未満 \(今は 1.5\)"
    with pytest.raises(ValueError, match=message):
        make_experiment(
            tmp_path / "betas",
            "optimizer:\n  name: Adam\n  params:\n    betas: [1.5, 0.999]\n"
        )
    assert not (tmp_path / "betas" / "outputs").exists()

    # 配列の中の ??? と解決できない補間は, スカラーと同じく確かめない
    make_experiment(
        tmp_path / "missing",
        "optimizer:\n  name: Adam\n  params:\n    betas: ['???', 0.9]\n"
    )

    # PyTorch も betas は 0 以上 1 未満を求める
    for name in ("Adam", "AdamW", "Adamax", "NAdam", "RAdam"):
        cls = getattr(optim, name)
        cls(torch.nn.Linear(1, 1).parameters(), betas=(0.0, 0.0))
        with pytest.raises(ValueError):
            cls(torch.nn.Linear(1, 1).parameters(), betas=(0.9, 1.0))

    # 範囲の確認から外れる誤り (自作の Optimizer の引数) も, 仮のパラメータで
    # 作ってみて出力先を作る前にエラー
    with pytest.raises(ValueError, match="RenamedSGD を作れません"):
        make_experiment(
            tmp_path / "custom",
            "optimizer:\n  name: test_registry.RenamedSGD\n  params:\n"
            "    lr: -0.1\n"
        )
    assert not (tmp_path / "custom" / "outputs").exists()

    # それ以外の Trainer の作成のエラー (モデルを移せないなど) は,
    # train.log と run.json に残る
    class BrokenModel(torch.nn.Linear):
        # デバイスに移せないモデル
        def to(self, *args: Any, **kwargs: Any) -> torch.nn.Module:
            """
            常にエラーにする関数

            Parameters
            ----------
            *args: Any
                移す先
            **kwargs: Any
                移す先

            Returns
            ----------
            model: torch.nn.Module
                返さない
            """
            raise ValueError("このモデルは移せません")

    experiment = make_experiment(tmp_path / "init")
    with pytest.raises(ValueError):
        Trainer(BrokenModel(1, 1), experiment, lambda m, b, p: {})
    status = json.loads(
        (experiment.run_dir / "run.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"
    assert "移せません" in status["error"]
    log = (experiment.run_dir / "train.log").read_text(encoding="utf-8")
    assert "Trainer の作成でエラーが起きました" in log


def test_try_build_optim(tmp_path: Path) -> None:
    """
    範囲を持たない引数の誤りも, 仮のパラメータで作ってみて出力先を作る前に
    エラーにすることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 範囲を持たない引数の組み合わせの誤り (SGD の nesterov には momentum)
    cases = {
        "sgd": (
            "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
            "    nesterov: true\n",
            "SGD を作れません"
        ),
        "cyclic": (
            "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
            "scheduler:\n  name: CyclicLR\n  interval: step\n  params:\n"
            "    base_lr: 0.01\n    max_lr: 0.1\n    mode: wrong\n",
            "CyclicLR を作れません"
        ),
    }
    for name, (text, message) in cases.items():
        with pytest.raises(ValueError, match=message):
            make_experiment(tmp_path / name, text)
        assert not (tmp_path / name / "outputs").exists()

    # 原因が trainer の値の場合は, trainer の誤りとしてエラー
    for name, text in (
        ("epochs", "trainer:\n  epochs: 0\n"),
        ("mode", "trainer:\n  mode: minimize\n"),
    ):
        with pytest.raises(ValueError, match=f"trainer.{name} は"):
            make_experiment(tmp_path / f"trainer_{name}", text)

    # データの量で決まる引数を書かない OneCycleLR は, 仮の値で補って確認
    make_experiment(
        tmp_path / "one_cycle",
        "scheduler:\n  name: OneCycleLR\n  interval: step\n  params:\n"
        "    max_lr: 0.1\n"
    )


class ProjectSGD(optim.SGD):
    def __init__(self, params: Any, lr: float = 0.1, momentum: float = 0.0):
        # 親クラスのコンストラクタを呼び出す
        super(ProjectSGD, self).__init__(params, lr=lr, momentum=momentum)


class KwargsSGD(optim.SGD):
    def __init__(self, params: Any, lr: float = 0.1, **kwargs: Any):
        # 親クラスのコンストラクタを呼び出す
        super(KwargsSGD, self).__init__(params, lr=lr, **kwargs)


def test_register_optimizer(tmp_path: Path) -> None:
    """
    プロジェクトの Optimizer を登録すると, 名前だけで使え, 登録簿と範囲の
    確認に加わることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 説明と範囲付きで登録
    register_optimizer(
        ProjectSGD, description="プロジェクトの SGD",
        param_notes={"lr": {"description": "学習率", "minimum": 0.0}}
    )
    try:
        # 名前だけで config に書いて学習を始められる
        make_experiment(
            tmp_path / "ok",
            "optimizer:\n  name: ProjectSGD\n  params:\n    lr: 0.05\n"
        )

        # 渡した範囲は出力先を作る前に確かめる (渡さない引数は確かめない)
        with pytest.raises(ValueError, match="lr は 0.0 以上"):
            make_experiment(
                tmp_path / "negative",
                "optimizer:\n  name: ProjectSGD\n  params:\n    lr: -1.0\n"
            )
        make_experiment(
            tmp_path / "momentum",
            "optimizer:\n  name: ProjectSGD\n  params:\n    momentum: 1.5\n"
        )

        # 登録簿に説明と引数の仕様が加わる
        specs = {spec["name"]: spec for spec in describe_optimizers()}
        assert specs["ProjectSGD"]["description"] == "プロジェクトの SGD"
        params = {
            param["name"]: param for param in specs["ProjectSGD"]["params"]
        }
        assert params["lr"]["minimum"] == 0.0
        assert params["momentum"]["description"] == ""
    finally:
        unregister_optimizer("ProjectSGD")

    # 登録を外すと名前だけでは使えない, torch の名前と無い引数はエラー
    with pytest.raises(ValueError):
        make_experiment(
            tmp_path / "removed", "optimizer:\n  name: ProjectSGD\n"
        )
    with pytest.raises(ValueError, match="torch.optim"):
        register_optimizer(ProjectSGD, name="SGD")
    with pytest.raises(ValueError, match="引数"):
        register_optimizer(ProjectSGD, param_notes={"beta": {}})
    with pytest.raises(TypeError):
        register_optimizer(int)

    # torch のクラス, 同じクラスの別の名前, 書き方の誤った説明, 確かめられない
    # クラスへの範囲はエラー (torch のクラスの範囲を消さないため)
    with pytest.raises(ValueError, match="torch のクラス"):
        register_optimizer(optim.SGD, name="MySGD")
    register_optimizer(ProjectSGD)
    try:
        with pytest.raises(ValueError, match="登録済み"):
            register_optimizer(ProjectSGD, name="OtherSGD")
    finally:
        unregister_optimizer("ProjectSGD")
    for notes in ({"lr": {"minimun": 0.0}}, {"lr": 0.0}):
        with pytest.raises(ValueError, match="辞書"):
            register_optimizer(ProjectSGD, param_notes=notes)
    with pytest.raises(ValueError, match="範囲を確かめられません"):
        register_optimizer(
            KwargsSGD, param_notes={"lr": {"minimum": 0.0}}
        )
    specs = {spec["name"]: spec for spec in describe_optimizers()}
    sgd_lr = next(
        param for param in specs["SGD"]["params"] if param["name"] == "lr"
    )
    assert sgd_lr["minimum"] == 0.0
