import inspect
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from torch import optim

from ml_found.training.optim import (
    resolve_optimizer_class, resolve_scheduler_class
)

# 学習の結果ではなく実装の都合で使う引数 (画面では詳細設定として扱う)
ADVANCED = {
    "foreach", "capturable", "differentiable", "fused", "last_epoch",
    "verbose", "maximize"
}

# Optimizer の説明 (無いものは引数だけを返す)
OPTIMIZER_NOTES = {
    "Adam": "勾配の 1 次と 2 次のモーメントで学習率を調整する. 多くの問題の最初の選択肢",
    "AdamW": "Adam の重み減衰を勾配から切り離したもの. Transformer などで標準的",
    "SGD": "確率的勾配降下法. momentum と組み合わせて画像分類などで使う",
    "RMSprop": "勾配の 2 乗の移動平均で学習率を調整する",
    "LBFGS": "準ニュートン法. 小さなデータのフルバッチ学習向け (closure を自動で渡す)",
    "Adagrad": "引数ごとに過去の勾配の 2 乗の和で学習率を下げる",
    "Adadelta": "Adagrad の学習率が下がり続ける問題を移動平均で抑えたもの",
    "Adamax": "Adam の 2 次のモーメントを無限ノルムにしたもの",
    "NAdam": "Adam に Nesterov の加速を加えたもの",
    "RAdam": "学習の初めの Adam の分散を補正したもの",
    "ASGD": "重みの平均を取る確率的勾配降下法",
    "Rprop": "勾配の符号だけで更新するフルバッチ向けの方法",
    "Adafactor": "2 次のモーメントを分解して省メモリにしたもの"
}

# そのままでは ML-Found で使えない Optimizer と理由
UNSUPPORTED = {
    "Muon": "2 次元の重みだけを受け付けるので, 他の Optimizer と組み合わせが必要",
    "SparseAdam": "疎な勾配 (nn.Embedding の sparse=True など) だけを受け付ける"
}

# Scheduler の説明
SCHEDULER_NOTES = {
    "StepLR": "step_size エポックごとに学習率を gamma 倍にする",
    "MultiStepLR": "milestones のエポックで学習率を gamma 倍にする",
    "ExponentialLR": "毎エポック学習率を gamma 倍にする",
    "CosineAnnealingLR": "T_max エポックかけて学習率を余弦の形で eta_min まで下げる",
    "CosineAnnealingWarmRestarts": "余弦の形で下げて T_0 ごとに戻す",
    "ReduceLROnPlateau": "監視する指標が改善しなくなったら学習率を下げる",
    "OneCycleLR": "1 周期で学習率を上げてから下げる (scheduler.interval は step)",
    "LinearLR": "total_iters かけて学習率の倍率を線形に変える",
    "ConstantLR": "total_iters の間だけ学習率を factor 倍にする",
    "CyclicLR": "base_lr と max_lr の間を周期的に動かす (interval は step)",
    "PolynomialLR": "total_iters かけて学習率を多項式の形で下げる"
}

# 引数の説明と範囲 (名前ごと, 全てのクラスで共通)
PARAM_NOTES: dict[str, dict[str, Any]] = {
    "lr": {"description": "学習率", "minimum": 0.0},
    "weight_decay": {"description": "重み減衰 (L2 正則化の強さ)", "minimum": 0.0},
    "momentum": {"description": "慣性の強さ", "minimum": 0.0, "maximum": 1.0},
    "betas": {"description": "1 次と 2 次のモーメントの減衰率"},
    "eps": {"description": "0 除算を避ける小さな値", "minimum": 0.0},
    "nesterov": {"description": "Nesterov の加速を使うか"},
    "amsgrad": {"description": "AMSGrad の補正を使うか"},
    "gamma": {"description": "学習率に掛ける倍率", "minimum": 0.0},
    "step_size": {"description": "学習率を下げる間隔 (エポック)", "minimum": 1},
    "milestones": {"description": "学習率を下げるエポックの一覧"},
    "T_max": {"description": "学習率を下げきるまでのエポック数", "minimum": 1},
    "eta_min": {"description": "学習率の下限", "minimum": 0.0},
    "factor": {"description": "学習率に掛ける倍率", "minimum": 0.0},
    "patience": {"description": "改善しないのを待つエポック数", "minimum": 0},
    "min_lr": {"description": "学習率の下限", "minimum": 0.0},
    "max_lr": {"description": "学習率の最大値", "minimum": 0.0},
    "mode": {"description": "監視する指標の向き (min か max)"},
    "max_iter": {"description": "1 ステップの中の最大の反復回数", "minimum": 1},
    "history_size": {"description": "保存する過去の勾配の数", "minimum": 1}
}


@dataclass
class ParamSpec:
    name: str
    kind: str
    default: Any
    required: bool
    advanced: bool
    description: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass
class ClassSpec:
    name: str
    description: str
    params: list[ParamSpec] = field(default_factory=list)
    supported: bool = True
    note: str = ""


def to_plain(value: Any) -> tuple[str, Any]:
    """
    引数の既定値を, 種類と JSON で表せる値に変換する関数

    Parameters
    ----------
    value: Any
        signature から取得した既定値

    Returns
    ----------
    kind: str
        bool, int, float, str, list, none, other のどれか
    plain: Any
        JSON で表せる値, 表せない場合は文字列
    """
    if value is None:
        return "none", None
    if isinstance(value, bool):
        return "bool", value
    if isinstance(value, int):
        return "int", value
    if isinstance(value, float):
        return "float", value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return "str", value
    if isinstance(value, (tuple, list)):
        items = [to_plain(item)[1] for item in value]
        return "list", items

    return "other", repr(value)


def describe_params(cls: type, skip: set[str]) -> list[ParamSpec]:
    """
    クラスの初期化の引数を仕様として取り出す関数

    Parameters
    ----------
    cls: type
        Optimizer か Scheduler のクラス
    skip: set[str]
        ML-Found が渡すので仕様に含めない引数 (params, optimizer)

    Returns
    ----------
    specs: list[ParamSpec]
        引数の名前, 種類, 既定値, 必須か, 説明
    """
    specs = []
    for param in inspect.signature(cls.__init__).parameters.values():
        # self, ML-Found が渡す引数, 可変長の引数は除外
        if param.name == "self" or param.name in skip:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        required = param.default is inspect.Parameter.empty
        kind, default = ("required", None) if required else to_plain(
            param.default
        )
        notes = PARAM_NOTES.get(param.name, {})
        specs.append(
            ParamSpec(
                name=param.name, kind=kind, default=default,
                required=required, advanced=param.name in ADVANCED,
                description=notes.get("description", ""),
                minimum=notes.get("minimum"), maximum=notes.get("maximum")
            )
        )

    return specs


def list_classes(module: Any, base: type) -> list[str]:
    """
    モジュールから基底クラスを継承したクラスの名前を列挙する関数

    Parameters
    ----------
    module: Any
        torch.optim か torch.optim.lr_scheduler
    base: type
        Optimizer か LRScheduler

    Returns
    ----------
    names: list[str]
        クラスの名前, 基底クラスと内部のクラスは除外
    """
    names = sorted(
        name for name, value in vars(module).items()
        if isinstance(value, type) and issubclass(value, base)
        and value is not base and not name.startswith("_")
    )

    return names


def describe_optimizers() -> list[dict[str, Any]]:
    """
    選べる Optimizer と引数の仕様を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    specs: list[dict[str, Any]]
        torch.optim の Optimizer ごとの名前, 説明, 引数, 使えるか
    """
    specs = []
    for name in list_classes(optim, optim.Optimizer):
        spec = ClassSpec(
            name=name, description=OPTIMIZER_NOTES.get(name, ""),
            params=describe_params(getattr(optim, name), {"params"}),
            supported=name not in UNSUPPORTED,
            note=UNSUPPORTED.get(name, "")
        )
        specs.append(asdict(spec))

    return specs


def describe_schedulers() -> list[dict[str, Any]]:
    """
    選べる Scheduler と引数の仕様を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    specs: list[dict[str, Any]]
        torch.optim.lr_scheduler の Scheduler ごとの名前, 説明, 引数
    """
    module = optim.lr_scheduler
    bases = (module.LRScheduler, module.ReduceLROnPlateau)
    names = sorted(
        set(list_classes(module, bases[0])) | {"ReduceLROnPlateau"}
    )

    # 他の Scheduler を組み合わせるものは config だけでは作れないので除外
    composite = {"SequentialLR", "ChainedScheduler", "LambdaLR",
                 "MultiplicativeLR"}
    specs = []
    for name in names:
        if name in composite or name.startswith("_"):
            continue
        spec = ClassSpec(
            name=name, description=SCHEDULER_NOTES.get(name, ""),
            params=describe_params(getattr(module, name), {"optimizer"})
        )
        specs.append(asdict(spec))

    return specs


def check_params(cls: type, params: dict[str, Any], skip: set[str]) -> None:
    """
    config の引数が, クラスの引数として正しいかを確かめる関数

    Parameters
    ----------
    cls: type
        Optimizer か Scheduler のクラス
    params: dict[str, Any]
        config の params
    skip: set[str]
        ML-Found が渡す引数

    Returns
    ----------
    None
    """
    # 任意の名前の引数を受け取るクラス (自作のものなど) は確認しない
    signature = inspect.signature(cls.__init__)
    if any(
        param.kind == param.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return

    # 存在しない名前の引数 (typo など) の確認
    specs = describe_params(cls, skip)
    names = {spec.name for spec in specs}
    unknown = sorted(set(params) - names)
    if unknown:
        raise ValueError(
            f"{cls.__name__} に無い引数です: {', '.join(unknown)} "
            f"(使える引数: {', '.join(sorted(names))})"
        )

    # 必須の引数の確認
    missing = [
        spec.name for spec in specs
        if spec.required and spec.name not in params
    ]
    if missing:
        raise ValueError(
            f"{cls.__name__} の必須の引数がありません: {', '.join(missing)}"
        )


def check_optim_params(cfg: Any) -> None:
    """
    config の Optimizer と Scheduler の引数を確かめる関数

    Parameters
    ----------
    cfg: Any
        学習に使う config (optimizer と scheduler の項目を持つもの)

    Returns
    ----------
    None
    """
    optimizer_cls = resolve_optimizer_class(cfg.optimizer.name)
    check_params(optimizer_cls, dict(cfg.optimizer.params), {"params"})

    # ReduceLROnPlateau の mode は ML-Found が trainer.mode から補うので省略可
    if cfg.scheduler.name is not None:
        scheduler_cls = resolve_scheduler_class(cfg.scheduler.name)
        check_params(scheduler_cls, dict(cfg.scheduler.params), {"optimizer"})
