import inspect
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from omegaconf import DictConfig
from torch import optim

from ml_found.training.optim import (
    resolve_optimizer_class, resolve_scheduler_class
)

# 学習の結果ではなく実装の都合で使う引数 (画面では詳細設定として扱う)
ADVANCED = {
    "foreach", "capturable", "differentiable", "fused", "last_epoch",
    "verbose", "maximize"
}

# Optimizer の説明 (無いものは説明なしで, 引数だけを返す設定)
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
    "weight_decay": {"description": "重み減衰の強さ", "minimum": 0.0},
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

# 同じ名前でもクラスによって意味が違う引数の説明 (上の説明より優先)
CLASS_PARAM_NOTES: dict[tuple[str, str], dict[str, Any]] = {
    ("ReduceLROnPlateau", "eps"): {
        "description": "これより小さい学習率の変化は無視", "minimum": 0.0
    },
    ("ReduceLROnPlateau", "mode"): {
        "description": "監視する指標の向き. 省略すると ML-Found が trainer.mode を使用"
    },
    ("CyclicLR", "mode"): {
        "description": "周期の形 (triangular, triangular2, exp_range)"
    },
    ("Adafactor", "eps"): {"description": "0 除算を避ける小さな値の組"},
    ("OneCycleLR", "total_steps"): {
        "description": "全ステップ数. 省略する場合は epochs と steps_per_epoch が必要"
    }
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
    # 型ごとに種類を決め, JSON で表せない値は文字列に変換
    kind = "other"
    plain: Any = repr(value)
    if value is None:
        kind, plain = "none", None
    elif isinstance(value, bool):
        kind, plain = "bool", value
    elif isinstance(value, int):
        kind, plain = "int", value
    elif isinstance(value, float):
        kind, plain = "float", value if math.isfinite(value) else str(value)
    elif isinstance(value, str):
        kind, plain = "str", value
    elif isinstance(value, (tuple, list)):
        kind, plain = "list", [to_plain(item)[1] for item in value]

    return kind, plain


def describe_params(cls: type) -> list[ParamSpec]:
    """
    クラスの初期化の引数を仕様として取り出す関数

    Parameters
    ----------
    cls: type
        Optimizer か Scheduler のクラス

    Returns
    ----------
    specs: list[ParamSpec]
        引数の名前, 種類, 既定値, 必須か, 説明
    """
    # self と, ML-Found が最初の位置で渡す引数 (params か optimizer) は名前に
    # 関係なく除外し, 可変長の引数も除外
    parameters = list(inspect.signature(cls.__init__).parameters.values())[2:]
    specs = []
    for param in parameters:
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        required = param.default is inspect.Parameter.empty
        kind, default = ("required", None) if required else to_plain(
            param.default
        )
        notes = CLASS_PARAM_NOTES.get(
            (cls.__name__, param.name), PARAM_NOTES.get(param.name, {})
        )
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
            params=describe_params(getattr(optim, name)),
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

    # 他の Scheduler を組み合わせるものと, 関数を受け取るものは
    # config だけでは作れないので除外
    excluded = {
        "SequentialLR", "ChainedScheduler", "LambdaLR", "MultiplicativeLR"
    }
    specs = []
    for name in list_classes(module, module.LRScheduler):
        if name in excluded:
            continue
        spec = ClassSpec(
            name=name, description=SCHEDULER_NOTES.get(name, ""),
            params=describe_params(getattr(module, name))
        )
        specs.append(asdict(spec))

    return specs


def check_params(cls: type, names: set[str]) -> None:
    """
    config の引数の名前が, クラスの引数として正しいかを確かめる関数

    Parameters
    ----------
    cls: type
        Optimizer か Scheduler のクラス
    names: set[str]
        config の params に書かれた引数の名前

    Returns
    ----------
    None
    """
    # 引数を取り出せないクラスと, 任意の名前の引数や位置だけの引数を受け取る
    # クラス (自作のものなど) は確認の対象外
    try:
        parameters = list(inspect.signature(cls.__init__).parameters.values())
    except (TypeError, ValueError):
        return
    if any(
        param.kind in (param.VAR_KEYWORD, param.POSITIONAL_ONLY)
        for param in parameters[2:]
    ):
        return

    # 存在しない名前の引数 (typo など) の確認
    specs = describe_params(cls)
    valid = {spec.name for spec in specs}
    unknown = sorted(names - valid)
    if unknown:
        raise ValueError(
            f"{cls.__name__} に無い引数です: {', '.join(unknown)} "
            f"(使える引数: {', '.join(sorted(valid))})"
        )

    # 必須の引数の確認 (OneCycleLR は総ステップ数の指定方法が 2 通り)
    missing = [
        spec.name for spec in specs
        if spec.required and spec.name not in names
    ]
    if issubclass(cls, optim.lr_scheduler.OneCycleLR) and not (
        "total_steps" in names
        or {"epochs", "steps_per_epoch"} <= names
    ):
        missing.append("total_steps (または epochs と steps_per_epoch)")
    if missing:
        raise ValueError(
            f"{cls.__name__} の必須の引数がありません: {', '.join(missing)}"
        )


def param_names(params: Any, section: str) -> set[str]:
    """
    config の params から引数の名前を取り出す関数

    Parameters
    ----------
    params: Any
        optimizer.params か scheduler.params
    section: str
        エラーの文に使う項目名

    Returns
    ----------
    names: set[str]
        引数の名前 (値は読まないので, 補間や ??? があっても可)
    """
    if not isinstance(params, (dict, DictConfig)):
        raise ValueError(f"{section}.params は「引数の名前: 値」の形で書いてください")
    names = {str(key) for key in params.keys()}

    return names


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
    # 自分で作った Optimizer を Trainer に渡す場合も, config の引数は確認
    optimizer_cls = resolve_optimizer_class(cfg.optimizer.name)
    check_params(
        optimizer_cls, param_names(cfg.optimizer.params, "optimizer")
    )

    # Scheduler を使う場合だけ確認
    if cfg.scheduler.name is not None:
        scheduler_cls = resolve_scheduler_class(cfg.scheduler.name)
        check_params(
            scheduler_cls, param_names(cfg.scheduler.params, "scheduler")
        )
