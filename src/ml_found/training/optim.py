import contextlib
import importlib
import inspect
from types import ModuleType
from typing import Any, Iterable, Iterator, Optional
import warnings

from omegaconf import DictConfig, OmegaConf
import torch
from torch import optim


def resolve_class(name: str, module: ModuleType) -> type:
    """
    名前からクラスを取得する関数

    Parameters
    ----------
    name: str
        クラス名, . を含む場合はモジュールのパスも含めた名前とみなす
    module: ModuleType
        クラス名だけが指定された場合に探すモジュール

    Returns
    ----------
    cls: type
        取得したクラス
    """
    # . を含む場合は自作のクラスなどとみなしてモジュールから読み込み
    if "." in name:
        module_name, class_name = name.rsplit(".", 1)
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"{name} が見つかりません") from e
        return cls

    # クラス名だけの場合は指定されたモジュールから探索
    if not hasattr(module, name):
        raise ValueError(f"{module.__name__} に {name} がありません")
    cls = getattr(module, name)

    return cls


def to_kwargs(params: DictConfig) -> dict:
    """
    config の引数をクラスに渡せる辞書にする関数

    Parameters
    ----------
    params: DictConfig
        クラスに渡す引数の config

    Returns
    ----------
    kwargs: dict
        クラスに渡す引数の辞書
    """
    # OmegaConf の型のままだと torch が扱えないので Python の辞書に変換
    kwargs = OmegaConf.to_container(params, resolve=True)

    return kwargs


# プロジェクトが register_optimizer で登録した Optimizer (名前ごとのクラス)
REGISTERED_OPTIMIZERS: dict[str, type] = {}


def resolve_optimizer_class(name: str) -> type:
    """
    名前から Optimizer のクラスを取得する関数

    Parameters
    ----------
    name: str
        torch.optim のクラス名, または . を含む自作のクラスの名前

    Returns
    ----------
    optimizer_cls: type
        Optimizer のクラス
    """
    # 登録した名前を優先してクラスを取得し, Optimizer であることの確認
    optimizer_cls = REGISTERED_OPTIMIZERS.get(name) or resolve_class(
        name, optim
    )
    if not (
        isinstance(optimizer_cls, type)
        and issubclass(optimizer_cls, optim.Optimizer)
    ):
        raise ValueError(f"{name} は Optimizer ではありません")

    return optimizer_cls


def resolve_scheduler_class(name: str) -> type:
    """
    名前から Scheduler のクラスを取得する関数

    Parameters
    ----------
    name: str
        torch.optim.lr_scheduler のクラス名, または . を含む自作のクラスの名前

    Returns
    ----------
    scheduler_cls: type
        Scheduler のクラス
    """
    # クラスを取得し, Scheduler であることの確認
    scheduler_cls = resolve_class(name, optim.lr_scheduler)
    if not (
        isinstance(scheduler_cls, type)
        and issubclass(scheduler_cls, optim.lr_scheduler.LRScheduler)
    ):
        raise ValueError(f"{name} は Scheduler ではありません")

    return scheduler_cls


def check_interval(interval: str) -> None:
    """
    学習率を更新する間隔が正しいことを確かめる関数

    Parameters
    ----------
    interval: str
        学習率を更新する間隔

    Returns
    ----------
    None
    """
    # 間隔は epoch か step に制限
    if interval not in ("epoch", "step"):
        raise ValueError(
            f"scheduler.interval は epoch か step にしてください: {interval}"
        )


def check_plateau(interval: str, plateau_mode: str, mode: str) -> None:
    """
    ReduceLROnPlateau の間隔と向きが正しいことを確かめる関数

    Parameters
    ----------
    interval: str
        学習率を更新する間隔
    plateau_mode: str
        ReduceLROnPlateau の mode
    mode: str
        監視する指標の良し悪しの向き (trainer.mode)

    Returns
    ----------
    None
    """
    # ReduceLROnPlateau は監視する指標を見るので, 間隔と向きの統一
    if interval != "epoch":
        raise ValueError(
            "ReduceLROnPlateau は scheduler.interval を epoch にしてください"
        )
    if plateau_mode != mode:
        raise ValueError(
            f"ReduceLROnPlateau の mode ({plateau_mode}) が "
            f"trainer.mode ({mode}) と違います"
        )


def build_optimizer(
    cfg: DictConfig, params: Iterable[Any]
) -> optim.Optimizer:
    """
    config から Optimizer を作る関数

    Parameters
    ----------
    cfg: DictConfig
        optimizer の config
    params: Iterable[Any]
        学習するパラメータ, model.parameters() やパラメータグループのリスト

    Returns
    ----------
    optimizer: optim.Optimizer
        作成した Optimizer
    """
    # クラスを取得して引数を渡して作成
    optimizer_cls = resolve_optimizer_class(cfg.name)
    optimizer = optimizer_cls(params, **to_kwargs(cfg.params))

    return optimizer


# step に closure が必要な torch の Optimizer
CLOSURE_OPTIMIZERS = (optim.LBFGS,)


def requires_closure(optimizer: optim.Optimizer) -> bool:
    """
    Optimizer の step が損失を計算し直す closure を必要とするかを判定する関数

    次の順に決める. クラスの属性 requires_closure (自作のクラスが宣言する),
    torch の Optimizer の仕様 (LBFGS), step のシグネチャ (closure が既定値の
    無い引数かどうか)

    Parameters
    ----------
    optimizer: optim.Optimizer
        判定する Optimizer

    Returns
    ----------
    required: bool
        closure が必要な場合は True
    """
    # クラスで判定 (登録簿の仕様と同じ判定)
    required = class_requires_closure(type(optimizer))

    return required


def class_uses_gradients(optimizer_cls: type) -> bool:
    """
    Optimizer のクラスが勾配を使うかを判定する関数

    Nelder-Mead 法のような勾配を使わない Optimizer は, クラスの属性
    uses_gradients = False で宣言する. その場合 ML-Found は逆伝播をせず,
    損失だけを求める closure を step に渡す

    Parameters
    ----------
    optimizer_cls: type
        判定する Optimizer のクラス

    Returns
    ----------
    uses: bool
        勾配を使う場合は True (宣言が無い場合も True)
    """
    # クラスの属性で False と宣言した場合だけ勾配を使わない
    uses = getattr(optimizer_cls, "uses_gradients", True) is not False

    return uses


def class_requires_closure(optimizer_cls: type) -> bool:
    """
    Optimizer のクラスの step が closure を必要とするかを判定する関数

    requires_closure と登録簿 (describe_optimizers) の両方で使う

    Parameters
    ----------
    optimizer_cls: type
        判定する Optimizer のクラス

    Returns
    ----------
    required: bool
        closure が必要な場合は True
    """
    # 自作のクラスがクラスの属性で宣言した場合はそれに従う
    declared = getattr(optimizer_cls, "requires_closure", None)
    if isinstance(declared, bool):
        return declared

    # torch のクラスは仕様で判定
    if issubclass(optimizer_cls, CLOSURE_OPTIMIZERS):
        return True
    if optimizer_cls.__module__.startswith("torch."):
        return False

    # それ以外は step のシグネチャで closure が必須の引数かどうかを確認
    try:
        param = inspect.signature(optimizer_cls.step).parameters.get(
            "closure"
        )
    except (TypeError, ValueError):
        return False
    required = (
        param is not None and param.default is inspect.Parameter.empty
    )

    return required


def uses_named_optimizers(cfg: Any) -> bool:
    """
    config が optimizers に名前付きで複数の Optimizer を書いているかを返す関数

    Parameters
    ----------
    cfg: Any
        学習に使う config

    Returns
    ----------
    named: bool
        optimizers が空でない場合は True
    """
    named = bool(cfg.get("optimizers")) if hasattr(cfg, "get") else False

    return named


def named_views(cfg: DictConfig) -> list[tuple[str, DictConfig]]:
    """
    optimizers の名前ごとに, optimizer と scheduler の 1 組を持つ config を
    作る関数

    1 つの Optimizer の場合の確認 (check_optim_config など) を, 名前ごとに
    そのまま使うために使う

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    views: list[tuple[str, DictConfig]]
        名前と, optimizer, scheduler, trainer を持つ config の組 (書いた順)
    """
    # 名前ごとに optimizer と scheduler を差し替えた config を作成
    # 新しい config には他の項目が無いので, 値の参照 (${...}) は元の config で解決
    views = []
    for name, entry in cfg.optimizers.items():
        view = OmegaConf.create({
            "optimizer": {
                "name": entry.name,
                "params": OmegaConf.to_container(entry.params, resolve=True)
            },
            "scheduler": OmegaConf.to_container(
                entry.scheduler, resolve=True
            ),
            "trainer": OmegaConf.to_container(cfg.trainer, resolve=True)
        })
        views.append((str(name), view))

    return views


def check_named_optimizers(cfg: DictConfig) -> None:
    """
    optimizers に名前付きで書いた Optimizer の設定を確かめる関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    None
    """
    # Scheduler は一番上の scheduler ではなく, Optimizer ごとの scheduler に記述
    # (既定値から変えた項目があればエラー)
    top = OmegaConf.to_container(cfg.scheduler)
    if top != {
        "name": None, "params": {}, "interval": "epoch",
        "warmup_steps": None, "warmup_ratio": None,
        "warmup_start_factor": 0.01
    }:
        raise ValueError(
            "optimizers を使う場合は, Scheduler と warmup を一番上の scheduler "
            "ではなく optimizers.<名前>.scheduler に書いてください"
        )

    # 受け持つ部分は 1 つ以上の名前のリスト
    intervals = set()
    for name, entry in cfg.optimizers.items():
        modules = list(entry.modules)
        if not modules or not all(
            isinstance(module, str) and module for module in modules
        ):
            raise ValueError(
                f"optimizers.{name}.modules に, 受け持つモデルの部分 (model の"
                f"属性の名前) を 1 つ以上書いてください: {modules}"
            )

        # 損失ごとに勾配を求めて更新するので, closure が必要なものと
        # 勾配を使わないものは不可
        with named_errors(str(name)):
            optimizer_cls = resolve_optimizer_class(entry.name)
        if class_requires_closure(optimizer_cls) or not class_uses_gradients(
            optimizer_cls
        ):
            raise ValueError(
                f"optimizers.{name}: {entry.name} は closure が必要か勾配を"
                "使わない Optimizer なので, optimizers では使えません"
            )
        if entry.scheduler.name is not None or entry.scheduler.get(
            "warmup_steps"
        ) or entry.scheduler.get("warmup_ratio"):
            intervals.add(entry.scheduler.interval)

    # 学習率を更新する間隔は, Scheduler を付けた全ての Optimizer で同じ
    if len(intervals) > 1:
        raise ValueError(
            "optimizers の Scheduler の interval は全て同じにしてください: "
            f"{sorted(intervals)}"
        )


def check_optim_config(cfg: DictConfig) -> None:
    """
    Optimizer と Scheduler の設定を, モデルを作る前に確かめる関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    None
    """
    # 名前付きの Optimizer は, 名前ごとに同じ確認を実行
    if uses_named_optimizers(cfg):
        check_named_optimizers(cfg)
        for name, view in named_views(cfg):
            with named_errors(name):
                check_optim_config(view)
        return

    # Optimizer のクラスが見つかり, Optimizer であることの確認
    # 自分で作った Optimizer を Trainer に渡す場合も config の名前を確認
    resolve_optimizer_class(cfg.optimizer.name)

    # 学習率を更新する間隔と warmup の確認
    interval = cfg.scheduler.interval
    check_interval(interval)
    check_warmup(cfg.scheduler)

    # Scheduler を使わない場合は以上で確認終了
    if cfg.scheduler.name is None:
        return

    # Scheduler のクラスを取得し, ReduceLROnPlateau は間隔と向きの組み合わせを確認
    # mode を省略した場合は build_scheduler と同じく trainer.mode を使用
    scheduler_cls = resolve_scheduler_class(cfg.scheduler.name)
    if issubclass(scheduler_cls, optim.lr_scheduler.ReduceLROnPlateau):
        plateau_mode = cfg.scheduler.params.get("mode", cfg.trainer.mode)
        check_plateau(interval, plateau_mode, cfg.trainer.mode)


# warmup と組み合わせられない Scheduler (指標で変えるものと, 自分で上げ下げ
# するもの)
NO_WARMUP_SCHEDULERS = (
    optim.lr_scheduler.ReduceLROnPlateau, optim.lr_scheduler.OneCycleLR,
    optim.lr_scheduler.CyclicLR
)


def check_warmup(cfg: DictConfig) -> None:
    """
    scheduler の warmup の設定を確かめる関数

    Parameters
    ----------
    cfg: DictConfig
        scheduler の config

    Returns
    ----------
    None
    """
    # 回数と割合はどちらか一方, 回数は 0 以上の整数, 割合は 0 以上 1 未満
    steps = cfg.get("warmup_steps")
    ratio = cfg.get("warmup_ratio")
    factor = cfg.get("warmup_start_factor", 0.01)
    if steps is not None and ratio is not None:
        raise ValueError(
            "scheduler.warmup_steps と warmup_ratio はどちらか一方にしてください"
        )
    if steps is not None and (
        not isinstance(steps, int) or isinstance(steps, bool) or steps < 0
    ):
        raise ValueError(
            f"scheduler.warmup_steps は 0 以上の整数にしてください: {steps}"
        )
    if ratio is not None and not 0.0 <= float(ratio) < 1.0:
        raise ValueError(
            f"scheduler.warmup_ratio は 0 以上 1 未満にしてください: {ratio}"
        )
    if not 0.0 < float(factor) <= 1.0:
        raise ValueError(
            "scheduler.warmup_start_factor は 0 より大きく 1 以下にしてください: "
            f"{factor}"
        )

    # warmup の要らない Scheduler との組み合わせはエラー
    if (steps or ratio) and cfg.name is not None:
        scheduler_cls = resolve_scheduler_class(cfg.name)
        if issubclass(scheduler_cls, NO_WARMUP_SCHEDULERS):
            raise ValueError(
                f"{cfg.name} には scheduler.warmup_steps と warmup_ratio を"
                "使えません"
            )


def check_scheduler(
    scheduler: Optional[optim.lr_scheduler.LRScheduler],
    interval: str, mode: str
) -> None:
    """
    Scheduler と学習率を更新する間隔の組み合わせが正しいことを確かめる関数

    Parameters
    ----------
    scheduler: Optional[optim.lr_scheduler.LRScheduler]
        確かめる Scheduler, None の場合は間隔だけを確かめる
    interval: str
        学習率を更新する間隔
    mode: str
        監視する指標の良し悪しの向き

    Returns
    ----------
    None
    """
    # 間隔の確認
    check_interval(interval)

    # ReduceLROnPlateau は間隔と向きの組み合わせを確認
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        check_plateau(interval, scheduler.mode, mode)


# 値に auto と書くと, 学習全体のステップ数 (interval が epoch ならエポック数,
# step ならエポック数 × バッチ数) で補う Scheduler の引数
AUTO_TOTAL_PARAMS = {
    optim.lr_scheduler.CosineAnnealingLR: ("T_max",),
    optim.lr_scheduler.LinearLR: ("total_iters",),
    optim.lr_scheduler.ConstantLR: ("total_iters",),
    optim.lr_scheduler.PolynomialLR: ("total_iters",),
}


def auto_total_params(scheduler_cls: type) -> tuple[str, ...]:
    """
    auto と書ける Scheduler の引数の名前を返す関数

    Parameters
    ----------
    scheduler_cls: type
        Scheduler のクラス

    Returns
    ----------
    names: tuple[str, ...]
        auto と書ける引数の名前, 無ければ空
    """
    # torch のクラスとそれを継承したクラスが対象 (OneCycleLR の補いと同じ)
    names: tuple[str, ...] = ()
    for base, params in AUTO_TOTAL_PARAMS.items():
        if isinstance(scheduler_cls, type) and issubclass(scheduler_cls, base):
            names = params

    return names


def auto_params(cfg: DictConfig) -> list[str]:
    """
    Scheduler の引数のうち, 値に auto と書いたものを返す関数

    Parameters
    ----------
    cfg: DictConfig
        scheduler の config

    Returns
    ----------
    names: list[str]
        auto と書いた引数の名前, 無ければ空
    """
    # 名前が無い場合と, auto と書けない Scheduler は対象外
    if cfg.name is None:
        return []
    scheduler_cls = resolve_scheduler_class(cfg.name)
    names = [
        name for name in auto_total_params(scheduler_cls)
        if cfg.params.get(name) == "auto"
    ]

    return names


def needs_data(cfg: DictConfig) -> bool:
    """
    Scheduler を作るのに学習用データの量が要るかを返す関数

    Parameters
    ----------
    cfg: DictConfig
        scheduler の config

    Returns
    ----------
    needed: bool
        データの量で決まる引数か, 割合で書いた warmup がある場合は True
    """
    # 補う引数か, 回数に直していない warmup の割合があるかを確認
    ratio_only = (
        cfg.get("warmup_ratio") is not None
        and cfg.get("warmup_steps") is None
    )
    needed = bool(missing_data_size_params(cfg)) or ratio_only

    return needed


def missing_data_size_params(cfg: DictConfig) -> list[str]:
    """
    データの量で決まる Scheduler の引数のうち, config に無いものを返す関数

    OneCycleLR の epochs と steps_per_epoch (total_steps が無い場合) と,
    auto と書いた引数 (CosineAnnealingLR の T_max など) が対象. Trainer が
    学習を始めるときに, 学習用データのバッチ数から補う

    Parameters
    ----------
    cfg: DictConfig
        scheduler の config

    Returns
    ----------
    missing: list[str]
        補う必要のある引数の名前, 無ければ空
    """
    # auto と書いた引数と, OneCycleLR で総ステップ数を書いていない場合が対象
    if cfg.name is None:
        return []
    scheduler_cls = resolve_scheduler_class(cfg.name)
    if not issubclass(scheduler_cls, optim.lr_scheduler.OneCycleLR):
        return auto_params(cfg)
    params = cfg.params
    if params.get("total_steps") is not None:
        return []
    missing = [
        name for name in ("epochs", "steps_per_epoch")
        if params.get(name) is None
    ]

    return missing


def build_scheduler(
    cfg: DictConfig, optimizer: optim.Optimizer, mode: str = "min"
) -> Optional[optim.lr_scheduler.LRScheduler]:
    """
    config から Scheduler を作る関数

    Parameters
    ----------
    cfg: DictConfig
        scheduler の config
    optimizer: optim.Optimizer
        学習率を調整する Optimizer
    mode: str = "min"
        監視する指標の良し悪しの向き, ReduceLROnPlateau に使う

    Returns
    ----------
    scheduler: Optional[optim.lr_scheduler.LRScheduler]
        作成した Scheduler, 使わない場合は None
    """
    # warmup の回数 (割合で書いた場合は学習を始めるときに補った値)
    warmup = cfg.get("warmup_steps")
    if cfg.get("warmup_ratio") is not None and warmup is None:
        raise ValueError(
            "scheduler.warmup_ratio は学習を始めるときに回数に直すので, "
            "その前には Scheduler を作れません"
        )

    # 名前が指定されていない場合は, warmup だけか不使用
    main = None
    if cfg.name is not None:
        # ReduceLROnPlateau の向きを指定しない場合は監視する指標の向きに統一
        scheduler_cls = resolve_scheduler_class(cfg.name)
        kwargs = to_kwargs(cfg.params)
        if issubclass(scheduler_cls, optim.lr_scheduler.ReduceLROnPlateau):
            kwargs.setdefault("mode", mode)
        main = scheduler_cls(optimizer, **kwargs)

    # warmup がある場合は, 線形に上げる Scheduler の後に続ける
    scheduler = main
    if warmup:
        warm = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=cfg.get("warmup_start_factor", 0.01),
            end_factor=1.0, total_iters=int(warmup)
        )
        scheduler = warm if main is None else optim.lr_scheduler.SequentialLR(
            optimizer, [warm, main], milestones=[int(warmup)]
        )

    # 間隔と向きの組み合わせの確認
    if scheduler is not None:
        check_scheduler(scheduler, cfg.interval, mode)

    return scheduler


def step_scheduler(
    scheduler: Optional[optim.lr_scheduler.LRScheduler],
    metric: Optional[float] = None
) -> None:
    """
    Scheduler の学習率を 1 回更新する関数

    Parameters
    ----------
    scheduler: Optional[optim.lr_scheduler.LRScheduler]
        更新する Scheduler, None の場合は何もしない
    metric: Optional[float] = None
        ReduceLROnPlateau が改善を判定するための指標の値

    Returns
    ----------
    None
    """
    # Scheduler を使わない場合は処理なし
    if scheduler is None:
        return

    # 名前付きの Optimizer の Scheduler をまとめた入れ物は, 中で 1 つずつ更新
    if not isinstance(
        scheduler, optim.lr_scheduler.LRScheduler
    ) and hasattr(scheduler, "schedulers"):
        scheduler.step(metric)
        return

    # ReduceLROnPlateau は指標の値を見て学習率を下げるので値の受け渡し
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        if metric is None:
            raise ValueError("ReduceLROnPlateau には指標の値が必要です")
        scheduler.step(metric)
    else:
        scheduler.step()


def try_build_optim(cfg: DictConfig, device: torch.device) -> None:
    """
    config の Optimizer と Scheduler を, 仮のパラメータで作れるかを確かめる関数

    範囲を持たない引数や組み合わせの誤り (PyTorch が作るときに出すエラー) を,
    出力先を作る前に見つけるために使う. データの量で決まる Scheduler の引数は
    仮の値で補う

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config
    device: torch.device
        学習に使うデバイス (デバイスで決まる引数, fused などのため)

    Returns
    ----------
    None
    """
    # 名前付きの Optimizer は, 名前ごとに同じ確認を実行
    if uses_named_optimizers(cfg):
        for name, view in named_views(cfg):
            with named_errors(name):
                try_build_optim(view, device)
        return

    # 値の決まっていない引数 (???) がある場合は, 範囲の確認と同じく対象外
    # (解決できない補間は, その前の確認でエラー)
    sections = [cfg.optimizer]
    if cfg.scheduler.name is not None:
        sections.append(cfg.scheduler)
    for section in sections:
        try:
            OmegaConf.to_container(
                section, resolve=True, throw_on_missing=True
            )
        except Exception:
            return

    # 学習と同じデバイスの仮のパラメータで Optimizer を作成
    # 作るときの警告は学習の本番でも出るので, ここでは非表示
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parameter = torch.nn.Parameter(torch.zeros(2, 2, device=device))
            optimizer = build_optimizer(cfg.optimizer, [parameter])
        except Exception as error:
            raise ValueError(
                f"optimizer の引数で {cfg.optimizer.name} を作れません: "
                f"{type(error).__name__}: {error}"
            ) from error

        # Scheduler も作れるかを確認 (データの量で決まる引数は仮の値)
        if cfg.scheduler.name is None:
            return
        scheduler_cfg = OmegaConf.create(
            OmegaConf.to_container(cfg.scheduler, resolve=True)
        )
        for name in missing_data_size_params(scheduler_cfg):
            scheduler_cfg.params[name] = (
                1 if name == "steps_per_epoch" else int(cfg.trainer.epochs)
            )
        if scheduler_cfg.get("warmup_ratio") is not None:
            scheduler_cfg.warmup_steps = 1
            scheduler_cfg.warmup_ratio = None
        try:
            build_scheduler(scheduler_cfg, optimizer, cfg.trainer.mode)
        except Exception as error:
            raise ValueError(
                f"scheduler の引数で {cfg.scheduler.name} を作れません: "
                f"{type(error).__name__}: {error}"
            ) from error


@contextlib.contextmanager
def named_errors(name: str) -> Iterator[None]:
    """
    名前付きの Optimizer の確認で出たエラーに, どの Optimizer かを付ける
    文脈を作る関数 (with で使う)

    Parameters
    ----------
    name: str
        optimizers に書いた名前

    Returns
    ----------
    context: Iterator[None]
        with で使う文脈
    """
    # ValueError だけ, 名前を付けて同じ種類で投げ直し
    try:
        yield
    except ValueError as error:
        raise ValueError(f"optimizers.{name}: {error}") from error
