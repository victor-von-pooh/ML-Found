import importlib
import inspect
from types import ModuleType
from typing import Any, Iterable, Optional

from omegaconf import DictConfig, OmegaConf
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
    # クラスを取得し, Optimizer であることの確認
    optimizer_cls = resolve_class(name, optim)
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


def requires_closure(optimizer: optim.Optimizer) -> bool:
    """
    Optimizer の step が損失を計算し直す closure を必要とするかを判定する関数

    Parameters
    ----------
    optimizer: optim.Optimizer
        判定する Optimizer

    Returns
    ----------
    required: bool
        step の closure が既定値の無い引数の場合は True (LBFGS など)
    """
    # step のシグネチャで closure が必須の引数かどうかを確認
    param = inspect.signature(optimizer.step).parameters.get("closure")
    required = (
        param is not None and param.default is inspect.Parameter.empty
    )

    return required


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
    # Optimizer のクラスが見つかり, Optimizer であることの確認
    # 自分で作った Optimizer を Trainer に渡す場合も config の名前を確認
    resolve_optimizer_class(cfg.optimizer.name)

    # 学習率を更新する間隔の確認
    interval = cfg.scheduler.interval
    check_interval(interval)

    # Scheduler を使わない場合は以上で確認終了
    if cfg.scheduler.name is None:
        return

    # Scheduler のクラスを取得し, ReduceLROnPlateau は間隔と向きの組み合わせを確認
    # mode を省略した場合は build_scheduler と同じく trainer.mode を使用
    scheduler_cls = resolve_scheduler_class(cfg.scheduler.name)
    if issubclass(scheduler_cls, optim.lr_scheduler.ReduceLROnPlateau):
        plateau_mode = cfg.scheduler.params.get("mode", cfg.trainer.mode)
        check_plateau(interval, plateau_mode, cfg.trainer.mode)


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


def missing_data_size_params(cfg: DictConfig) -> list[str]:
    """
    データの量で決まる Scheduler の引数のうち, config に無いものを返す関数

    今は OneCycleLR の epochs と steps_per_epoch (total_steps が無い場合) が
    対象. Trainer が学習を始めるときに, 学習用データのバッチ数から補う

    Parameters
    ----------
    cfg: DictConfig
        scheduler の config

    Returns
    ----------
    missing: list[str]
        補う必要のある引数の名前, 無ければ空
    """
    # OneCycleLR で, 総ステップ数を書いていない場合だけが対象
    if cfg.name is None:
        return []
    scheduler_cls = resolve_scheduler_class(cfg.name)
    if not issubclass(scheduler_cls, optim.lr_scheduler.OneCycleLR):
        return []
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
    # 名前が指定されていない場合は不使用
    if cfg.name is None:
        scheduler = None
        return scheduler

    # クラスを取得
    scheduler_cls = resolve_scheduler_class(cfg.name)

    # ReduceLROnPlateau の向きを指定しない場合は監視する指標の向きに統一
    kwargs = to_kwargs(cfg.params)
    if issubclass(scheduler_cls, optim.lr_scheduler.ReduceLROnPlateau):
        kwargs.setdefault("mode", mode)

    # 引数を渡して作り, 間隔と向きの組み合わせの確認
    scheduler = scheduler_cls(optimizer, **kwargs)
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

    # ReduceLROnPlateau は指標の値を見て学習率を下げるので値の受け渡し
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        if metric is None:
            raise ValueError("ReduceLROnPlateau には指標の値が必要です")
        scheduler.step(metric)
    else:
        scheduler.step()
