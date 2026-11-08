import importlib
from types import ModuleType
from typing import Any, Iterable, Optional, Union

from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


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
    -------
    cls: type
        取得したクラス
    """
    # . を含む場合は自作のクラスなどとみなしてモジュールから読み込む
    if "." in name:
        module_name, class_name = name.rsplit(".", 1)
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"{name} が見つかりません") from e
        return cls

    # クラス名だけの場合は指定されたモジュールから探す
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
    -------
    kwargs: dict
        クラスに渡す引数の辞書
    """
    # OmegaConf の型のままだと torch が扱えないので Python の辞書にする
    kwargs = OmegaConf.to_container(params, resolve=True)

    return kwargs


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
    -------
    optimizer: optim.Optimizer
        作成した Optimizer
    """
    # クラスを取得し, Optimizer であることを確かめる
    optimizer_cls = resolve_class(cfg.name, optim)
    if not (
        isinstance(optimizer_cls, type)
        and issubclass(optimizer_cls, optim.Optimizer)
    ):
        raise ValueError(f"{cfg.name} は Optimizer ではありません")

    # 引数を渡して作る
    optimizer = optimizer_cls(params, **to_kwargs(cfg.params))

    return optimizer


def build_scheduler(
    cfg: DictConfig, optimizer: optim.Optimizer, mode: str = "min"
) -> Optional[Union[LRScheduler, ReduceLROnPlateau]]:
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
    -------
    scheduler: Optional[Union[LRScheduler, ReduceLROnPlateau]]
        作成した Scheduler, 使わない場合は None
    """
    # 名前が指定されていない場合は使わない
    if cfg.name is None:
        scheduler = None
        return scheduler

    # 学習率を更新する間隔が正しいことを確かめる
    if cfg.interval not in ("epoch", "step"):
        raise ValueError(
            f"scheduler.interval は epoch か step にしてください: "
            f"{cfg.interval}"
        )

    # クラスを取得し, Scheduler であることを確かめる
    scheduler_cls = resolve_class(cfg.name, optim.lr_scheduler)
    if not (
        isinstance(scheduler_cls, type)
        and issubclass(scheduler_cls, (LRScheduler, ReduceLROnPlateau))
    ):
        raise ValueError(f"{cfg.name} は Scheduler ではありません")

    # ReduceLROnPlateau は監視する指標を見るので, 間隔と向きを揃える
    kwargs = to_kwargs(cfg.params)
    if issubclass(scheduler_cls, ReduceLROnPlateau):
        if cfg.interval != "epoch":
            raise ValueError(
                "ReduceLROnPlateau は scheduler.interval を epoch にしてください"
            )
        kwargs.setdefault("mode", mode)
        if kwargs["mode"] != mode:
            raise ValueError(
                f"ReduceLROnPlateau の mode ({kwargs['mode']}) が "
                f"trainer.mode ({mode}) と違います"
            )

    # 引数を渡して作る
    scheduler = scheduler_cls(optimizer, **kwargs)

    return scheduler


def step_scheduler(
    scheduler: Optional[Union[LRScheduler, ReduceLROnPlateau]],
    metric: Optional[float] = None
) -> None:
    """
    Scheduler の学習率を 1 回更新する関数

    Parameters
    ----------
    scheduler: Optional[Union[LRScheduler, ReduceLROnPlateau]]
        更新する Scheduler, None の場合は何もしない
    metric: Optional[float] = None
        ReduceLROnPlateau が改善を判定するための指標の値

    Returns
    -------
    None
    """
    # Scheduler を使わない場合は何もしない
    if scheduler is None:
        return

    # ReduceLROnPlateau は指標の値を見て学習率を下げるので値を渡す
    if isinstance(scheduler, ReduceLROnPlateau):
        if metric is None:
            raise ValueError("ReduceLROnPlateau には指標の値が必要です")
        scheduler.step(metric)
    else:
        scheduler.step()
