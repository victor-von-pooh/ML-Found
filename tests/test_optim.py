from omegaconf import OmegaConf
import pytest
from torch import nn, optim

from ml_found.training.optim import (
    build_optimizer, build_scheduler, step_scheduler
)


def make_optimizer(lr: float = 0.1) -> optim.Optimizer:
    """
    テスト用の SGD を作る関数

    Parameters
    ----------
    lr: float = 0.1
        学習率

    Returns
    ----------
    optimizer: optim.Optimizer
        小さなモデルの SGD
    """
    # 小さなモデルのパラメータで SGD の作成
    cfg = OmegaConf.create({"name": "SGD", "params": {"lr": lr}})
    optimizer = build_optimizer(cfg, nn.Linear(2, 1).parameters())

    return optimizer


def get_lr(optimizer: optim.Optimizer) -> float:
    """
    現在の学習率を取得する関数

    Parameters
    ----------
    optimizer: optim.Optimizer
        学習率を調べる Optimizer

    Returns
    ----------
    lr: float
        先頭のパラメータグループの学習率
    """
    # 先頭のパラメータグループから取り出し
    lr = optimizer.param_groups[0]["lr"]

    return lr


def test_build_optimizer_with_params() -> None:
    """
    クラス名と引数から Optimizer が作られることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # リストで書いた betas を含めて Adam の作成
    cfg = OmegaConf.create(
        {"name": "Adam", "params": {"lr": 0.01, "betas": [0.8, 0.9]}}
    )
    optimizer = build_optimizer(cfg, nn.Linear(2, 1).parameters())

    assert isinstance(optimizer, optim.Adam)
    assert get_lr(optimizer) == 0.01
    assert tuple(optimizer.param_groups[0]["betas"]) == (0.8, 0.9)


def test_build_optimizer_dotted_name() -> None:
    """
    モジュールのパスを含めた名前でも作れることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # モジュールのパスを含めて指定
    cfg = OmegaConf.create({"name": "torch.optim.AdamW", "params": {}})
    optimizer = build_optimizer(cfg, nn.Linear(2, 1).parameters())

    assert isinstance(optimizer, optim.AdamW)


@pytest.mark.parametrize(
    "name", ["Adamm", "lr_scheduler", "no_such_module.Optimizer"]
)
def test_build_optimizer_invalid_name(name: str) -> None:
    """
    存在しない名前や Optimizer でない名前でエラーになることを確かめる関数

    Parameters
    ----------
    name: str
        正しくない名前

    Returns
    ----------
    None
    """
    # 正しくない名前を指定
    cfg = OmegaConf.create({"name": name, "params": {}})

    with pytest.raises(ValueError):
        build_optimizer(cfg, nn.Linear(2, 1).parameters())


def test_build_scheduler_none() -> None:
    """
    名前が null の場合は Scheduler を使わないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 名前を null に設定
    cfg = OmegaConf.create({"name": None, "params": {}, "interval": "epoch"})
    scheduler = build_scheduler(cfg, make_optimizer())

    assert scheduler is None


def test_step_lr() -> None:
    """
    StepLR で学習率が下がることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 1 回ごとに学習率を 0.5 倍にする StepLR の作成
    optimizer = make_optimizer(lr=0.1)
    cfg = OmegaConf.create(
        {
            "name": "StepLR", "params": {"step_size": 1, "gamma": 0.5},
            "interval": "epoch"
        }
    )
    scheduler = build_scheduler(cfg, optimizer)

    # 2 回更新
    optimizer.step()
    step_scheduler(scheduler)
    optimizer.step()
    step_scheduler(scheduler)

    assert get_lr(optimizer) == pytest.approx(0.025)


def test_reduce_lr_on_plateau() -> None:
    """
    ReduceLROnPlateau で改善しない場合に学習率が下がることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 1 回改善しなければ学習率を 0.1 倍にする ReduceLROnPlateau の作成
    optimizer = make_optimizer(lr=0.1)
    cfg = OmegaConf.create(
        {
            "name": "ReduceLROnPlateau",
            "params": {"patience": 0, "factor": 0.1},
            "interval": "epoch"
        }
    )
    scheduler = build_scheduler(cfg, optimizer)

    # 同じ値を続けて受け渡し
    step_scheduler(scheduler, 1.0)
    step_scheduler(scheduler, 1.0)

    assert get_lr(optimizer) == pytest.approx(0.01)

    # 指標の値を渡さないとエラー
    with pytest.raises(ValueError):
        step_scheduler(scheduler)


def test_build_scheduler_invalid() -> None:
    """
    正しくない間隔や Scheduler でない名前でエラーになることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 間隔が正しくない場合
    cfg = OmegaConf.create(
        {"name": "StepLR", "params": {"step_size": 1}, "interval": "batch"}
    )
    with pytest.raises(ValueError):
        build_scheduler(cfg, make_optimizer())

    # Scheduler でない名前の場合
    cfg = OmegaConf.create({"name": "Adam", "params": {}, "interval": "epoch"})
    with pytest.raises(ValueError):
        build_scheduler(cfg, make_optimizer())


def test_reduce_lr_on_plateau_follows_mode() -> None:
    """
    ReduceLROnPlateau の向きが監視する指標の向きに揃うことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 大きい方が良い指標を監視する設定で作成
    cfg = OmegaConf.create(
        {"name": "ReduceLROnPlateau", "params": {}, "interval": "epoch"}
    )
    scheduler = build_scheduler(cfg, make_optimizer(), mode="max")

    assert scheduler.mode == "max"


@pytest.mark.parametrize(
    "params, interval",
    [({"mode": "min"}, "epoch"), ({}, "step")]
)
def test_reduce_lr_on_plateau_invalid(params: dict, interval: str) -> None:
    """
    ReduceLROnPlateau の向きや間隔が正しくない場合にエラーになることを確かめる関数

    Parameters
    ----------
    params: dict
        ReduceLROnPlateau に渡す引数
    interval: str
        学習率を更新する間隔

    Returns
    ----------
    None
    """
    # 監視する指標と向きが違う場合や, step ごとに更新する場合
    cfg = OmegaConf.create(
        {"name": "ReduceLROnPlateau", "params": params, "interval": interval}
    )

    with pytest.raises(ValueError, match="ReduceLROnPlateau"):
        build_scheduler(cfg, make_optimizer(), mode="max")
