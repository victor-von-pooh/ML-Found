from typing import Optional

from torch import nn, optim


def build_ema_model(
    model: nn.Module, decay: float, use_buffers: bool
) -> optim.swa_utils.AveragedModel:
    """
    学習するモデルの重みの指数移動平均 (EMA) を持つモデルを作る関数

    Parameters
    ----------
    model: nn.Module
        学習するモデル, 写しを同じデバイスに作る
    decay: float
        減衰率, 更新ごとに EMA = decay × EMA + (1 - decay) × 今の重み
    use_buffers: bool
        バッファも平均するかどうか, False の場合は学習するモデルの値を写す

    Returns
    ----------
    ema_model: optim.swa_utils.AveragedModel
        EMA の重みを持つモデル, 中のモデルは ema_model.module
    """
    # 最初の更新で今の重みを写し, その後は指数移動平均で更新するモデルの作成
    ema_model = optim.swa_utils.AveragedModel(
        model, multi_avg_fn=optim.swa_utils.get_ema_multi_avg_fn(decay),
        use_buffers=use_buffers
    )

    return ema_model


def ema_state_dict(
    ema_model: Optional[optim.swa_utils.AveragedModel], decay: Optional[float],
    use_buffers: bool
) -> Optional[dict]:
    """
    チェックポイントに入れる EMA の状態をまとめる関数

    Parameters
    ----------
    ema_model: Optional[optim.swa_utils.AveragedModel]
        EMA の重みを持つモデル, EMA を使わない場合は None
    decay: Optional[float]
        減衰率
    use_buffers: bool
        バッファも平均するかどうか

    Returns
    ----------
    state: Optional[dict]
        減衰率, use_buffers, 重みと更新回数 (state). EMA を使わない場合は
        None
    """
    # EMA を使わない場合は状態なし
    if ema_model is None:
        return None

    # 設定と, 重みと更新回数 (n_averaged) をまとめて受け渡し
    state = {
        "decay": decay,
        "use_buffers": use_buffers,
        "state": ema_model.state_dict()
    }

    return state


def reset_ema_model(
    ema_model: optim.swa_utils.AveragedModel, model: nn.Module
) -> None:
    """
    EMA を学習するモデルの今の重みから始め直す関数

    Parameters
    ----------
    ema_model: optim.swa_utils.AveragedModel
        EMA の重みを持つモデル
    model: nn.Module
        学習するモデル

    Returns
    ----------
    None
    """
    # 重みを写し, 更新回数を 0 に戻す (次の更新で今の重みを写し直す)
    ema_model.module.load_state_dict(model.state_dict())
    ema_model.n_averaged.zero_()
