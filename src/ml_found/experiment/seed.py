import random

import numpy as np
import torch


def fix_seed(seed: int, deterministic: bool = True) -> None:
    """
    乱数シードを固定する関数

    Parameters
    ----------
    seed: int
        乱数シードの値
    deterministic: bool = True
        cuDNN の計算を毎回同じ結果になるようにするかどうか

    Returns
    -------
    None
    """
    # Python 標準の乱数を固定する
    random.seed(seed)

    # NumPy の乱数を固定する
    np.random.seed(seed)

    # PyTorch の乱数を CPU, GPU ともに固定する
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 速度より再現性を優先する場合は cuDNN の自動選択を止める
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_rng_state() -> dict:
    """
    乱数の状態を取得する関数

    Parameters
    ----------
    None

    Returns
    -------
    state: dict
        Python, NumPy, PyTorch の乱数の状態
    """
    # NumPy の状態は配列を含み安全に読み込めないので整数のリストにする
    np_state = np.random.get_state()
    numpy_state = [
        np_state[0], np_state[1].tolist(), np_state[2], np_state[3],
        np_state[4]
    ]

    # 各ライブラリの状態をまとめる
    state = {
        "python": random.getstate(),
        "numpy": numpy_state,
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None
        ),
        "mps": (
            torch.mps.get_rng_state()
            if torch.backends.mps.is_available() else None
        )
    }

    return state


def set_rng_state(state: dict) -> None:
    """
    乱数の状態を戻す関数

    Parameters
    ----------
    state: dict
        get_rng_state で取得した乱数の状態

    Returns
    -------
    None
    """
    # Python と NumPy の状態を戻す
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
            numpy_state[2], numpy_state[3], numpy_state[4]
        )
    )

    # PyTorch の状態を CPU, GPU の順に戻す
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if state["mps"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())
