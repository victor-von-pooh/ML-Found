import random
import warnings
from typing import Any, Callable

import numpy as np
import torch

# 利用者が登録した外部ライブラリの乱数 (名前ごとに固定, 取得, 復元の関数)
EXTRA_RNGS: dict[str, dict[str, Callable]] = {}


def register_rng(
    name: str, seed_fn: Callable[[int], None], get_state: Callable[[], Any],
    set_state: Callable[[Any], None]
) -> None:
    """
    外部ライブラリの乱数を, 固定とチェックポイントへの保存の対象に加える関数

    start_experiment (シードの固定) の前に呼ぶ. get_state が返す状態は,
    チェックポイントを安全に読み込めるように, テンソル, 数値, 文字列, None
    と, それらのリスト, タプル, 辞書だけで作る

    Parameters
    ----------
    name: str
        乱数の名前, チェックポイントの中の区別に使う (例: qiskit)
    seed_fn: Callable[[int], None]
        シードを受け取って乱数を固定する関数
    get_state: Callable[[], Any]
        乱数の今の状態を返す関数
    set_state: Callable[[Any], None]
        get_state が返した状態に戻す関数

    Returns
    ----------
    None
    """
    # 名前は空でない文字列だけ, 同じ名前の登録は上書き
    if not isinstance(name, str) or name == "":
        raise ValueError(f"乱数の名前は空でない文字列にしてください: {name!r}")
    for key, fn in [
        ("seed_fn", seed_fn), ("get_state", get_state),
        ("set_state", set_state)
    ]:
        if not callable(fn):
            raise TypeError(f"{key} には関数を渡してください: {fn!r}")
    EXTRA_RNGS[name] = {
        "seed": seed_fn, "get": get_state, "set": set_state
    }


def unregister_rng(name: str) -> None:
    """
    register_rng で加えた乱数を対象から外す関数

    Parameters
    ----------
    name: str
        外す乱数の名前

    Returns
    ----------
    None
    """
    # 登録されていない名前は何もしない
    EXTRA_RNGS.pop(name, None)


def set_seed(seed: int) -> None:
    """
    Python, NumPy, PyTorch の乱数のシードを設定する関数

    Parameters
    ----------
    seed: int
        乱数シードの値

    Returns
    ----------
    None
    """
    # Python 標準の乱数を固定
    random.seed(seed)

    # NumPy の乱数を固定
    np.random.seed(seed)

    # PyTorch の乱数を CPU, GPU ともに固定 (torch.manual_seed は CUDA と MPS も設定)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 登録された外部ライブラリの乱数を固定
    for rng in EXTRA_RNGS.values():
        rng["seed"](seed)


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
    ----------
    None
    """
    # Python, NumPy, PyTorch の乱数を固定
    set_seed(seed)

    # 速度より再現性を優先する場合は cuDNN の自動選択の停止
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_rng_state() -> dict:
    """
    乱数の状態を取得する関数

    Parameters
    ----------
    None

    Returns
    ----------
    state: dict
        Python, NumPy, PyTorch と, 登録された外部ライブラリの乱数の状態
    """
    # NumPy の状態は配列を含み安全に読み込めないので整数のリストに変換
    np_state = np.random.get_state()
    numpy_state = [
        np_state[0], np_state[1].tolist(), np_state[2], np_state[3],
        np_state[4]
    ]

    # 各ライブラリの状態を集約
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
        ),
        "extra": {
            name: rng["get"]() for name, rng in EXTRA_RNGS.items()
        }
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
    ----------
    None
    """
    # Python と NumPy の状態の復元
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
            numpy_state[2], numpy_state[3], numpy_state[4]
        )
    )

    # PyTorch の状態を CPU, GPU の順に復元
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if state["mps"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())

    # 登録された外部ライブラリの状態の復元
    # 片方にしか無い乱数は戻せないので警告 (登録の前のチェックポイントなど)
    extra = state.get("extra") or {}
    missing = sorted(set(EXTRA_RNGS) - set(extra))
    unknown = sorted(set(extra) - set(EXTRA_RNGS))
    if missing:
        warnings.warn(
            f"乱数 {missing} の状態が保存されていないので戻せません "
            "(登録する前に保存したチェックポイントなど)", stacklevel=2
        )
    if unknown:
        warnings.warn(
            f"保存された乱数 {unknown} は登録されていないので戻せません. "
            "register_rng で登録してください", stacklevel=2
        )
    for name, rng in EXTRA_RNGS.items():
        if name in extra:
            rng["set"](extra[name])
