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
