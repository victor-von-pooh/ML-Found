import torch


def get_device(device: str = "auto") -> torch.device:
    """
    学習に使うデバイスを取得する関数

    Parameters
    ----------
    device: str = "auto"
        デバイス名, auto の場合は使えるものから自動で選ぶ

    Returns
    -------
    selected: torch.device
        学習に使うデバイス
    """
    # 指定されている場合はそのまま使う
    if device != "auto":
        selected = torch.device(device)
        return selected

    # cuda, mps, cpu の順に使えるものを選ぶ
    if torch.cuda.is_available():
        selected = torch.device("cuda")
    elif torch.backends.mps.is_available():
        selected = torch.device("mps")
    else:
        selected = torch.device("cpu")

    return selected
