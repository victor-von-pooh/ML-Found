import torch


def check_device(device: torch.device) -> None:
    """
    指定されたデバイスが使えることを確かめる関数

    Parameters
    ----------
    device: torch.device
        確かめるデバイス

    Returns
    ----------
    None
    """
    # CUDA が使えない場合や, GPU の番号が存在しない場合はエラー
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA が使えません")
        if device.index is not None and (
            device.index >= torch.cuda.device_count()
        ):
            raise ValueError(f"GPU {device.index} がありません")

    # MPS が使えない場合はエラー
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS が使えません")


def get_device(device: str = "auto") -> torch.device:
    """
    学習に使うデバイスを取得する関数

    Parameters
    ----------
    device: str = "auto"
        デバイス名, auto の場合は使えるものから自動で選ぶ

    Returns
    ----------
    selected: torch.device
        学習に使うデバイス
    """
    # 指定されている場合は使えることを確かめてから使用
    if device != "auto":
        try:
            selected = torch.device(device)
        except RuntimeError as e:
            raise ValueError(f"デバイス名が正しくありません: {device}") from e
        check_device(selected)
        return selected

    # cuda, mps, cpu の順に使えるものの選択
    if torch.cuda.is_available():
        selected = torch.device("cuda")
    elif torch.backends.mps.is_available():
        selected = torch.device("mps")
    else:
        selected = torch.device("cpu")

    return selected
