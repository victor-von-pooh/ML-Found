from typing import Any

import torch


def move_to_device(batch: Any, device: torch.device) -> Any:
    """
    バッチの中身を再帰的にデバイスへ移す関数

    Parameters
    ----------
    batch: Any
        テンソル, またはテンソルを含むリスト, タプル, 辞書など
    device: torch.device
        移動先のデバイス

    Returns
    -------
    moved: Any
        デバイスに移したバッチ
    """
    # テンソルや PyG の Data のように to を持つものはそのまま移す
    if isinstance(batch, torch.Tensor) or hasattr(batch, "to"):
        moved = batch.to(device)
        return moved

    # 辞書の場合は値ごとに移す
    if isinstance(batch, dict):
        moved = {
            key: move_to_device(value, device)
            for key, value in batch.items()
        }
        return moved

    # namedtuple の場合はフィールドごとに移して同じ型に戻す
    if isinstance(batch, tuple) and hasattr(batch, "_fields"):
        moved = type(batch)(
            *[move_to_device(value, device) for value in batch]
        )
        return moved

    # リストやタプルの場合は要素ごとに移して同じ型に戻す
    if isinstance(batch, (list, tuple)):
        moved = type(batch)(
            move_to_device(value, device) for value in batch
        )
        return moved

    # それ以外の場合は何もしない
    moved = batch

    return moved
