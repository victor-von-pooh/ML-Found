from typing import Any, Optional

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


def find_batch_size(batch: Any) -> Optional[int]:
    """
    バッチの中からサンプル数を探す関数

    Parameters
    ----------
    batch: Any
        テンソル, またはテンソルを含むリスト, タプル, 辞書など

    Returns
    -------
    size: Optional[int]
        見つかったサンプル数, 見つからない場合は None
    """
    # PyG の Batch のようにグラフの数を持つ場合はそれを使う
    if isinstance(getattr(batch, "num_graphs", None), int):
        size = batch.num_graphs
        return size

    # 1 次元以上のテンソルの場合は 0 次元目をサンプル数とみなす
    if isinstance(batch, torch.Tensor):
        size = batch.shape[0] if batch.dim() > 0 else None
        return size

    # 辞書の場合は値を, リストやタプルの場合は要素を順に調べる
    if isinstance(batch, dict):
        values = list(batch.values())
    elif isinstance(batch, (list, tuple)):
        values = list(batch)
    else:
        values = []

    # 見つかるまで次の要素を調べる
    size = None
    for value in values:
        size = find_batch_size(value)
        if size is not None:
            break

    return size


def get_batch_size(batch: Any) -> int:
    """
    バッチのサンプル数を推定する関数

    Parameters
    ----------
    batch: Any
        テンソル, またはテンソルを含むリスト, タプル, 辞書など

    Returns
    -------
    size: int
        最初に見つかったサンプル数, 見つからない場合は 1
    """
    # バッチの中から探し, 見つからない場合は 1 とみなす
    size = find_batch_size(batch)
    if size is None:
        size = 1

    return size
