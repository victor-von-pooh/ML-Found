from collections import namedtuple

import torch

from ml_found.utils.batch import get_batch_size, move_to_device
from ml_found.utils.device import get_device


def test_get_device_explicit():
    """
    指定したデバイスがそのまま返ることを確かめる関数
    """
    # cpu を指定して取得する
    device = get_device("cpu")

    assert device == torch.device("cpu")


def test_get_device_auto():
    """
    auto の場合に使えるデバイスが選ばれることを確かめる関数
    """
    # 自動で選ばせる
    device = get_device("auto")

    assert device.type in ("cuda", "mps", "cpu")


def test_move_to_device_nested():
    """
    入れ子のバッチが型を保ったまま移ることを確かめる関数
    """
    # テンソル, リスト, タプル, 辞書, テンソル以外を混ぜたバッチを作る
    batch = {
        "x": torch.zeros(2, 3),
        "pair": (torch.ones(2), [torch.ones(1), 5]),
        "name": "sample"
    }

    # cpu に移す
    moved = move_to_device(batch, torch.device("cpu"))

    assert isinstance(moved, dict)
    assert isinstance(moved["pair"], tuple)
    assert isinstance(moved["pair"][1], list)
    assert moved["pair"][1][1] == 5
    assert moved["name"] == "sample"
    assert moved["x"].device.type == "cpu"


def test_move_to_device_namedtuple():
    """
    namedtuple が同じ型のまま移ることを確かめる関数
    """
    # namedtuple のバッチを作る
    Batch = namedtuple("Batch", ["x", "y"])
    batch = Batch(x=torch.zeros(2), y=torch.ones(2))

    # cpu に移す
    moved = move_to_device(batch, torch.device("cpu"))

    assert isinstance(moved, Batch)
    assert torch.equal(moved.y, torch.ones(2))


def test_get_batch_size():
    """
    バッチのサンプル数を推定できることを確かめる関数
    """
    # テンソル, タプル, 辞書, テンソルを含まないものを調べる
    x = torch.zeros(4, 3)

    assert get_batch_size(x) == 4
    assert get_batch_size((x, torch.zeros(4))) == 4
    assert get_batch_size({"name": "a", "x": [x]}) == 4
    assert get_batch_size(["a", 1]) == 1
