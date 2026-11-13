from collections import namedtuple
from pathlib import Path

import torch
from torch import nn

from ml_found.utils.batch import get_batch_size, move_to_device
from ml_found.utils.device import get_device
from ml_found.utils.model_summary import (
    count_parameters, save_model_summary, summarize_model
)


def test_get_device_explicit() -> None:
    """
    指定したデバイスがそのまま返ることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # cpu を指定して取得
    device = get_device("cpu")

    assert device == torch.device("cpu")


def test_get_device_auto() -> None:
    """
    auto の場合に使えるデバイスが選ばれることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 自動で選択
    device = get_device("auto")

    assert device.type in ("cuda", "mps", "cpu")


def test_move_to_device_nested() -> None:
    """
    入れ子のバッチが型を保ったまま移ることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # テンソル, リスト, タプル, 辞書, テンソル以外を混ぜたバッチの作成
    batch = {
        "x": torch.zeros(2, 3),
        "pair": (torch.ones(2), [torch.ones(1), 5]),
        "name": "sample"
    }

    # cpu に移動
    moved = move_to_device(batch, torch.device("cpu"))

    assert isinstance(moved, dict)
    assert isinstance(moved["pair"], tuple)
    assert isinstance(moved["pair"][1], list)
    assert moved["pair"][1][1] == 5
    assert moved["name"] == "sample"
    assert moved["x"].device.type == "cpu"


def test_move_to_device_namedtuple() -> None:
    """
    namedtuple が同じ型のまま移ることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # namedtuple のバッチの作成
    Batch = namedtuple("Batch", ["x", "y"])
    batch = Batch(x=torch.zeros(2), y=torch.ones(2))

    # cpu に移動
    moved = move_to_device(batch, torch.device("cpu"))

    assert isinstance(moved, Batch)
    assert torch.equal(moved.y, torch.ones(2))


def test_get_batch_size() -> None:
    """
    バッチのサンプル数を推定できることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # テンソル, タプル, 辞書, テンソルを含まないものの確認
    x = torch.zeros(4, 3)

    assert get_batch_size(x) == 4
    assert get_batch_size((x, torch.zeros(4))) == 4
    assert get_batch_size({"name": "a", "x": [x]}) == 4
    assert get_batch_size(["a", 1]) == 1


def test_model_summary(tmp_path: Path) -> None:
    """
    パラメータとバッファの一覧と, 種類ごとの要素数を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 学習しない層とバッファを持つモデル
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
    model[0].bias.requires_grad_(False)
    rows = summarize_model(model)
    by_name = {row["name"]: row for row in rows}
    assert by_name["0.weight"] == {
        "name": "0.weight", "kind": "parameter", "module": "Linear",
        "shape": "4x3", "dtype": "float32", "numel": 12,
        "requires_grad": True, "same_as": ""
    }
    assert by_name["0.bias"]["requires_grad"] is False
    assert by_name["1.running_mean"]["kind"] == "buffer"
    assert by_name["1.num_batches_tracked"]["shape"] == ""
    assert count_parameters(rows) == {
        "trainable": 12 + 8, "frozen": 4, "buffer": 4 + 4 + 1
    }

    # CSV に保存
    path = tmp_path / "model_summary.csv"
    save_model_summary(rows, str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "name,kind,module,shape,dtype,numel,requires_grad,same_as"
    )
    assert len(lines) == len(rows) + 1

    # 同じ層を 2 回使うモデルは, 2 つ目の名前を最初の名前付きで出し,
    # 要素数は 1 回だけ数える
    layer = nn.Linear(2, 2)
    shared = nn.Sequential(layer, nn.ReLU(), layer)
    rows = summarize_model(shared)
    assert [row["name"] for row in rows] == [
        "0.weight", "0.bias", "2.weight", "2.bias"
    ]
    assert rows[2]["same_as"] == "0.weight"
    assert count_parameters(rows)["trainable"] == 6
