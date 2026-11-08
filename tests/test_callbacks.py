from pathlib import Path

import pytest
import torch
from torch import nn

from ml_found.training.callbacks import (
    Checkpoint, EarlyStopping, is_improved, load_checkpoint
)


@pytest.mark.parametrize(
    "value, best, mode, min_delta, expected",
    [
        (1.0, None, "min", 0.0, True),
        (0.9, 1.0, "min", 0.0, True),
        (0.95, 1.0, "min", 0.1, False),
        (1.1, 1.0, "max", 0.0, True),
        (0.9, 1.0, "max", 0.0, False)
    ]
)
def test_is_improved(
    value: float, best: float, mode: str, min_delta: float, expected: bool
):
    """
    向きと min_delta に応じて改善を判定できることを確かめる関数
    """
    assert is_improved(value, best, mode, min_delta) == expected


def test_early_stopping():
    """
    patience 回続けて改善しなければ止まることを確かめる関数
    """
    # 2 回まで待つ Early Stopping を作る
    early_stopping = EarlyStopping(patience=2)

    # 改善, 悪化, 改善, 悪化, 悪化の順に渡す
    results = [early_stopping.step(v) for v in [1.0, 1.2, 0.8, 0.9, 0.9]]

    assert results == [False, False, False, False, True]
    assert early_stopping.best == 0.8


def test_early_stopping_state_dict():
    """
    state_dict から状態を戻せることを確かめる関数
    """
    # 1 回悪化した状態を別の Early Stopping に移す
    early_stopping = EarlyStopping(patience=2)
    early_stopping.step(1.0)
    early_stopping.step(1.5)
    restored = EarlyStopping(patience=2)
    restored.load_state_dict(early_stopping.state_dict())

    # もう 1 回悪化すると止まる
    assert restored.step(1.5) is True


def test_checkpoint_restores_best_weights(tmp_path: Path):
    """
    重みが書き換わってもベストモデルの重みに戻せることを確かめる関数
    """
    # ベストモデルを記録した後に, optimizer.step のように重みを直接書き換える
    model = nn.Linear(2, 1)
    checkpoint = Checkpoint(tmp_path)
    best_weight = model.weight.detach().clone()
    checkpoint.update(1.0, 1, model, {"epoch": 1})
    with torch.no_grad():
        model.weight.add_(10.0)
    checkpoint.update(2.0, 2, model, {"epoch": 2})

    # ベストモデルの重みを戻す
    checkpoint.restore_best(model)

    assert torch.equal(model.weight, best_weight)
    assert checkpoint.best_epoch == 1


def test_checkpoint_files(tmp_path: Path):
    """
    best.pt はベストのエポック, last.pt は最後のエポックで保存されることを確かめる関数
    """
    # 改善, 悪化の順に渡す
    model = nn.Linear(2, 1)
    checkpoint = Checkpoint(tmp_path)
    checkpoint.update(1.0, 1, model, {"epoch": 1})
    checkpoint.update(2.0, 2, model, {"epoch": 2})

    # 保存したファイルを読み込む
    best = load_checkpoint(str(tmp_path / "best.pt"), torch.device("cpu"))
    last = load_checkpoint(str(tmp_path / "last.pt"), torch.device("cpu"))

    assert best["epoch"] == 1
    assert last["epoch"] == 2
    assert not (tmp_path / "best.pt.tmp").exists()


def test_checkpoint_without_files(tmp_path: Path):
    """
    ファイルに保存しない場合もベストモデルの重みに戻せることを確かめる関数
    """
    # ファイルに保存しない設定で記録して重みを書き換える
    model = nn.Linear(2, 1)
    checkpoint = Checkpoint(tmp_path, save_best=False, save_last=False)
    best_weight = model.weight.detach().clone()
    checkpoint.update(1.0, 1, model, {"epoch": 1})
    with torch.no_grad():
        model.weight.add_(10.0)

    # ベストモデルの重みを戻す
    checkpoint.restore_best(model)

    assert torch.equal(model.weight, best_weight)
    assert list(tmp_path.iterdir()) == []


def test_invalid_mode(tmp_path: Path):
    """
    min と max 以外の向きでエラーになることを確かめる関数
    """
    with pytest.raises(ValueError):
        EarlyStopping(patience=1, mode="minimum")
    with pytest.raises(ValueError):
        Checkpoint(tmp_path, mode="minimum")
