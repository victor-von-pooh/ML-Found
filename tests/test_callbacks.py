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
) -> None:
    """
    向きと min_delta に応じて改善を判定できることを確かめる関数

    Parameters
    ----------
    value: float
        今回の指標の値
    best: float
        これまでの最良値
    mode: str
        良し悪しの向き
    min_delta: float
        改善とみなす最小の変化
    expected: bool
        改善とみなされるべきかどうか

    Returns
    ----------
    None
    """
    assert is_improved(value, best, mode, min_delta) == expected


def test_early_stopping() -> None:
    """
    patience 回続けて改善しなければ止まることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 2 回まで待つ Early Stopping の作成
    early_stopping = EarlyStopping(patience=2)

    # 改善, 悪化, 改善, 悪化, 悪化の順に受け渡し
    results = [early_stopping.step(v) for v in [1.0, 1.2, 0.8, 0.9, 0.9]]

    assert results == [False, False, False, False, True]
    assert early_stopping.best == 0.8
    assert early_stopping.stopped is True


def test_early_stopping_state_dict() -> None:
    """
    state_dict から状態を戻せることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 1 回悪化した状態を別の Early Stopping に移動
    early_stopping = EarlyStopping(patience=2)
    early_stopping.step(1.0)
    early_stopping.step(1.5)
    restored = EarlyStopping(patience=2)
    restored.load_state_dict(early_stopping.state_dict())

    # もう 1 回悪化すると停止
    assert restored.stopped is False
    assert restored.step(1.5) is True


def test_checkpoint_restores_best_weights(tmp_path: Path) -> None:
    """
    重みが書き換わってもベストモデルの重みに戻せることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # ベストモデルを記録した後に, optimizer.step のように重みを直接書き換え
    model = nn.Linear(2, 1)
    checkpoint = Checkpoint(tmp_path)
    best_weight = model.weight.detach().clone()
    checkpoint.update(1.0, 1, model)
    with torch.no_grad():
        model.weight.add_(10.0)
    checkpoint.update(2.0, 2, model)

    # ベストモデルの重みの復元
    checkpoint.restore_best(model)

    assert torch.equal(model.weight, best_weight)
    assert checkpoint.best_epoch == 1


def test_checkpoint_files(tmp_path: Path) -> None:
    """
    best.pt はベストのエポック, last.pt は最後のエポックで保存されることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 改善, 悪化の順に記録して保存
    model = nn.Linear(2, 1)
    checkpoint = Checkpoint(tmp_path)
    for epoch, value in [(1, 1.0), (2, 2.0)]:
        improved = checkpoint.update(value, epoch, model)
        checkpoint.save({"epoch": epoch}, improved)

    # 保存したファイルの読み込み
    best = load_checkpoint(str(tmp_path / "best.pt"), torch.device("cpu"))
    last = load_checkpoint(str(tmp_path / "last.pt"), torch.device("cpu"))

    assert best["epoch"] == 1
    assert last["epoch"] == 2
    assert not (tmp_path / "best.pt.tmp").exists()


def test_checkpoint_without_best_file(tmp_path: Path) -> None:
    """
    best.pt を保存しない場合はベストモデルの重みを状態に含めることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # best.pt を保存しない設定で記録
    model = nn.Linear(2, 1)
    checkpoint = Checkpoint(tmp_path, save_best=False, save_last=False)
    best_weight = model.weight.detach().clone()
    improved = checkpoint.update(1.0, 1, model)
    checkpoint.save({"epoch": 1}, improved)

    # 状態を別の Checkpoint に移してから重みを書き換えたモデルに復元
    restored = Checkpoint(tmp_path, save_best=False)
    restored.load_state_dict(checkpoint.state_dict())
    with torch.no_grad():
        model.weight.add_(10.0)
    restored.restore_best(model)

    assert torch.equal(model.weight, best_weight)
    assert list(tmp_path.iterdir()) == []


def test_invalid_mode(tmp_path: Path) -> None:
    """
    min と max 以外の向きでエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    with pytest.raises(ValueError):
        EarlyStopping(patience=1, mode="minimum")
    with pytest.raises(ValueError):
        Checkpoint(tmp_path, mode="minimum")


def test_load_checkpoint_keeps_unmovable_tensors(tmp_path: Path) -> None:
    """
    デバイスに移せないテンソルを cpu に残して読み込むことを確かめる関数

    mps は float64 を扱えないので, 更新ごとの学習率の記録などを含む
    チェックポイントも読み込めるようにする

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    path = tmp_path / "state.pt"
    torch.save({
        "model": {"weight": torch.ones(2)},
        "lr_steps": {"lr": torch.zeros(3, dtype=torch.float64)},
        "items": [torch.ones(1), 2]
    }, path)

    # 使えるデバイスに読み込み, mps でも float64 は cpu のまま
    available = torch.backends.mps.is_available()
    device = torch.device("mps" if available else "cpu")
    state = load_checkpoint(str(path), device)
    assert state["model"]["weight"].device.type == device.type
    assert state["lr_steps"]["lr"].dtype == torch.float64
    assert state["items"][1] == 2
