import csv
from pathlib import Path

import pytest
import torch

from ml_found.training.history import History, MetricTracker
from ml_found.utils.plotting import plot_history


def make_history() -> History:
    """
    テスト用の記録を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    history: History
        3 エポック分の記録, 2 エポック目だけ valid が無い
    """
    # train, valid, 学習率を記録
    history = History()
    history.append(
        {"epoch": 1, "train/loss": 1.0, "valid/loss": 1.2, "lr": 0.1}
    )
    history.append({"epoch": 2, "train/loss": 0.8, "lr": 0.1})
    history.append(
        {"epoch": 3, "train/loss": 0.5, "valid/loss": 0.7, "lr": 0.01}
    )

    return history


def test_metric_tracker_weighted_mean() -> None:
    """
    サンプル数で重み付けした平均が求まることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # サンプル数の違う 2 バッチの追加
    tracker = MetricTracker()
    tracker.update({"loss": torch.tensor(1.0), "acc": 0.5}, n=3)
    tracker.update({"loss": torch.tensor(3.0), "acc": 1.0}, n=1)
    means = tracker.compute()

    assert means["loss"] == pytest.approx(1.5)
    assert means["acc"] == pytest.approx(0.625)

    # 空にすると何も残らないことの確認
    tracker.reset()

    assert tracker.compute() == {}


def test_history_get_and_keys() -> None:
    """
    途中で欠けた項目も含めて取り出せることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 記録の作成
    history = make_history()

    assert history.keys() == ["epoch", "train/loss", "valid/loss", "lr"]
    assert history.get("valid/loss") == [1.2, None, 0.7]


def test_history_save_csv(tmp_path: Path) -> None:
    """
    CSV に 1 エポック 1 行で保存されることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # CSV に保存して読み込み
    path = tmp_path / "history.csv"
    make_history().save_csv(str(path))
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    assert rows[1]["valid/loss"] == ""
    assert rows[2]["lr"] == "0.01"


def test_history_state_dict() -> None:
    """
    state_dict から記録を戻せることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 記録を別の History に移動
    history = make_history()
    restored = History()
    restored.load_state_dict(history.state_dict())

    assert restored.records == history.records


def test_plot_history(tmp_path: Path) -> None:
    """
    指標ごとに学習曲線の画像が保存されることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 学習曲線を保存
    paths = plot_history(make_history().records, str(tmp_path))

    assert [path.name for path in paths] == ["loss_curve.png", "lr_curve.png"]
    assert all(path.exists() for path in paths)


def test_history_state_dict_columns() -> None:
    """
    記録をチェックポイントに列ごとの数の配列で保存し, 同じ記録に戻せることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 途中から現れる列と, 値の無い記録を含む
    history = History()
    history.append({"epoch": 1, "train/loss": 0.5, "lr": 0.1})
    history.append({"epoch": 2, "train/loss": 0.4, "lr": 0.1})
    state = history.state_dict()
    history.append(
        {"epoch": 3, "train/loss": 0.3, "train_eval/loss": 0.25, "lr": None}
    )
    state = history.state_dict()
    assert state["columns"] == ["epoch", "train/loss", "lr", "train_eval/loss"]
    assert all(isinstance(column, torch.Tensor) for column in state["values"])

    # 読み込むと同じ記録 (epoch は整数, 無い値は無いまま)
    restored = History()
    restored.load_state_dict(state)
    assert restored.records == [
        {"epoch": 1, "train/loss": 0.5, "lr": 0.1},
        {"epoch": 2, "train/loss": 0.4, "lr": 0.1},
        {"epoch": 3, "train/loss": 0.3, "train_eval/loss": 0.25},
    ]
    assert isinstance(restored.records[0]["epoch"], int)

    # 保存した後に記録が増えても, 前に取り出した状態は変わらない
    history.append({"epoch": 4, "train/loss": 0.2})
    assert len(state["values"][0]) == 3
    assert len(history.state_dict()["values"][0]) == 4

    # 整数の列は整数に戻る
    counts = History()
    counts.append({"epoch": 1, "n": 7, "acc": 0.5})
    counts_restored = History()
    counts_restored.load_state_dict(counts.state_dict())
    assert counts_restored.records == [{"epoch": 1, "n": 7, "acc": 0.5}]
    assert isinstance(counts_restored.records[0]["n"], int)

    # 以前の形 (辞書のリスト) も読める
    old = History()
    old.load_state_dict({"records": [{"epoch": 1, "train/loss": 0.5}]})
    assert old.records == [{"epoch": 1, "train/loss": 0.5}]

    # 記録を置き換えた場合は作り直す
    restored.records = [{"epoch": 1, "valid/loss": 0.9}]
    assert restored.state_dict()["columns"] == ["epoch", "valid/loss"]
