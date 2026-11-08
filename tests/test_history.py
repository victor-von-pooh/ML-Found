import csv
from pathlib import Path

import pytest
import torch

from ml_found.training.history import History, MetricTracker
from ml_found.utils.plotting import plot_history


def make_history() -> History:
    """
    テスト用の記録を作る関数

    Returns
    -------
    history: History
        3 エポック分の記録, 2 エポック目だけ valid が無い
    """
    # train, valid, 学習率を記録する
    history = History()
    history.append(
        {"epoch": 1, "train/loss": 1.0, "valid/loss": 1.2, "lr": 0.1}
    )
    history.append({"epoch": 2, "train/loss": 0.8, "lr": 0.1})
    history.append(
        {"epoch": 3, "train/loss": 0.5, "valid/loss": 0.7, "lr": 0.01}
    )

    return history


def test_metric_tracker_weighted_mean():
    """
    サンプル数で重み付けした平均が求まることを確かめる関数
    """
    # サンプル数の違う 2 バッチを加える
    tracker = MetricTracker()
    tracker.update({"loss": torch.tensor(1.0), "acc": 0.5}, n=3)
    tracker.update({"loss": torch.tensor(3.0), "acc": 1.0}, n=1)
    means = tracker.compute()

    assert means["loss"] == pytest.approx(1.5)
    assert means["acc"] == pytest.approx(0.625)

    # 空にすると何も残らない
    tracker.reset()

    assert tracker.compute() == {}


def test_history_get_and_keys():
    """
    途中で欠けた項目も含めて取り出せることを確かめる関数
    """
    # 記録を作る
    history = make_history()

    assert history.keys() == ["epoch", "train/loss", "valid/loss", "lr"]
    assert history.get("valid/loss") == [1.2, None, 0.7]


def test_history_save_csv(tmp_path: Path):
    """
    CSV に 1 エポック 1 行で保存されることを確かめる関数
    """
    # CSV に保存して読み込む
    path = tmp_path / "history.csv"
    make_history().save_csv(str(path))
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    assert rows[1]["valid/loss"] == ""
    assert rows[2]["lr"] == "0.01"


def test_history_state_dict():
    """
    state_dict から記録を戻せることを確かめる関数
    """
    # 記録を別の History に移す
    history = make_history()
    restored = History()
    restored.load_state_dict(history.state_dict())

    assert restored.records == history.records


def test_plot_history(tmp_path: Path):
    """
    指標ごとに学習曲線の画像が保存されることを確かめる関数
    """
    # 学習曲線を保存する
    paths = plot_history(make_history().records, str(tmp_path))

    assert [path.name for path in paths] == ["loss_curve.png", "lr_curve.png"]
    assert all(path.exists() for path in paths)
