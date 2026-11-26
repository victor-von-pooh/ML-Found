import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import ml_found
from ml_found import (
    Trainer, load_evaluations, ndcg_at_k, recall_at_k, save_evaluation
)
from test_trainer import make_experiment, make_loader, regression_step

# docs/DESIGN.md の照合の見本 (並べた品物, 正解, K ごとの Recall と NDCG)
LOG3 = math.log2(3)
SAMPLES = {
    "mixed": (
        [[10, 11, 12], [20, 21, 22], [], [40, 41, 42]],
        [{11, 13, 14}, set(), {30}, {40}],
        {
            1: (1 / 3, 1 / 3),
            2: (4 / 9, (1 / math.log2(6) + 1) / 3),
            3: (4 / 9, (1 / (1 + 1.5 * LOG3) + 1) / 3),
            5: (4 / 9, (1 / (1 + 1.5 * LOG3) + 1) / 3)
        }
    ),
    "many_relevant": (
        [[3, 1, 2, 7]], [{1, 2, 3, 4, 5}],
        {
            2: (2 / 5, 1.0),
            3: (3 / 5, 1.0),
            4: (3 / 5, (1.5 + 1 / LOG3) / (1.5 + 1 / LOG3
                                           + 1 / math.log2(5))),
            6: (3 / 5, (1.5 + 1 / LOG3) / (
                1.5 + 1 / LOG3 + 1 / math.log2(5) + 1 / math.log2(6)
            ))
        }
    ),
    "no_users": (
        [[1, 2], []], [set(), set()], {1: (0.0, 0.0), 2: (0.0, 0.0)}
    )
}


def skeleton_metrics(top: list, relevant: list, k: int) -> tuple:
    """
    ML-Quest の骨組みと ML-Inference と同じ数え方を, 素直に書いた関数

    Parameters
    ----------
    top: list
        利用者ごとの並べた品物
    relevant: list
        利用者ごとの正解の品物の集合
    k: int
        上位から数える品物の数

    Returns
    ----------
    values: tuple
        Recall@K と NDCG@K
    """
    # 正解の無い利用者は除き, 数える利用者がいなければ 0
    recalls, ndcgs = [], []
    for items, wanted in zip(top, relevant):
        if not wanted:
            continue
        hits = [r for r, item in enumerate(items[:k], 1) if item in wanted]
        recalls.append(len(hits) / len(wanted))
        ideal = sum(1 / math.log2(r + 1)
                    for r in range(1, min(k, len(wanted)) + 1))
        ndcgs.append(sum(1 / math.log2(r + 1) for r in hits) / ideal)
    count = max(len(recalls), 1)
    values = (sum(recalls) / count, sum(ndcgs) / count)

    return values


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_documented_samples(name: str) -> None:
    """
    docs/DESIGN.md の見本で, 手で数えた値が返ることを確かめる関数

    Parameters
    ----------
    name: str
        見本の名前

    Returns
    ----------
    None
    """
    # K ごとの Recall と NDCG
    top, relevant, at_k = SAMPLES[name]
    for k, (recall, ndcg) in at_k.items():
        value = recall_at_k(top, relevant, k)
        assert isinstance(value, float)
        assert value == pytest.approx(recall, abs=1e-12), k
        value = ndcg_at_k(top, relevant, k)
        assert isinstance(value, float)
        assert value == pytest.approx(ndcg, abs=1e-12), k
        assert skeleton_metrics(top, relevant, k) == pytest.approx(
            (recall, ndcg), abs=1e-12
        )


def test_empty_users() -> None:
    """
    利用者が 0 人の場合に 0 になることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 並びと辞書の両方
    assert recall_at_k([], [], 3) == 0.0
    assert ndcg_at_k([], [], 3) == 0.0
    assert recall_at_k({}, {"a": {1}}, 3) == 0.0
    assert ndcg_at_k({}, {"a": {1}}, 3) == 0.0


@pytest.mark.parametrize("seed", range(5))
def test_matches_skeleton(seed: int) -> None:
    """
    正解の無い利用者, 上位が空の利用者, K より多い正解を含む乱数の見本で,
    骨組みと同じ数え方の値になることを確かめる関数

    Parameters
    ----------
    seed: int
        乱数の種

    Returns
    ----------
    None
    """
    # 30 人の利用者と 20 個の品物
    generator = random.Random(seed)
    top, relevant = [], []
    for _ in range(30):
        top.append(generator.sample(range(20), generator.randint(0, 8)))
        relevant.append(set(generator.sample(range(20),
                                             generator.randint(0, 6))))
    for k in [1, 3, 5, 10, 25]:
        recall, ndcg = skeleton_metrics(top, relevant, k)
        assert recall_at_k(top, relevant, k) == pytest.approx(recall)
        assert ndcg_at_k(top, relevant, k) == pytest.approx(ndcg)


def test_mapping_inputs() -> None:
    """
    辞書の場合は並べた品物の辞書の利用者を数え, 正解の辞書にいない
    利用者は正解が無いとみなすことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 見本の mixed の辞書と, 並べていない利用者 e の正解
    top, relevant, at_k = SAMPLES["mixed"]
    users = ["a", "b", "c", "d"]
    top_map = dict(zip(users, top))
    relevant_map = {u: r for u, r in zip(users, relevant) if r}
    relevant_map["e"] = {1, 2}
    for k, (recall, ndcg) in at_k.items():
        assert recall_at_k(top_map, relevant_map, k) == pytest.approx(recall)
        assert ndcg_at_k(top_map, relevant_map, k) == pytest.approx(ndcg)


@pytest.mark.parametrize("convert", [
    lambda t, r: (t, r),
    lambda t, r: (tuple(tuple(i) for i in t), tuple(tuple(i) for i in r)),
    lambda t, r: (
        [np.array(i, dtype=np.int64) for i in t],
        [np.array(sorted(i), dtype=np.int64) for i in r]
    ),
    lambda t, r: (
        [torch.tensor(i, dtype=torch.long) for i in t],
        [torch.tensor(sorted(i), dtype=torch.long) for i in r]
    ),
    lambda t, r: (t, [list(i) + list(i) for i in r]),
    lambda t, r: (
        [[f"item{n}" for n in i] for i in t],
        [frozenset(f"item{n}" for n in i) for i in r]
    )
])
def test_input_types(convert: Any) -> None:
    """
    リスト, タプル, NumPy の配列, テンソル, 重複を含む正解, 名前の品物で
    同じ値になることを確かめる関数

    Parameters
    ----------
    convert: Any
        見本の並べた品物と正解を入力の型に写す関数

    Returns
    ----------
    None
    """
    # 見本の mixed の K = 2
    top, relevant, at_k = SAMPLES["mixed"]
    moved_top, moved_relevant = convert(top, relevant)
    recall, ndcg = at_k[2]
    assert recall_at_k(moved_top, moved_relevant, 2) == pytest.approx(recall)
    assert ndcg_at_k(moved_top, moved_relevant, 2) == pytest.approx(ndcg)


def test_repeated_items_after_k() -> None:
    """
    上位 K 件より後の同じ品物は誤りにならないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 2 件目までは別の品物
    assert recall_at_k([[1, 2, 1]], [{1}], 2) == 1.0


@pytest.mark.parametrize("top, relevant, message", [
    ([[1, 1]], [{1}], "同じ品物が 2 度"),
    ([{1, 2}], [{1}], "リストかタプル"),
    ([[1]], ["1"], "集合, リストかタプル"),
    (["12"], [{1}], "リストかタプル"),
    ([[1]], [None], "集合, リストかタプル"),
    ([[[1]]], [{1}], "番号や名前"),
    ([[1]], [[[1]]], "番号や名前"),
    ([np.array([[1]])], [{1}], "1 次元"),
    ([[1], [2]], [{1}], "利用者の数"),
    ({"a": [1]}, [{1}], "両方とも"),
    ([[1]], {"a": {1}}, "両方とも"),
    ("12", "12", "リストかタプル"),
    (np.array([[1]]), [{1}], "リストかタプル")
])
def test_invalid_inputs(top: Any, relevant: Any, message: str) -> None:
    """
    形の誤り, 同じ品物の重複, 集合にできない品物が誤りになることを
    確かめる関数

    Parameters
    ----------
    top: Any
        並べた品物
    relevant: Any
        正解
    message: str
        誤りのメッセージに含まれる文字

    Returns
    ----------
    None
    """
    # 2 つの関数とも同じ誤り
    with pytest.raises(ValueError, match=message):
        recall_at_k(top, relevant, 2)
    with pytest.raises(ValueError, match=message):
        ndcg_at_k(top, relevant, 2)


@pytest.mark.parametrize("k", [0, -1, 1.5, True, "3", None])
def test_invalid_k(k: Any) -> None:
    """
    1 以上の整数でない K が誤りになることを確かめる関数

    Parameters
    ----------
    k: Any
        上位から数える品物の数

    Returns
    ----------
    None
    """
    # 利用者のいない場合も誤り
    for top, relevant in [([[1]], [{1}]), ([], [])]:
        with pytest.raises(ValueError, match="k は 1 以上の整数"):
            recall_at_k(top, relevant, k)
        with pytest.raises(ValueError, match="k は 1 以上の整数"):
            ndcg_at_k(top, relevant, k)


def test_numpy_integer_k() -> None:
    """
    NumPy の整数の K が使えることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 見本の mixed の K = 2
    top, relevant, at_k = SAMPLES["mixed"]
    assert recall_at_k(top, relevant, np.int64(2)) == pytest.approx(
        at_k[2][0]
    )
    assert ndcg_at_k(top, relevant, np.int64(2)) == pytest.approx(
        at_k[2][1]
    )


def test_public_names() -> None:
    """
    2 つの関数が公開の __all__ にあることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # トップレベルと training の両方
    for name in ["ndcg_at_k", "recall_at_k"]:
        assert name in ml_found.__all__
        assert name in ml_found.training.__all__


class RankingTrainer(Trainer):
    def on_epoch_end(self, phase: str) -> dict:
        """
        検証のときだけ, 決まった見本の Recall@K と NDCG@K を返す関数

        Parameters
        ----------
        phase: str
            train, train_eval, valid のどれか

        Returns
        ----------
        metrics: dict
            検証の場合は recall_at_2 と ndcg_at_2, それ以外は空の辞書
        """
        # 検証の他は空の辞書
        metrics = {}
        if phase != "valid":
            return metrics

        # 見本の並べた品物と正解
        top, relevant, _ = SAMPLES["mixed"]
        metrics = {
            "recall_at_2": recall_at_k(top, relevant, 2),
            "ndcg_at_2": ndcg_at_k(top, relevant, 2)
        }

        return metrics


def test_used_in_training_and_evaluation(tmp_path: Path) -> None:
    """
    on_epoch_end の監視する指標と, save_evaluation の表に使えることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # valid/ndcg_at_2 を監視する学習
    experiment = make_experiment(
        tmp_path,
        "trainer:\n  epochs: 2\n  monitor: valid/ndcg_at_2\n  mode: max\n"
    )
    trainer = RankingTrainer(nn.Linear(1, 1), experiment, regression_step)
    history = trainer.fit(make_loader(), make_loader())
    recall, ndcg = SAMPLES["mixed"][2][2]
    assert history.get("valid/ndcg_at_2") == pytest.approx([ndcg] * 2)
    assert history.get("valid/recall_at_2") == pytest.approx([recall] * 2)

    # 学習の後の評価の表
    top, relevant, _ = SAMPLES["mixed"]
    save_evaluation(
        experiment, "ranking",
        {
            "recall_at_2": recall_at_k(top, relevant, 2),
            "ndcg_at_2": ndcg_at_k(top, relevant, 2)
        },
        "test", better={"recall_at_2": "higher", "ndcg_at_2": "higher"}
    )
    evaluation = load_evaluations(experiment.run_dir)[0]
    assert evaluation["name"] == "ranking"
    assert evaluation["better"] == {
        "recall_at_2": "higher", "ndcg_at_2": "higher"
    }
