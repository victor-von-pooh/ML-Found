from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import ml_found
from ml_found import (
    Trainer, average_precision, load_evaluations, precision_at_k,
    save_evaluation
)
from test_trainer import make_experiment, make_loader, regression_step

# docs/DESIGN.md の照合の見本 (度合い, 正解, PR-AUC, K ごとの適合率)
SAMPLES = {
    "ties": (
        [0.9, 0.8, 0.8, 0.5, 0.3, 0.3], [1, 0, 1, 0, 1, 0], 13 / 18,
        {1: 1.0, 2: 0.75, 3: 2 / 3, 5: 0.5, 10: 0.5}
    ),
    "top_normal": (
        [0.1, 0.4, 0.35, 0.8, 0.4], [0, 1, 1, 0, 0], 5 / 12,
        {2: 0.25, 3: 1 / 3}
    ),
    "all_tied": (
        [0.5, 0.5, 0.5, 0.5], [1, 0, 0, 1], 0.5, {1: 0.5, 3: 0.5}
    ),
    "no_anomaly": ([0.3, 0.2, 0.1], [0, 0, 0], 0.0, {2: 0.0})
}


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
    # PR-AUC と K ごとの適合率
    scores, labels, expected, at_k = SAMPLES[name]
    value = average_precision(scores, labels)
    assert isinstance(value, float)
    assert value == pytest.approx(expected, abs=1e-12)
    for k, expected_k in at_k.items():
        value_k = precision_at_k(scores, labels, k)
        assert isinstance(value_k, float)
        assert value_k == pytest.approx(expected_k, abs=1e-12), k


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_order_does_not_matter(name: str) -> None:
    """
    行を並べ替えても, 同点の行を含めて同じ値になることを確かめる関数

    Parameters
    ----------
    name: str
        見本の名前

    Returns
    ----------
    None
    """
    # 逆の順と, 決まった乱数の順
    scores, labels, expected, at_k = SAMPLES[name]
    generator = torch.Generator().manual_seed(0)
    orders = [
        list(range(len(scores)))[::-1],
        torch.randperm(len(scores), generator=generator).tolist()
    ]
    for order in orders:
        moved_scores = [scores[i] for i in order]
        moved_labels = [labels[i] for i in order]
        assert average_precision(moved_scores, moved_labels) == (
            pytest.approx(expected, abs=1e-12)
        )
        for k, expected_k in at_k.items():
            assert precision_at_k(moved_scores, moved_labels, k) == (
                pytest.approx(expected_k, abs=1e-12)
            )


@pytest.mark.parametrize("convert", [
    lambda s, y: (np.array(s), np.array(y, dtype=bool)),
    lambda s, y: (np.array(s, dtype=np.float32), np.array(y)),
    lambda s, y: (
        torch.tensor(s, requires_grad=True), torch.tensor(y, dtype=torch.bool)
    ),
    lambda s, y: (torch.tensor(s, dtype=torch.float16), torch.tensor(y)),
    lambda s, y: (s, [float(v) for v in y])
])
def test_input_types(convert: Any) -> None:
    """
    テンソル, NumPy の配列, 数の並び, 真偽値の正解で同じ値になることを
    確かめる関数

    Parameters
    ----------
    convert: Any
        見本の度合いと正解を入力の型に写す関数

    Returns
    ----------
    None
    """
    # 半精度でも見本の度合いの大小は変わらない
    scores, labels, expected, at_k = SAMPLES["ties"]
    moved_scores, moved_labels = convert(scores, labels)
    assert average_precision(moved_scores, moved_labels) == (
        pytest.approx(expected, abs=1e-12)
    )
    assert precision_at_k(moved_scores, moved_labels, 2) == 0.75


def test_infinite_scores_are_ties() -> None:
    """
    無限大の度合いが使え, 同じ符号の無限大どうしが同点になることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 見本の 0.9 と 0.8 の同点を無限大に置き換えても同じ値
    inf = float("inf")
    scores = [inf, 2.0, 2.0, 0.5, -inf, -inf]
    labels = [1, 0, 1, 0, 1, 0]
    assert average_precision(scores, labels) == pytest.approx(13 / 18)
    assert precision_at_k(scores, labels, 5) == pytest.approx(0.5)
    assert average_precision([inf, inf], [1, 0]) == pytest.approx(0.5)


@pytest.mark.parametrize("scores, labels, message", [
    ([0.1, float("nan")], [0, 1], "nan"),
    ([0.1, 0.2], [0, 2], "0 と 1"),
    ([0.1, 0.2], [0, 0.5], "0 と 1"),
    ([0.1, 0.2], [0, float("nan")], "0 と 1"),
    ([0.1, 0.2], [0, 1, 1], "長さ"),
    ([[0.1, 0.2]], [[0, 1]], "1 次元"),
    ([], [], "空")
])
def test_invalid_inputs(scores: Any, labels: Any, message: str) -> None:
    """
    nan の度合い, 0 と 1 でない正解, 形の誤り, 空の入力が誤りになることを
    確かめる関数

    Parameters
    ----------
    scores: Any
        度合い
    labels: Any
        正解
    message: str
        誤りのメッセージに含まれる文字

    Returns
    ----------
    None
    """
    # 2 つの関数とも同じ誤り
    with pytest.raises(ValueError, match=message):
        average_precision(scores, labels)
    with pytest.raises(ValueError, match=message):
        precision_at_k(scores, labels, 1)


@pytest.mark.parametrize("k", [0, -1, 1.5, True, "3", None])
def test_invalid_k(k: Any) -> None:
    """
    1 以上の整数でない K が誤りになることを確かめる関数

    Parameters
    ----------
    k: Any
        上位から数える行の数

    Returns
    ----------
    None
    """
    # 0 以下, 小数, 真偽値, 文字, None
    with pytest.raises(ValueError, match="k は 1 以上の整数"):
        precision_at_k([0.2, 0.1], [1, 0], k)


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
    # 見本の K = 2
    scores, labels, _, _ = SAMPLES["ties"]
    assert precision_at_k(scores, labels, np.int64(2)) == 0.75


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
    for name in ["average_precision", "precision_at_k"]:
        assert name in ml_found.__all__
        assert name in ml_found.training.__all__


class AnomalyTrainer(Trainer):
    def on_epoch_end(self, phase: str) -> dict:
        """
        検証のときだけ, 決まった見本の PR-AUC と上位 K 件の適合率を返す関数

        Parameters
        ----------
        phase: str
            train, train_eval, valid のどれか

        Returns
        ----------
        metrics: dict
            検証の場合は pr_auc と precision_at_2, それ以外は空の辞書
        """
        # 検証の他は何もしない
        metrics = {}
        if phase != "valid":
            return metrics

        # 見本の度合いと正解
        scores, labels, _, _ = SAMPLES["ties"]
        metrics = {
            "pr_auc": average_precision(scores, labels),
            "precision_at_2": precision_at_k(scores, labels, 2)
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
    # valid/pr_auc を監視して学習
    experiment = make_experiment(
        tmp_path,
        "trainer:\n  epochs: 2\n  monitor: valid/pr_auc\n  mode: max\n"
    )
    trainer = AnomalyTrainer(nn.Linear(1, 1), experiment, regression_step)
    history = trainer.fit(make_loader(), make_loader())
    assert history.get("valid/pr_auc") == pytest.approx([13 / 18] * 2)
    assert history.get("valid/precision_at_2") == [0.75, 0.75]

    # 学習の後の評価の表
    scores, labels, _, _ = SAMPLES["ties"]
    save_evaluation(
        experiment, "anomaly",
        {
            "pr_auc": average_precision(scores, labels),
            "precision_at_2": precision_at_k(scores, labels, 2)
        },
        "test", better={"pr_auc": "higher", "precision_at_2": "higher"}
    )
    evaluation = load_evaluations(experiment.run_dir)[0]
    assert evaluation["name"] == "anomaly"
    assert evaluation["better"] == {
        "pr_auc": "higher", "precision_at_2": "higher"
    }
