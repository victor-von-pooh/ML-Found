import math
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import ml_found
from ml_found import (
    Trainer, load_evaluations, qini_coefficient, qini_curve, save_evaluation,
    uplift_at_k
)
from test_trainer import make_experiment, make_loader, regression_step

NAN = math.nan

# docs/DESIGN.md の照合の見本 (見積もり, 処置, 結果, 曲線の点, 係数,
# K ごとの効果の差)
SAMPLES = {
    "ties": (
        [0.9, 0.8, 0.8, 0.6, 0.5, 0.5, 0.3, 0.2], [1, 1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 1],
        [(0, 0), (1, 1), (3, -1), (4, -1), (6, 0), (7, 2 / 3), (8, 0)],
        -5 / 72, {0.25: -1 / 3, 0.3: -1 / 3, 0.75: 0.0, 1: 0.0}
    ),
    "top_treated": (
        [0.9, 0.7, 0.7, 0.4, 0.4, 0.1], [1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 1], [(0, 0), (1, 1), (3, 2), (5, 2), (6, 1)],
        12 / 13, {0.25: NAN, 0.5: 1.0, 1: 1 / 3}
    ),
    "amount": (
        [3.0, 2.0, 2.0, 1.0, 0.5, 0.5], [1, 1, 0, 1, 0, 0],
        [500, 0, 200, 300, 0, 100],
        [(0, 0), (1, 500), (3, 100), (4, 200), (6, 500)],
        2 / 23, {0.25: NAN, 0.5: 50.0, 1: 500 / 3}
    ),
    "all_tied": (
        [0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0], [1, 0, 0, 1], [(0, 0), (4, 0)],
        0.0, {0.25: 0.0, 0.5: 0.0}
    ),
    "zero_outcome": (
        [0.3, 0.2, 0.1], [1, 0, 1], [0, 0, 0],
        [(0, 0), (1, 0), (2, 0), (3, 0)], 0.0, {0.5: NAN, 1: 0.0}
    )
}


def assert_same(value: float, expected: float) -> None:
    """
    nan どうしも等しいとして, 値が見本と同じことを確かめる関数

    Parameters
    ----------
    value: float
        数えた値
    expected: float
        見本の値

    Returns
    ----------
    None
    """
    # nan の見本と数の見本
    assert isinstance(value, float)
    if math.isnan(expected):
        assert math.isnan(value)
    else:
        assert value == pytest.approx(expected, abs=1e-12)


def reference_curve(
    scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    scikit-uplift の qini_curve を写した, 並べ替えで数える曲線の関数

    Parameters
    ----------
    scores: np.ndarray
        見積もり
    treatment: np.ndarray
        処置
    outcome: np.ndarray
        結果

    Returns
    ----------
    curve: tuple[np.ndarray, np.ndarray]
        先頭に 0 を足した上位の行の数と Qini の値
    """
    # 見積もりの高い順の並べ替えと, 同点のまとまりの終わり
    order = np.argsort(scores, kind="mergesort")[::-1]
    scores = scores[order]
    treatment = treatment[order]
    outcome = outcome[order]
    ends = np.r_[np.where(np.diff(scores))[0], scores.size - 1]

    # 終わりごとの累積の数と和
    treated = np.cumsum(treatment)[ends]
    rows = ends + 1
    control = rows - treated
    treated_sum = np.cumsum(outcome * treatment)[ends]
    control_sum = np.cumsum(outcome * (1 - treatment))[ends]
    ratio = np.divide(
        treated, control, out=np.zeros_like(treated, dtype=float),
        where=control != 0
    )
    curve = (np.r_[0, rows], np.r_[0, treated_sum - control_sum * ratio])

    return curve


def reference_coefficient(
    scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray
) -> float:
    """
    scikit-uplift の qini_auc_score を写した係数の関数

    Parameters
    ----------
    scores: np.ndarray
        見積もり
    treatment: np.ndarray
        処置
    outcome: np.ndarray
        結果

    Returns
    ----------
    value: float
        理想の並びとの面積の比
    """
    # 見積もりの並び, 理想の並び, 無作為の直線の面積
    x, y = reference_curve(scores, treatment, outcome)
    perfect_scores = outcome * treatment - outcome * (1 - treatment)
    px, py = reference_curve(perfect_scores, treatment, outcome)
    baseline = px[-1] * py[-1] / 2
    actual = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))
    best = float(np.sum((px[1:] - px[:-1]) * (py[1:] + py[:-1]) / 2))
    value = (actual - baseline) / (best - baseline)

    return value


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
    # 曲線の点, 割合, 無作為の直線
    scores, treatment, outcome, points, expected, at_k = SAMPLES[name]
    curve = qini_curve(scores, treatment, outcome)
    assert curve["rows"] == [rows for rows, _ in points]
    assert curve["qini"] == pytest.approx([v for _, v in points], abs=1e-12)
    total = len(scores)
    assert curve["fraction"] == pytest.approx(
        [rows / total for rows, _ in points]
    )
    assert curve["random"] == pytest.approx(
        [rows / total * points[-1][1] for rows, _ in points], abs=1e-12
    )

    # 係数と K ごとの効果の差
    assert_same(qini_coefficient(scores, treatment, outcome), expected)
    for k, expected_k in at_k.items():
        assert_same(uplift_at_k(scores, treatment, outcome, k), expected_k)


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
    scores, treatment, outcome, points, expected, at_k = SAMPLES[name]
    generator = torch.Generator().manual_seed(0)
    orders = [
        list(range(len(scores)))[::-1],
        torch.randperm(len(scores), generator=generator).tolist()
    ]
    for order in orders:
        moved = [[column[i] for i in order]
                 for column in (scores, treatment, outcome)]
        curve = qini_curve(*moved)
        assert curve["qini"] == pytest.approx(
            [v for _, v in points], abs=1e-12
        )
        assert_same(qini_coefficient(*moved), expected)
        for k, expected_k in at_k.items():
            assert_same(uplift_at_k(*moved, k), expected_k)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("amount", [False, True])
def test_matches_reference(seed: int, amount: bool) -> None:
    """
    同点を多く含む乱数の見本で, 曲線と係数が scikit-uplift を写した
    並べ替えの計算と同じ値になることを確かめる関数

    Parameters
    ----------
    seed: int
        乱数の種
    amount: bool
        結果を額にするか

    Returns
    ----------
    None
    """
    # 少ない種類の見積もりで同点を作った見本
    generator = np.random.default_rng(seed)
    scores = generator.integers(0, 8, size=60) / 7
    treatment = generator.integers(0, 2, size=60)
    treatment[:2] = [0, 1]
    outcome = generator.integers(0, 2, size=60).astype(float)
    if amount:
        outcome = outcome * generator.uniform(-50, 500, size=60)

    # 曲線と係数
    rows, values = reference_curve(scores, treatment, outcome)
    curve = qini_curve(scores, treatment, outcome)
    assert curve["rows"] == rows.tolist()
    assert curve["qini"] == pytest.approx(values.tolist(), abs=1e-9)
    assert qini_coefficient(scores, treatment, outcome) == pytest.approx(
        reference_coefficient(scores, treatment, outcome), abs=1e-12
    )


@pytest.mark.parametrize("k, rows", [(0.29, 29), (0.3, 30), (0.57, 57)])
def test_uplift_at_k_without_ties(k: float, rows: int) -> None:
    """
    同点の無い見積もりで, 上位 K 割の効果の差が K × N の切り捨ての行の
    平均の差になることを確かめる関数

    Parameters
    ----------
    k: float
        上位から数える行の割合
    rows: int
        10 進の表記どおりに切り捨てた行の数

    Returns
    ----------
    None
    """
    # 0.29 × 100 のように 2 進では切り捨てが 1 つ減る割合も含む見本
    generator = np.random.default_rng(1)
    scores = generator.permutation(100) / 100
    treatment = np.tile([0, 1], 50)
    outcome = generator.uniform(0, 10, size=100)

    # 並べ替えて上位の行だけで数えた平均の差
    top = np.argsort(-scores)[:rows]
    treated_mean = outcome[top][treatment[top] == 1].mean()
    control_mean = outcome[top][treatment[top] == 0].mean()
    expected = treated_mean - control_mean
    assert uplift_at_k(scores, treatment, outcome, k) == pytest.approx(
        expected, abs=1e-12
    )


def repeated_uplift(
    scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray,
    rows: int
) -> float:
    """
    境の同点のまとまりの重みを, 行を整数の回数だけ繰り返して数える関数

    Parameters
    ----------
    scores: np.ndarray
        見積もり
    treatment: np.ndarray
        処置
    outcome: np.ndarray
        結果
    rows: int
        上位から数える行の数

    Returns
    ----------
    value: float
        処置の群と対照の群の平均の差. 片方の群が無い場合は nan
    """
    # 見積もりの高い順のまとまりごとの, 境の手前の行の数と行の数
    repeats = []
    before = 0
    for value in sorted(set(scores.tolist()), reverse=True):
        members = np.flatnonzero(scores == value)
        taken = min(max(rows - before, 0), members.size)
        repeats.append((members, taken))
        before += members.size

    # 重み taken / まとまりの行の数を, 全ての重みの分母の積で整数の回数に
    scale = math.prod(members.size for members, _ in repeats)
    times = np.zeros(scores.size, dtype=np.int64)
    for members, taken in repeats:
        times[members] = taken * scale // members.size
    treated = np.repeat(outcome[treatment == 1], times[treatment == 1])
    control = np.repeat(outcome[treatment == 0], times[treatment == 0])
    if treated.size == 0 or control.size == 0:
        return math.nan
    value = float(treated.mean() - control.mean())

    return value


@pytest.mark.parametrize("seed", range(10))
def test_tie_weights_match_repeated_rows(seed: int) -> None:
    """
    同点を多く含む乱数の見本で, 上位 K 割の効果の差の境の同点の重みが,
    行を繰り返して数えた平均の差と同じになることを確かめる関数

    Parameters
    ----------
    seed: int
        乱数の種

    Returns
    ----------
    None
    """
    # 少ない種類の見積もりで同点を作った見本
    generator = np.random.default_rng(seed)
    n = int(generator.integers(5, 30))
    scores = generator.integers(0, 5, size=n) / 4
    treatment = generator.integers(0, 2, size=n)
    treatment[:2] = [0, 1]
    outcome = generator.uniform(-10, 100, size=n)

    # 行の数が 1 から n までの K
    for rows in range(1, n + 1):
        k = Fraction(rows, n)
        expected = repeated_uplift(scores, treatment, outcome, rows)
        assert_same(uplift_at_k(scores, treatment, outcome, k), expected)


@pytest.mark.parametrize("k", [
    0.29, np.float32(0.29), np.float64(0.29), Decimal("0.29"),
    Fraction(29, 100)
])
def test_k_is_read_as_decimal(k: Any) -> None:
    """
    float32 と Decimal の K も 10 進の表記どおりに数えることを確かめる関数

    Parameters
    ----------
    k: Any
        上位から数える行の割合

    Returns
    ----------
    None
    """
    # 100 行の 0.29 は 29 行 (float32 の 0.29 は 2 進では 0.29 より小さい)
    scores = np.arange(100, 0, -1, dtype=float)
    treatment = np.zeros(100, dtype=int)
    treatment[:28:2] = 1
    treatment[28] = 1
    outcome = np.zeros(100)
    outcome[28] = 29.0
    treated_mean = 29.0 / 15
    assert uplift_at_k(scores, treatment, outcome, k) == pytest.approx(
        treated_mean
    )


def test_thinned_curve() -> None:
    """
    points で間引いた曲線が, 先頭と最後と等しい間隔に初めて届く点だけを
    間引く前と同じ値で持つことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 同点の無い 1000 行の見本
    generator = np.random.default_rng(2)
    scores = generator.permutation(1000).astype(float)
    treatment = generator.integers(0, 2, size=1000)
    treatment[:2] = [0, 1]
    outcome = generator.integers(0, 2, size=1000)
    full = qini_curve(scores, treatment, outcome)
    thin = qini_curve(scores, treatment, outcome, points=11)

    # 0, 100, ..., 1000 行の 11 点と, 間引く前と同じ値
    assert thin["rows"] == list(range(0, 1001, 100))
    for column in ["fraction", "qini", "random"]:
        assert thin[column] == [full[column][i] for i in thin["rows"]]

    # 同点のまとまりの終わりの点から選び, 重なりを除いた点
    curve = qini_curve(
        [3, 3, 3, 2, 1, 1, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 1, 0, 1, 1, 0, 0], points=4
    )
    assert curve["rows"] == [0, 4, 8, 10]

    # 点が少ない曲線はそのまま
    small = qini_curve([0.2, 0.1], [1, 0], [1, 0], points=5)
    assert small["rows"] == [0, 1, 2]


@pytest.mark.parametrize("convert", [
    lambda s, t, y: (np.array(s), np.array(t, dtype=bool), np.array(y)),
    lambda s, t, y: (np.array(s, dtype=np.float32), np.array(t), y),
    lambda s, t, y: (
        torch.tensor(s, requires_grad=True), torch.tensor(t, dtype=torch.bool),
        torch.tensor(y, dtype=torch.int64)
    ),
    lambda s, t, y: (torch.tensor(s, dtype=torch.float16), t, y),
    lambda s, t, y: (s, [float(v) for v in t], [float(v) for v in y])
])
def test_input_types(convert: Any) -> None:
    """
    テンソル, NumPy の配列, 数の並び, 真偽値の処置で同じ値になることを
    確かめる関数

    Parameters
    ----------
    convert: Any
        見本の見積もり, 処置, 結果を入力の型に写す関数

    Returns
    ----------
    None
    """
    # 半精度でも大小の変わらない見本の見積もりでの値
    scores, treatment, outcome, _, expected, _ = SAMPLES["ties"]
    moved = convert(scores, treatment, outcome)
    assert qini_coefficient(*moved) == pytest.approx(expected, abs=1e-12)
    assert uplift_at_k(*moved, 0.25) == pytest.approx(-1 / 3, abs=1e-12)
    assert qini_curve(*moved)["rows"] == [0, 1, 3, 4, 6, 7, 8]


def test_infinite_scores_are_ties() -> None:
    """
    無限大の見積もりが使え, 同じ符号の無限大どうしが同点になることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 見本の 0.9 を無限大, 0.8 の同点を正の無限大の同点にした値
    inf = float("inf")
    scores = [inf, inf, inf, 0.6, 0.5, 0.5, 0.3, -inf]
    treatment = [1, 1, 0, 1, 0, 1, 0, 0]
    outcome = [1, 0, 1, 1, 0, 0, 0, 1]
    assert qini_curve(scores, treatment, outcome)["rows"] == [
        0, 3, 4, 6, 7, 8
    ]
    assert uplift_at_k(scores, treatment, outcome, 0.375) == (
        pytest.approx(1 / 2 - 1)
    )


@pytest.mark.parametrize("scores, treatment, outcome, message", [
    ([0.1, float("nan")], [0, 1], [0, 1], "nan"),
    ([0.1, 0.2], [0, 2], [0, 1], "0 と 1"),
    ([0.1, 0.2], [0, 0.5], [0, 1], "0 と 1"),
    ([0.1, 0.2], [0, float("nan")], [0, 1], "0 と 1"),
    ([0.1, 0.2], [0, 1], [0, float("nan")], "nan か無限大"),
    ([0.1, 0.2], [0, 1], [0, float("inf")], "nan か無限大"),
    ([0.1, 0.2], [1, 1], [0, 1], "両方の行"),
    ([0.1, 0.2], [0, 0], [0, 1], "両方の行"),
    ([0.1, 0.2], [0, 1, 1], [0, 1], "長さ"),
    ([0.1, 0.2], [0, 1], [0], "長さ"),
    ([[0.1, 0.2]], [[0, 1]], [[0, 1]], "1 次元"),
    ([], [], [], "空")
])
def test_invalid_inputs(
    scores: Any, treatment: Any, outcome: Any, message: str
) -> None:
    """
    nan の見積もり, 0 と 1 でない処置, 有限でない結果, 片方の群の欠け,
    形の誤り, 空の入力が誤りになることを確かめる関数

    Parameters
    ----------
    scores: Any
        見積もり
    treatment: Any
        処置
    outcome: Any
        結果
    message: str
        誤りのメッセージに含まれる文字

    Returns
    ----------
    None
    """
    # 3 つの関数とも同じ誤り
    with pytest.raises(ValueError, match=message):
        qini_curve(scores, treatment, outcome)
    with pytest.raises(ValueError, match=message):
        qini_coefficient(scores, treatment, outcome)
    with pytest.raises(ValueError, match=message):
        uplift_at_k(scores, treatment, outcome, 0.5)


@pytest.mark.parametrize(
    "k", [0, -0.1, 1.5, float("nan"), float("inf"), True, "0.3", None]
)
def test_invalid_k(k: Any) -> None:
    """
    0 より大きく 1 以下の割合でない K が誤りになることを確かめる関数

    Parameters
    ----------
    k: Any
        上位から数える行の割合

    Returns
    ----------
    None
    """
    # 0 以下, 1 より大きい値, nan, 真偽値, 文字, None
    with pytest.raises(ValueError, match="k は 0 より大きく 1 以下"):
        uplift_at_k([0.2, 0.1], [1, 0], [1, 0], k)


def test_numpy_k() -> None:
    """
    NumPy の数の K が使えることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 見本の K = 0.25 と K = 1
    scores, treatment, outcome, _, _, _ = SAMPLES["ties"]
    assert uplift_at_k(scores, treatment, outcome, np.float64(0.25)) == (
        pytest.approx(-1 / 3)
    )
    assert uplift_at_k(scores, treatment, outcome, np.int64(1)) == (
        pytest.approx(0.0)
    )


@pytest.mark.parametrize("points", [1, 0, 2.5, True, "3"])
def test_invalid_points(points: Any) -> None:
    """
    2 以上の整数でない points が誤りになることを確かめる関数

    Parameters
    ----------
    points: Any
        残す点の数の目安

    Returns
    ----------
    None
    """
    # 1 以下, 小数, 真偽値, 文字
    with pytest.raises(ValueError, match="points は 2 以上の整数"):
        qini_curve([0.2, 0.1], [1, 0], [1, 0], points=points)


def test_public_names() -> None:
    """
    3 つの関数が公開の __all__ にあることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # トップレベルと training の両方
    for name in ["qini_coefficient", "qini_curve", "uplift_at_k"]:
        assert name in ml_found.__all__
        assert name in ml_found.training.__all__


class UpliftTrainer(Trainer):
    def on_epoch_end(self, phase: str) -> dict:
        """
        検証のときだけ, 決まった見本の Qini の係数と上位 K 割の効果の差を
        返す関数

        Parameters
        ----------
        phase: str
            train, train_eval, valid のどれか

        Returns
        ----------
        metrics: dict
            検証の場合は qini と uplift_at_50, それ以外は空の辞書
        """
        # 検証の他は空の辞書
        metrics = {}
        if phase != "valid":
            return metrics

        # 見本の見積もり, 処置, 結果
        scores, treatment, outcome, _, _, _ = SAMPLES["top_treated"]
        metrics = {
            "qini": qini_coefficient(scores, treatment, outcome),
            "uplift_at_50": uplift_at_k(scores, treatment, outcome, 0.5)
        }

        return metrics


def test_used_in_training_and_evaluation(tmp_path: Path) -> None:
    """
    on_epoch_end の監視する指標と, save_evaluation の表と曲線に使えることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # valid/qini を監視する学習
    experiment = make_experiment(
        tmp_path,
        "trainer:\n  epochs: 2\n  monitor: valid/qini\n  mode: max\n"
    )
    trainer = UpliftTrainer(nn.Linear(1, 1), experiment, regression_step)
    history = trainer.fit(make_loader(), make_loader())
    assert history.get("valid/qini") == pytest.approx([12 / 13] * 2)
    assert history.get("valid/uplift_at_50") == [1.0, 1.0]

    # 学習の後の評価の表と曲線 (上位に片方の群が無い nan は null)
    scores, treatment, outcome, _, _, _ = SAMPLES["top_treated"]
    save_evaluation(
        experiment, "uplift",
        {
            "qini": qini_coefficient(scores, treatment, outcome),
            "uplift_at_25": uplift_at_k(scores, treatment, outcome, 0.25)
        },
        "test", better={"qini": "higher", "uplift_at_25": "higher"}
    )
    save_evaluation(
        experiment, "qini_curve", qini_curve(scores, treatment, outcome),
        "test", kind="curve", x="fraction"
    )
    evaluations = {
        item["name"]: item for item in load_evaluations(experiment.run_dir)
    }
    assert set(evaluations) == {"uplift", "qini_curve"}
    assert evaluations["qini_curve"]["kind"] == "curve"
    assert evaluations["qini_curve"]["x"] == "fraction"
    assert evaluations["uplift"]["rows"] == [
        {"qini": pytest.approx(12 / 13), "uplift_at_25": None}
    ]
    assert evaluations["qini_curve"]["columns"] == [
        "rows", "fraction", "qini", "random"
    ]
    assert [row["rows"] for row in evaluations["qini_curve"]["rows"]] == [
        0, 1, 3, 5, 6
    ]
