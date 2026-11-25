import math
import numbers
import operator
from fractions import Fraction
from typing import Any, Optional

import torch


def to_uplift_vectors(
    scores: Any, treatment: Any, outcome: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    効果の見積もり, 処置, 結果を確かめて CPU の float64 のベクトルにする関数

    Parameters
    ----------
    scores: Any
        行ごとの効果の見積もり (大きいほど処置する価値がある) の 1 次元の並び
    treatment: Any
        行ごとの処置 (処置の群なら 1 か True, 対照の群なら 0 か False)
    outcome: Any
        行ごとの結果 (0 と 1 か, 額) の 1 次元の並び

    Returns
    ----------
    vectors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        見積もり, 0 か 1 の処置, 結果のベクトル
    """
    # CPU の float64 のテンソルへの変換
    scores = torch.as_tensor(scores).detach().to("cpu", torch.float64)
    treatment = torch.as_tensor(treatment).detach().to("cpu", torch.float64)
    outcome = torch.as_tensor(outcome).detach().to("cpu", torch.float64)

    # 形の確認
    shapes = [tuple(v.shape) for v in (scores, treatment, outcome)]
    if any(len(shape) != 1 for shape in shapes):
        raise ValueError(
            f"見積もり, 処置, 結果は 1 次元にしてください ({shapes[0]}, "
            f"{shapes[1]}, {shapes[2]})"
        )
    if len({shape[0] for shape in shapes}) != 1:
        raise ValueError(
            f"見積もり, 処置, 結果の長さが違います ({shapes[0][0]}, "
            f"{shapes[1][0]}, {shapes[2][0]})"
        )
    if shapes[0][0] == 0:
        raise ValueError("見積もり, 処置, 結果が空です")

    # 値の確認
    if bool(torch.isnan(scores).any()):
        raise ValueError(
            "見積もりに nan があります. 欠けた行は渡す前に除いてください"
        )
    if not bool(((treatment == 0) | (treatment == 1)).all()):
        raise ValueError("処置は真偽値か, 0 と 1 だけにしてください")
    if not bool(torch.isfinite(outcome).all()):
        raise ValueError(
            "結果に nan か無限大があります. 欠けた行は渡す前に除いてください"
        )

    # 両方の群の確認
    treated = int(treatment.sum())
    if treated == 0 or treated == treatment.shape[0]:
        raise ValueError(
            "処置の群と対照の群の両方の行が要ります "
            f"(処置の群 {treated} 行, 対照の群 "
            f"{treatment.shape[0] - treated} 行)"
        )
    vectors = (scores, treatment, outcome)

    return vectors


def group_sums(
    scores: torch.Tensor, treatment: torch.Tensor, outcome: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    見積もりの高い順に, 同点の行をまとめた行の数, 処置の群の行の数,
    処置の群と対照の群の結果の和を求める関数

    Parameters
    ----------
    scores: torch.Tensor
        行ごとの効果の見積もり
    treatment: torch.Tensor
        行ごとの 0 か 1 の処置
    outcome: torch.Tensor
        行ごとの結果

    Returns
    ----------
    sums: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        同点のまとまりごとの行の数, 処置の群の行の数, 処置の群の結果の和,
        対照の群の結果の和 (見積もりの高い順)
    """
    # 見積もりの高い順の同じ見積もりのまとまりの番号
    values, inverse = torch.unique(scores, return_inverse=True)
    order = values.shape[0] - 1 - inverse

    # まとまりごとの和
    columns = [
        torch.ones_like(treatment), treatment, outcome * treatment,
        outcome * (1 - treatment)
    ]
    sums = []
    for column in columns:
        total = torch.zeros(values.shape[0], dtype=torch.float64)
        total.index_add_(0, order, column)
        sums.append(total)
    sums = tuple(sums)

    return sums


def curve_points(
    scores: torch.Tensor, treatment: torch.Tensor, outcome: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    同点のまとまりの終わりごとの行の数と Qini の値を求める関数

    Parameters
    ----------
    scores: torch.Tensor
        行ごとの効果の見積もり
    treatment: torch.Tensor
        行ごとの 0 か 1 の処置
    outcome: torch.Tensor
        行ごとの結果

    Returns
    ----------
    points: tuple[torch.Tensor, torch.Tensor]
        先頭に 0 を足した, 上位の行の数と Qini の値
    """
    # 上位の行の累積の数と和
    sizes, treated, treated_sum, control_sum = group_sums(
        scores, treatment, outcome
    )
    rows = sizes.cumsum(0)
    treated = treated.cumsum(0)
    control = rows - treated
    treated_sum = treated_sum.cumsum(0)
    control_sum = control_sum.cumsum(0)

    # 対照の群の和を処置の群の数に合わせた差 (対照の群が無い間は処置の群の和)
    ratio = torch.where(
        control > 0, treated / control.clamp(min=1.0, max=None),
        torch.zeros_like(control)
    )
    qini = treated_sum - control_sum * ratio

    # 先頭の (0, 0)
    zero = torch.zeros(1, dtype=torch.float64)
    points = (torch.cat([zero, rows]), torch.cat([zero, qini]))

    return points


def trapezoid_area(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    折れ線の下の面積を台形で数える関数

    Parameters
    ----------
    x: torch.Tensor
        横軸の値 (小さい順)
    y: torch.Tensor
        縦軸の値

    Returns
    ----------
    area: float
        折れ線と横軸の間の符号付きの面積
    """
    # 台形の面積の和
    area = float(((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2).sum())

    return area


def thin_points(rows: torch.Tensor, points: Any) -> torch.Tensor:
    """
    曲線の点を, 行の割合が等しい間隔に初めて届く点に間引く番号を求める関数

    Parameters
    ----------
    rows: torch.Tensor
        先頭に 0 を足した, 点ごとの上位の行の数
    points: Any
        残す点の数の目安 (2 以上の整数)

    Returns
    ----------
    index: torch.Tensor
        残す点の番号 (小さい順. 先頭と最後を含む)
    """
    # 点の数の確認
    message = f"points は 2 以上の整数にしてください ({points!r})"
    if isinstance(points, bool):
        raise ValueError(message)
    try:
        points = operator.index(points)
    except TypeError:
        raise ValueError(message) from None
    if points < 2:
        raise ValueError(message)
    if rows.shape[0] <= points:
        return torch.arange(rows.shape[0])

    # rows × (points - 1) が j × 全ての行の数に初めて届く点 (整数で比べる)
    total = int(rows[-1])
    targets = torch.arange(1, points - 1, dtype=torch.float64) * total
    inner = torch.searchsorted(rows * (points - 1), targets)
    ends = torch.tensor([0, rows.shape[0] - 1])
    index = torch.unique(torch.cat([ends, inner]))

    return index


def qini_curve(
    scores: Any, treatment: Any, outcome: Any, points: Optional[int] = None
) -> dict[str, list]:
    """
    効果を見積もるの Qini の曲線を数える関数

    Parameters
    ----------
    scores: Any
        行ごとの効果の見積もり (大きいほど処置する価値がある) の 1 次元の並び
    treatment: Any
        行ごとの処置 (処置の群なら 1 か True, 対照の群なら 0 か False)
    outcome: Any
        行ごとの結果 (0 と 1 か, 額) の 1 次元の並び
    points: Optional[int] = None
        残す点の数の目安, None の場合は全ての点

    Returns
    ----------
    curve: dict[str, list]
        上位の行の数 rows, 行の割合 fraction, Qini の値 qini,
        無作為に処置した場合の直線の値 random の列. 先頭は 0 行で,
        同点のまとまりの終わりごとに 1 点
    """
    # 入力の確認と曲線の点
    scores, treatment, outcome = to_uplift_vectors(scores, treatment, outcome)
    rows, qini = curve_points(scores, treatment, outcome)
    total = float(rows[-1])
    end = qini[-1]

    # 点の間引き
    if points is not None:
        index = thin_points(rows, points)
        rows = rows[index]
        qini = qini[index]

    # 割合と無作為の直線を添えた列
    curve = {
        "rows": [int(v) for v in rows.tolist()],
        "fraction": (rows / total).tolist(),
        "qini": qini.tolist(),
        "random": (rows / total * end).tolist()
    }

    return curve


def qini_coefficient(scores: Any, treatment: Any, outcome: Any) -> float:
    """
    効果を見積もるの Qini の係数を数える関数

    Parameters
    ----------
    scores: Any
        行ごとの効果の見積もり (大きいほど処置する価値がある) の 1 次元の並び
    treatment: Any
        行ごとの処置 (処置の群なら 1 か True, 対照の群なら 0 か False)
    outcome: Any
        行ごとの結果 (0 と 1 か, 額) の 1 次元の並び

    Returns
    ----------
    value: float
        Qini の曲線と無作為の直線の間の面積を, 理想の並びの曲線と無作為の
        直線の間の面積で割った値. 割る面積が 0 の場合は 0
    """
    # 入力の確認と, 見積もりの並びと理想の並びの曲線
    scores, treatment, outcome = to_uplift_vectors(scores, treatment, outcome)
    rows, qini = curve_points(scores, treatment, outcome)
    perfect_scores = outcome * treatment - outcome * (1 - treatment)
    perfect_rows, perfect = curve_points(perfect_scores, treatment, outcome)

    # 無作為の直線との間の面積の比
    baseline = float(rows[-1]) * float(qini[-1]) / 2
    actual = trapezoid_area(rows, qini) - baseline
    best = trapezoid_area(perfect_rows, perfect) - baseline
    if best == 0:
        return 0.0
    value = actual / best

    return value


def top_rows(k: Any, n: int) -> int:
    """
    上位 K 割の行の数を, K の 10 進の表記どおりに数えて切り捨てる関数

    Parameters
    ----------
    k: Any
        上位から数える行の割合 (0 より大きく 1 以下)
    n: int
        行の数

    Returns
    ----------
    rows: int
        K × n の切り捨て
    """
    # K の確認
    message = f"k は 0 より大きく 1 以下の割合にしてください ({k!r})"
    if isinstance(k, bool) or not isinstance(k, numbers.Real):
        raise ValueError(message)
    k = float(k)
    if not 0 < k <= 1:
        raise ValueError(message)

    # 2 進の丸めを入れない切り捨て (0.3 は 3/10)
    rows = math.floor(Fraction(repr(k)) * n)

    return rows


def uplift_at_k(scores: Any, treatment: Any, outcome: Any, k: Any) -> float:
    """
    効果を見積もるの上位 K 割の効果の差を数える関数

    Parameters
    ----------
    scores: Any
        行ごとの効果の見積もり (大きいほど処置する価値がある) の 1 次元の並び
    treatment: Any
        行ごとの処置 (処置の群なら 1 か True, 対照の群なら 0 か False)
    outcome: Any
        行ごとの結果 (0 と 1 か, 額) の 1 次元の並び
    k: Any
        上位から数える行の割合 (0 より大きく 1 以下. 0.3 で上位 3 割)

    Returns
    ----------
    value: float
        上位 K 割の行の処置の群と対照の群の結果の平均の差. 境の同点の行は
        残りの枠の数の割合の重みで数える. 上位に片方の群が無い場合は nan
    """
    # 入力の確認と同点のまとまり
    scores, treatment, outcome = to_uplift_vectors(scores, treatment, outcome)
    rows = top_rows(k, scores.shape[0])
    sizes, treated, treated_sum, control_sum = group_sums(
        scores, treatment, outcome
    )

    # 境より上のまとまりは全て, 境のまとまりは残りの枠の割合の重み
    before = sizes.cumsum(0) - sizes
    taken = (rows - before).clamp(min=0.0, max=None)
    weights = torch.minimum(taken, sizes) / sizes

    # 群ごとの重み付きの平均の差 (上位に片方の群が無い場合は nan)
    treated_count = float((weights * treated).sum())
    control_count = float((weights * (sizes - treated)).sum())
    if treated_count == 0 or control_count == 0:
        return math.nan
    treated_mean = float((weights * treated_sum).sum()) / treated_count
    control_mean = float((weights * control_sum).sum()) / control_count
    value = treated_mean - control_mean

    return value
