import operator
from typing import Any

import torch


def to_vectors(
    scores: Any, labels: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    異常の度合いと正解を確かめて CPU の float64 のベクトルにする関数

    Parameters
    ----------
    scores: Any
        行ごとの異常の度合い (大きいほど異常) の 1 次元の並び
    labels: Any
        行ごとの正解 (異常なら 1 か True) の 1 次元の並び

    Returns
    ----------
    vectors: tuple[torch.Tensor, torch.Tensor]
        度合いと, 0 か 1 の正解のベクトル
    """
    # CPU の float64 のテンソルへの変換
    scores = torch.as_tensor(scores).detach().to("cpu", torch.float64)
    labels = torch.as_tensor(labels).detach().to("cpu", torch.float64)

    # 形の確認
    if scores.dim() != 1 or labels.dim() != 1:
        raise ValueError(
            f"度合いと正解は 1 次元にしてください ({tuple(scores.shape)} と "
            f"{tuple(labels.shape)})"
        )
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(
            f"度合いと正解の長さが違います ({scores.shape[0]} と "
            f"{labels.shape[0]})"
        )
    if scores.shape[0] == 0:
        raise ValueError("度合いと正解が空です")

    # 値の確認
    if bool(torch.isnan(scores).any()):
        raise ValueError(
            "度合いに nan があります. 欠けた行は渡す前に除いてください"
        )
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("正解は真偽値か, 0 と 1 だけにしてください")
    vectors = (scores, labels)

    return vectors


def tie_groups(
    scores: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    度合いの高い順に, 同点の行をまとめた行の数と異常の数を求める関数

    Parameters
    ----------
    scores: torch.Tensor
        行ごとの異常の度合い
    labels: torch.Tensor
        行ごとの 0 か 1 の正解

    Returns
    ----------
    groups: tuple[torch.Tensor, torch.Tensor]
        同点のまとまりごとの行の数と異常の数 (度合いの高い順)
    """
    # 度合いの高い順の同じ度合いのまとまりの番号
    values, inverse = torch.unique(scores, return_inverse=True)
    order = values.shape[0] - 1 - inverse

    # まとまりごとの行の数と異常の数
    sizes = torch.zeros(values.shape[0], dtype=torch.float64)
    sizes.index_add_(0, order, torch.ones_like(labels))
    positives = torch.zeros(values.shape[0], dtype=torch.float64)
    positives.index_add_(0, order, labels)
    groups = (sizes, positives)

    return groups


def average_precision(scores: Any, labels: Any) -> float:
    """
    異常の検知の PR-AUC (平均の適合率) を数える関数

    Parameters
    ----------
    scores: Any
        行ごとの異常の度合い (大きいほど異常) の 1 次元の並び
    labels: Any
        行ごとの正解 (異常なら 1 か True) の 1 次元の並び

    Returns
    ----------
    value: float
        同点の行を 1 つのしきい値にまとめ, 補間せずに数えた平均の適合率.
        正解の異常が無い場合は 0
    """
    # 入力の確認と同点のまとまり
    scores, labels = to_vectors(scores, labels)
    sizes, positives = tie_groups(scores, labels)
    total = float(positives.sum())
    if total == 0:
        return 0.0

    # しきい値ごとの適合率と, 再現率の増え分の積の和
    precision = positives.cumsum(0) / sizes.cumsum(0)
    value = float((positives / total * precision).sum())

    return value


def precision_at_k(scores: Any, labels: Any, k: int) -> float:
    """
    異常の検知の上位 K 件の適合率を数える関数

    Parameters
    ----------
    scores: Any
        行ごとの異常の度合い (大きいほど異常) の 1 次元の並び
    labels: Any
        行ごとの正解 (異常なら 1 か True) の 1 次元の並び
    k: int
        上位から数える行の数 (行の数より多い場合は行の数)

    Returns
    ----------
    value: float
        上位 K 件のうちの異常の割合. K 件目の境の同点の行は,
        残りの枠の数だけを同点の行の中の異常の割合で数える
    """
    # K の確認
    if isinstance(k, bool):
        raise ValueError(f"k は 1 以上の整数にしてください ({k!r})")
    try:
        k = operator.index(k)
    except TypeError:
        raise ValueError(f"k は 1 以上の整数にしてください ({k!r})") from None
    if k < 1:
        raise ValueError(f"k は 1 以上の整数にしてください ({k!r})")

    # 入力の確認と同点のまとまり
    scores, labels = to_vectors(scores, labels)
    sizes, positives = tie_groups(scores, labels)
    k = min(k, scores.shape[0])

    # 境より上のまとまりの全ての行と, 境のまとまりの残りの枠の分の行の異常の数
    before = sizes.cumsum(0) - sizes
    taken = (k - before).clamp(min=0.0, max=None)
    taken = torch.minimum(taken, sizes)
    hits = (taken * positives / sizes).sum()
    value = float(hits) / k

    return value
