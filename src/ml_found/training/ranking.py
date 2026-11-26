import math
import operator
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def check_k(k: Any) -> int:
    """
    上位から数える品物の数 K を確かめる関数

    Parameters
    ----------
    k: Any
        上位から数える品物の数

    Returns
    ----------
    checked: int
        1 以上の整数の K
    """
    # 真偽値, 整数でない値, 0 以下の値は誤り
    if isinstance(k, bool):
        raise ValueError(f"k は 1 以上の整数にしてください ({k!r})")
    try:
        checked = operator.index(k)
    except TypeError:
        raise ValueError(f"k は 1 以上の整数にしてください ({k!r})") from None
    if checked < 1:
        raise ValueError(f"k は 1 以上の整数にしてください ({k!r})")

    return checked


def to_items(value: Any, name: str, ordered: bool) -> list:
    """
    1 人の利用者の品物を確かめて Python の値の並びにする関数

    Parameters
    ----------
    value: Any
        リスト, タプル, 1 次元の NumPy の配列かテンソル (順の要らない
        場合は集合も)
    name: str
        誤りのメッセージに出す名前
    ordered: bool
        順位の順の並びが要る場合は True (集合は誤り)

    Returns
    ----------
    items: list
        品物の並び (NumPy の配列とテンソルは Python の数に写す)
    """
    # NumPy の配列とテンソルの 1 次元の確認
    if isinstance(value, (np.ndarray, torch.Tensor)):
        if value.ndim != 1:
            raise ValueError(
                f"{name}は 1 次元にしてください ({tuple(value.shape)})"
            )
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        items = value.tolist()
        return items

    # 文字列と, 順の要る場合の集合は誤り
    allowed = (list, tuple) if ordered else (list, tuple, set, frozenset)
    if isinstance(value, (str, bytes)) or not isinstance(value, allowed):
        kinds = "リストかタプル" if ordered else "集合, リストかタプル"
        raise ValueError(
            f"{name}は{kinds}, 1 次元の NumPy の配列かテンソルにしてください "
            f"({type(value).__name__})"
        )
    items = list(value)

    return items


def user_pairs(top: Any, relevant: Any) -> list[tuple[Any, Any]]:
    """
    利用者ごとの並べた品物と正解の組を作る関数

    Parameters
    ----------
    top: Any
        利用者ごとの並べた品物の並びか, 利用者から並べた品物への辞書
    relevant: Any
        利用者ごとの正解の品物の並びか, 利用者から正解の品物への辞書

    Returns
    ----------
    pairs: list[tuple[Any, Any]]
        数える候補の利用者ごとの (並べた品物, 正解) の組. 辞書の場合は
        並べた品物の辞書の利用者だけで, 正解の無い利用者は空の正解
    """
    # 辞書どうしの, 並べた品物の辞書の利用者での組
    top_map = isinstance(top, Mapping)
    relevant_map = isinstance(relevant, Mapping)
    if top_map and relevant_map:
        pairs = [(items, relevant.get(user, ())) for user, items in
                 top.items()]
        return pairs
    if top_map or relevant_map:
        raise ValueError(
            "並べた品物と正解は, 両方とも辞書か, 両方とも並びにしてください"
        )

    # 並びどうしの, 同じ位置の利用者での組
    for value, name in [(top, "並べた品物"), (relevant, "正解")]:
        if isinstance(value, (str, bytes)) or not isinstance(
            value, (list, tuple)
        ):
            raise ValueError(
                f"{name}は利用者ごとの値のリストかタプル, または辞書に"
                f"してください ({type(value).__name__})"
            )
    if len(top) != len(relevant):
        raise ValueError(
            f"並べた品物と正解の利用者の数が違います ({len(top)} と "
            f"{len(relevant)})"
        )
    pairs = list(zip(top, relevant))

    return pairs


def item_set(items: list) -> set:
    """
    品物の並びを集合にする関数

    Parameters
    ----------
    items: list
        品物の並び

    Returns
    ----------
    unique: set
        品物の集合
    """
    # 集合にできない品物 (リストなど) は誤り
    try:
        unique = set(items)
    except TypeError as error:
        raise ValueError(
            f"品物は番号や名前のような値にしてください ({error})"
        ) from None

    return unique


def check_repeated(ranked: list, k: int) -> None:
    """
    上位 K 件の中に同じ品物が 2 度無いことを確かめる関数

    Parameters
    ----------
    ranked: list
        1 人の利用者の上位 K 件の品物
    k: int
        上位から数える品物の数

    Returns
    ----------
    None
    """
    # 集合にできるかの確認と, 前の順位と同じ品物の誤り
    item_set(ranked)
    earlier = set()
    for item in ranked:
        if item in earlier:
            raise ValueError(
                f"上位 {k} 件の中に同じ品物が 2 度あります ({item!r})"
            )
        earlier.add(item)


def user_hits(
    top: Any, relevant: Any, k: int
) -> list[tuple[list[int], int]]:
    """
    数える利用者ごとに, 上位 K 件の正解の順位と正解の数を求める関数

    Parameters
    ----------
    top: Any
        利用者ごとの並べた品物 (順位の順)
    relevant: Any
        利用者ごとの正解の品物
    k: int
        上位から数える品物の数

    Returns
    ----------
    hits: list[tuple[list[int], int]]
        正解のある利用者ごとの, 上位 K 件の正解の順位 (1 から) と正解の数
    """
    # K の確認と利用者ごとの組
    k = check_k(k)
    pairs = user_pairs(top, relevant)

    # 正解のある利用者ごとの, 上位 K 件の正解の順位と正解の数
    hits = []
    for ranked, wanted in pairs:
        ranked = to_items(ranked, "利用者ごとの並べた品物", True)[:k]
        wanted = item_set(to_items(wanted, "利用者ごとの正解", False))
        check_repeated(ranked, k)
        if not wanted:
            continue
        places = [rank for rank, item in enumerate(ranked, 1)
                  if item in wanted]
        hits.append((places, len(wanted)))

    return hits


def recall_at_k(top: Any, relevant: Any, k: int) -> float:
    """
    推薦の上位 K 件の Recall@K を数える関数

    Parameters
    ----------
    top: Any
        利用者ごとの並べた品物 (順位の順) の並びか, 利用者から並べた品物
        への辞書
    relevant: Any
        利用者ごとの正解の品物の並びか, 利用者から正解の品物への辞書
        (top と同じ形)
    k: int
        上位から数える品物の数

    Returns
    ----------
    value: float
        上位 K 件に入った正解の数を正解の全数で割った値の, 正解のある
        利用者の平均. 数える利用者がいない場合は 0
    """
    # 利用者ごとの値の平均
    hits = user_hits(top, relevant, k)
    if not hits:
        return 0.0
    recalls = [len(places) / total for places, total in hits]
    value = float(sum(recalls) / len(recalls))

    return value


def ndcg_at_k(top: Any, relevant: Any, k: int) -> float:
    """
    推薦の上位 K 件の NDCG@K を数える関数

    Parameters
    ----------
    top: Any
        利用者ごとの並べた品物 (順位の順) の並びか, 利用者から並べた品物
        への辞書
    relevant: Any
        利用者ごとの正解の品物の並びか, 利用者から正解の品物への辞書
        (top と同じ形)
    k: int
        上位から数える品物の数

    Returns
    ----------
    value: float
        上位 K 件の正解の順位 r ごとの 1 / log2(r + 1) の和を, 正解を
        上から min(K, 正解の数) 件並べたときの和で割った値の, 正解の
        ある利用者の平均. 数える利用者がいない場合は 0
    """
    # 利用者ごとの DCG と理想の DCG の比の平均
    k = check_k(k)
    hits = user_hits(top, relevant, k)
    if not hits:
        return 0.0
    gains = []
    for places, total in hits:
        dcg = sum(1.0 / math.log2(rank + 1) for rank in places)
        ideal = sum(1.0 / math.log2(rank + 1)
                    for rank in range(1, min(k, total) + 1))
        gains.append(dcg / ideal)
    value = float(sum(gains) / len(gains))

    return value
