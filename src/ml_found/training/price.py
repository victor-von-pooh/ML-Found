import numbers
from decimal import Decimal
from fractions import Fraction
from typing import Any, Optional

import torch

# 同じ利益とみなす差の割合
PROFIT_TOLERANCE = 1e-9


def to_vector(values: Any, name: str, positive: bool = False) -> torch.Tensor:
    """
    商品ごとの値を確かめて CPU の float64 のベクトルにする関数

    Parameters
    ----------
    values: Any
        商品ごとの値の 1 次元の並び
    name: str
        エラーのメッセージに出す値の名前
    positive: bool = False
        正の数だけを受けるか (False の場合は 0 以上の数)

    Returns
    ----------
    vector: torch.Tensor
        CPU の float64 のベクトル
    """
    # CPU の float64 のテンソルへの変換と形の確認
    vector = torch.as_tensor(values).detach().to("cpu", torch.float64)
    if vector.dim() != 1:
        raise ValueError(
            f"{name}は 1 次元にしてください ({tuple(vector.shape)})"
        )
    if vector.shape[0] == 0:
        raise ValueError(f"{name}が空です")

    # 値の確認
    check_values(vector, name, positive)

    return vector


def check_values(values: torch.Tensor, name: str, positive: bool) -> None:
    """
    値が有限の 0 以上 (か正) の数であることを確かめる関数

    Parameters
    ----------
    values: torch.Tensor
        確かめる値
    name: str
        エラーのメッセージに出す値の名前
    positive: bool
        正の数だけを受けるか (False の場合は 0 以上の数)

    Returns
    ----------
    None
    """
    # 有限の値と符号の確認
    if not bool(torch.isfinite(values).all()):
        raise ValueError(
            f"{name}に nan か無限大があります. 欠けた行は渡す前に除いてください"
        )
    if positive and not bool((values > 0).all()):
        raise ValueError(f"{name}は正の数にしてください")
    if not positive and not bool((values >= 0).all()):
        raise ValueError(f"{name}は 0 以上の数にしてください")


def decimal_value(value: Any, name: str) -> Fraction:
    """
    数を 10 進の表記どおりの分数にする関数

    Parameters
    ----------
    value: Any
        数 (Python と NumPy の数, Decimal, Fraction)
    name: str
        エラーのメッセージに出す値の名前

    Returns
    ----------
    fraction: Fraction
        10 進の表記どおりの分数 (0.1 は 1/10)
    """
    # 真偽値と数でない値, nan と無限大の確認
    message = f"{name}は有限の数にしてください ({value!r})"
    if isinstance(value, bool) or not isinstance(
        value, (numbers.Real, Decimal)
    ):
        raise ValueError(message)
    try:
        fraction = Fraction(str(value))
    except ValueError:
        raise ValueError(message) from None

    return fraction


def to_candidates(candidates: Any) -> list[Fraction]:
    """
    値段の候補の変える割合を確かめて 10 進の表記どおりの分数にする関数

    Parameters
    ----------
    candidates: Any
        今の値段に対する変える割合の 1 次元の並び (0 を含む)

    Returns
    ----------
    ratios: list[Fraction]
        候補の順の割合
    """
    # テンソルはその値のまま, 他は要素ごと
    if isinstance(candidates, torch.Tensor):
        if candidates.dim() != 1:
            raise ValueError(
                "値段の候補は 1 次元にしてください "
                f"({tuple(candidates.shape)})"
            )
        candidates = candidates.detach().to("cpu").tolist()
    try:
        values = list(candidates)
    except TypeError:
        raise ValueError(
            f"値段の候補は割合の並びにしてください ({candidates!r})"
        ) from None
    ratios = [decimal_value(v, "値段の候補の割合") for v in values]

    # 空, 範囲, 重なり, 今の値段の確認
    if len(ratios) == 0:
        raise ValueError("値段の候補が空です")
    if any(ratio <= -1 for ratio in ratios):
        raise ValueError(
            f"値段の候補の割合は -1 より大きくしてください ({values!r})"
        )
    if len(set(ratios)) != len(ratios):
        raise ValueError(f"値段の候補に同じ割合があります ({values!r})")
    if 0 not in ratios:
        raise ValueError(
            f"値段の候補に今の値段 (割合 0) を含めてください ({values!r})"
        )

    return ratios


def to_limit(max_change: Any) -> Optional[Fraction]:
    """
    値段を変える幅の上限を確かめて 10 進の表記どおりの分数にする関数

    Parameters
    ----------
    max_change: Any
        変える割合の絶対値の上限 (0 以上). None の場合は上限無し

    Returns
    ----------
    limit: Optional[Fraction]
        上限の分数. 上限無しの場合は None
    """
    # 上限無しと, 0 以上の有限の数の確認
    if max_change is None:
        return None
    limit = decimal_value(max_change, "変える幅の上限 max_change")
    if limit < 0:
        raise ValueError(
            f"変える幅の上限 max_change は 0 以上にしてください "
            f"({max_change!r})"
        )

    return limit


def price_elasticity(units_now: Any, units_up: Any) -> dict[str, Any]:
    """
    値段を変える判断の値段の弾力性を数える関数

    Parameters
    ----------
    units_now: Any
        商品ごとの今の値段での予測した数 (0 以上) の 1 次元の並び
    units_up: Any
        商品ごとの 1 % 上げた値段での予測した数 (0 以上) の 1 次元の並び

    Returns
    ----------
    result: dict[str, Any]
        items (列 elasticity の商品ごとの弾力性. 今の予測が 0 の商品は
        nan), mean (nan でない弾力性の平均. 無ければ nan),
        counted (平均に数えた商品の数) の辞書
    """
    # 入力の確認
    units_now = to_vector(units_now, "今の値段での予測した数")
    units_up = to_vector(units_up, "1 % 上げた値段での予測した数")
    if units_now.shape != units_up.shape:
        raise ValueError(
            "今の値段と 1 % 上げた値段での予測した数の長さが違います "
            f"({units_now.shape[0]}, {units_up.shape[0]})"
        )

    # 1 % 上げたときの予測した数の変わる割合 (今の予測が 0 の商品は nan)
    counted = units_now > 0
    elasticity = torch.full_like(units_now, float("nan"))
    elasticity[counted] = (
        100 * (units_up[counted] - units_now[counted]) / units_now[counted]
    )

    # 数えた商品の平均
    count = int(counted.sum())
    mean = float(elasticity[counted].mean()) if count > 0 else float("nan")
    result = {
        "items": {"elasticity": elasticity.tolist()},
        "mean": mean,
        "counted": count
    }

    return result


def choose_candidates(
    profits: torch.Tensor, ratios: list[Fraction],
    limit: Optional[Fraction]
) -> torch.Tensor:
    """
    商品ごとに上限の中で見込みの利益が最も大きい候補の番号を選ぶ関数

    Parameters
    ----------
    profits: torch.Tensor
        商品ごと候補ごとの見込みの利益 (商品の数, 候補の数)
    ratios: list[Fraction]
        候補の順の変える割合
    limit: Optional[Fraction]
        変える割合の絶対値の上限. None の場合は上限無し

    Returns
    ----------
    chosen: torch.Tensor
        商品ごとの選んだ候補の番号
    """
    # 上限の中の候補 (割合 0 はいつも入る)
    allowed = torch.tensor(
        [limit is None or abs(ratio) <= limit for ratio in ratios]
    )
    masked = profits.masked_fill(~allowed, float("-inf"))

    # 最も大きい利益と同じとみなす候補
    best = masked.max(dim=1, keepdim=True).values
    tolerance = PROFIT_TOLERANCE * best.abs().clamp(min=1.0, max=None)
    tied = masked >= best - tolerance

    # 同じ利益の中で, 割合の絶対値の小さい順, 割合の小さい順の先頭
    order = sorted(
        range(len(ratios)), key=lambda i: (abs(ratios[i]), ratios[i])
    )
    rank = torch.empty(len(ratios), dtype=torch.float64)
    rank[order] = torch.arange(len(ratios), dtype=torch.float64)
    ranked = torch.where(
        tied, rank.expand_as(masked), torch.full_like(masked, float("inf"))
    )
    chosen = ranked.argmin(dim=1)

    return chosen


def profit_lift(
    price: Any, candidates: Any, units: Any, cost: Any = None,
    max_change: Any = None
) -> dict[str, Any]:
    """
    値段を変える判断の選んだ値段と今の値段の見込みの利益を比べる関数

    Parameters
    ----------
    price: Any
        商品ごとの今の値段 (正の数) の 1 次元の並び
    candidates: Any
        今の値段に対する変える割合の 1 次元の並び (0 を含む. 0.1 で 1 割
        上げる)
    units: Any
        商品ごと候補ごとの予測した数 (0 以上) の (商品の数, 候補の数) の並び
    cost: Any = None
        商品ごとの原価 (0 以上) の 1 次元の並び, None の場合は全て 0
    max_change: Any = None
        変える割合の絶対値の上限 (0 以上), None の場合は上限無し

    Returns
    ----------
    result: dict[str, Any]
        items (列 change, price, profit, current_profit の商品ごとの値),
        total (選んだ値段の見込みの利益の和), current_total (今の値段の
        見込みの利益の和), gain (total - current_total), ratio
        (total / current_total. current_total が 0 以下なら nan),
        changed (割合が 0 でない候補を選んだ商品の数) の辞書
    """
    # 値段, 候補, 上限の確認
    price = to_vector(price, "今の値段", positive=True)
    ratios = to_candidates(candidates)
    limit = to_limit(max_change)

    # 予測した数の確認
    units = torch.as_tensor(units).detach().to("cpu", torch.float64)
    expected = (price.shape[0], len(ratios))
    if tuple(units.shape) != expected:
        raise ValueError(
            "予測した数は (商品の数, 候補の数) の形にしてください "
            f"({tuple(units.shape)}, {expected})"
        )
    check_values(units, "予測した数", positive=False)

    # 原価の確認 (無ければ 0)
    if cost is None:
        cost = torch.zeros_like(price)
    else:
        cost = to_vector(cost, "原価")
        if cost.shape != price.shape:
            raise ValueError(
                "今の値段と原価の長さが違います "
                f"({price.shape[0]}, {cost.shape[0]})"
            )

    # 候補の値段ごとの見込みの利益
    changes = torch.tensor([float(r) for r in ratios], dtype=torch.float64)
    prices = price[:, None] * (1 + changes[None, :])
    profits = (prices - cost[:, None]) * units

    # 商品ごとの選んだ候補と今の値段の利益
    chosen = choose_candidates(profits, ratios, limit)
    rows = torch.arange(price.shape[0])
    current = ratios.index(0)
    chosen_profit = profits[rows, chosen]
    current_profit = profits[:, current]

    # 和と比べ (今の利益の和が 0 以下なら比は nan)
    total = float(chosen_profit.sum())
    current_total = float(current_profit.sum())
    ratio = total / current_total if current_total > 0 else float("nan")
    result = {
        "items": {
            "change": changes[chosen].tolist(),
            "price": prices[rows, chosen].tolist(),
            "profit": chosen_profit.tolist(),
            "current_profit": current_profit.tolist()
        },
        "total": total,
        "current_total": current_total,
        "gain": total - current_total,
        "ratio": ratio,
        "changed": int((chosen != current).sum())
    }

    return result
