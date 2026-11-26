from typing import Any, Optional

import torch
from torch import nn

from ml_found.training.price import (
    check_values, decimal_value, to_float64
)


def to_values(values: Any, name: str, nonnegative: bool) -> torch.Tensor:
    """
    行ごとの値を確かめて CPU の float64 のベクトルにする関数

    Parameters
    ----------
    values: Any
        行ごとの値の 1 次元の並び
    name: str
        エラーのメッセージに出す値の名前
    nonnegative: bool
        0 以上の数だけを受けるか (False の場合は有限の数なら受ける)

    Returns
    ----------
    vector: torch.Tensor
        CPU の float64 のベクトル
    """
    # CPU の float64 のテンソルへの変換と形の確認
    vector = to_float64(values)
    if vector.dim() != 1:
        raise ValueError(
            f"{name}は 1 次元にしてください ({tuple(vector.shape)})"
        )
    if vector.shape[0] == 0:
        raise ValueError(f"{name}が空です")

    # 値の確認
    check_finite(vector, name, nonnegative)

    return vector


def check_finite(values: torch.Tensor, name: str, nonnegative: bool) -> None:
    """
    値が有限の数 (か有限の 0 以上の数) であることを確かめる関数

    Parameters
    ----------
    values: torch.Tensor
        確かめる値
    name: str
        エラーのメッセージに出す値の名前
    nonnegative: bool
        0 以上の数だけを受けるか

    Returns
    ----------
    None
    """
    # 0 以上の数は値段を変える判断と同じ確認
    if nonnegative:
        check_values(values, name, positive=False)
    elif not bool(torch.isfinite(values).all()):
        raise ValueError(
            f"{name}に nan か無限大があります. 欠けた行は渡す前に除いてください"
        )


def to_levels(levels: Any) -> list[float]:
    """
    分位点の水準を確かめて 10 進の表記どおりの数にする関数

    Parameters
    ----------
    levels: Any
        0 より大きく 1 より小さい水準の 1 次元の並び

    Returns
    ----------
    values: list[float]
        渡した順の水準
    """
    # テンソルは NumPy の配列にして, 要素ごとに 10 進の表記で読む
    if isinstance(levels, torch.Tensor):
        if levels.dim() != 1:
            raise ValueError(
                f"分位点の水準は 1 次元にしてください ({tuple(levels.shape)})"
            )
        levels = levels.detach().to("cpu")
        if levels.dtype == torch.bfloat16:
            levels = levels.to(torch.float32)
        levels = levels.numpy()
    try:
        items = list(levels)
    except TypeError:
        raise ValueError(
            f"分位点の水準は数の並びにしてください ({levels!r})"
        ) from None
    fractions = [decimal_value(v, "分位点の水準") for v in items]

    # 空, 範囲, 重なりの確認
    if len(fractions) == 0:
        raise ValueError("分位点の水準が空です")
    if any(not 0 < level < 1 for level in fractions):
        raise ValueError(
            "分位点の水準は 0 より大きく 1 より小さくしてください "
            f"({items!r})"
        )
    if len(set(fractions)) != len(fractions):
        raise ValueError(f"分位点の水準に同じ水準があります ({items!r})")

    return [float(level) for level in fractions]


def to_cost(value: Any, name: str) -> float:
    """
    商品 1 つあたりの損を確かめて数にする関数

    Parameters
    ----------
    value: Any
        損 (有限の正の数)
    name: str
        エラーのメッセージに出す値の名前

    Returns
    ----------
    cost: float
        10 進の表記どおりに読んだ損
    """
    # 数と正の確認
    cost = decimal_value(value, name)
    if cost <= 0:
        raise ValueError(f"{name}は正の数にしてください ({value!r})")

    return float(cost)


def pinball_loss(actual: Any, quantiles: Any, levels: Any) -> dict[str, Any]:
    """
    時間をまたぐ量の判断の分位点の損失を数える関数

    Parameters
    ----------
    actual: Any
        行ごとの実際の値 (0 以上) の 1 次元の並び
    quantiles: Any
        行ごと水準ごとの分位点の予測の (行の数, 水準の数) の並び
    levels: Any
        分位点の水準 (0 より大きく 1 より小さい) の 1 次元の並び

    Returns
    ----------
    result: dict[str, Any]
        levels (列 level, loss の水準ごとの損失の平均), mean (全ての行と
        水準の損失の平均) の辞書
    """
    # 実際の値, 水準, 予測の確認
    actual = to_values(actual, "実際の値", nonnegative=True)
    values = to_levels(levels)
    quantiles = to_float64(quantiles)
    expected = (actual.shape[0], len(values))
    if tuple(quantiles.shape) != expected:
        raise ValueError(
            "分位点の予測は (行の数, 水準の数) の形にしてください "
            f"({tuple(quantiles.shape)}, {expected})"
        )
    check_finite(quantiles, "分位点の予測", nonnegative=False)

    # 水準 q の損 max(q (y - ŷ), (q - 1) (y - ŷ))
    level = torch.tensor(values, dtype=torch.float64)[None, :]
    diff = actual[:, None] - quantiles
    loss = torch.maximum(level * diff, (level - 1) * diff)

    # 水準ごとの平均と全体の平均
    result = {
        "levels": {"level": values, "loss": loss.mean(dim=0).tolist()},
        "mean": float(loss.mean())
    }

    return result


class PinballLoss(nn.Module):
    def __init__(self, levels: Any):
        # 親クラスのコンストラクタを呼び出す
        super(PinballLoss, self).__init__()

        # 水準を保存 (重みではないので state_dict に入れない)
        self.levels = to_levels(levels)

    def forward(self, actual: Any, quantiles: torch.Tensor) -> torch.Tensor:
        """
        分位点の損失を勾配の付くスカラーのテンソルで数える関数

        Parameters
        ----------
        actual: Any
            行ごとの実際の値 (0 以上) の 1 次元の並び
        quantiles: torch.Tensor
            行ごと水準ごとの分位点の予測の (行の数, 水準の数) の浮動小数点数
            のテンソル

        Returns
        ----------
        loss: torch.Tensor
            全ての行と水準の損失の平均 (予測と同じ装置と型のスカラー)
        """
        # 予測と実際の値の確認 (実際の値は予測と同じ装置と型に)
        check_prediction(quantiles)
        actual = to_target(actual, quantiles)
        expected = (actual.shape[0], len(self.levels))
        if tuple(quantiles.shape) != expected:
            raise ValueError(
                "分位点の予測は (行の数, 水準の数) の形にしてください "
                f"({tuple(quantiles.shape)}, {expected})"
            )
        with torch.no_grad():
            check_finite(quantiles, "分位点の予測", nonnegative=False)

        # 水準 q の損 max(q (y - ŷ), (q - 1) (y - ŷ)) の全体の平均
        level = torch.tensor(
            self.levels, dtype=quantiles.dtype, device=quantiles.device
        )[None, :]
        diff = actual[:, None] - quantiles
        loss = torch.maximum(level * diff, (level - 1) * diff).mean()

        return loss

    def extra_repr(self) -> str:
        """
        print で出す水準の文字列を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        text: str
            水準の並び
        """
        # 渡した順の水準
        text = f"levels={self.levels}"

        return text


def check_prediction(quantiles: Any) -> None:
    """
    分位点の予測が浮動小数点数のテンソルであることを確かめる関数

    Parameters
    ----------
    quantiles: Any
        分位点の予測

    Returns
    ----------
    None
    """
    # 勾配を流すので浮動小数点数のテンソルだけを受ける
    if not isinstance(quantiles, torch.Tensor):
        raise ValueError(
            "分位点の予測はテンソルにしてください "
            f"({type(quantiles).__name__})"
        )
    if not quantiles.is_floating_point():
        raise ValueError(
            f"分位点の予測は浮動小数点数にしてください ({quantiles.dtype})"
        )


def to_target(actual: Any, quantiles: torch.Tensor) -> torch.Tensor:
    """
    実際の値を確かめて予測と同じ装置と型のベクトルにする関数

    Parameters
    ----------
    actual: Any
        行ごとの実際の値の 1 次元の並び
    quantiles: torch.Tensor
        分位点の予測

    Returns
    ----------
    vector: torch.Tensor
        予測と同じ装置と型のベクトル
    """
    # 予測と同じ装置と型のテンソルへの変換と形の確認
    if isinstance(actual, torch.Tensor):
        vector = actual.to(quantiles.device, quantiles.dtype)
    else:
        vector = torch.as_tensor(
            actual, dtype=quantiles.dtype, device=quantiles.device
        )
    if vector.dim() != 1:
        raise ValueError(
            f"実際の値は 1 次元にしてください ({tuple(vector.shape)})"
        )
    if vector.shape[0] == 0:
        raise ValueError("実際の値が空です")

    # 値の確認
    with torch.no_grad():
        check_finite(vector, "実際の値", nonnegative=True)

    return vector


def order_cost(
    demand: Any, order: Any, shortage_cost: Any, holding_cost: Any,
    baseline: Any = None
) -> dict[str, Any]:
    """
    時間をまたぐ量の判断の発注の数の見込みの損を数える関数

    Parameters
    ----------
    demand: Any
        行ごとの実際の需要 (0 以上) の 1 次元の並び
    order: Any
        行ごとの売れる数 (0 以上. 在庫の数と発注の数を足した数) の 1 次元の
        並び
    shortage_cost: Any
        1 つ足りないときの損 (正の数)
    holding_cost: Any
        1 つ余ったときの損 (正の数. shortage_cost と同じお金の単位)
    baseline: Any = None
        比べる発注の数での売れる数 (0 以上. order と同じく在庫の数を足した
        数) の 1 次元の並び, None の場合は比べない

    Returns
    ----------
    result: dict[str, Any]
        items (列 shortage, excess, cost と, 比べる場合は baseline_cost の
        行ごとの値), total (損の和), baseline_total (比べる発注の数での
        損の和), saving (baseline_total - total), ratio (total /
        baseline_total. baseline_total が 0 なら nan) の辞書. 比べない
        場合の baseline_total, saving, ratio は nan
    """
    # 需要, 売れる数, 損の確認
    demand = to_values(demand, "実際の需要", nonnegative=True)
    order = to_values(order, "売れる数", nonnegative=True)
    shortage = to_cost(shortage_cost, "品切れの損 shortage_cost")
    holding = to_cost(holding_cost, "売れ残りの損 holding_cost")
    check_length(demand, order, "売れる数")

    # 行ごとの足りない数, 余った数, 損
    short, excess, cost = row_costs(demand, order, shortage, holding)
    total = float(cost.sum())
    result: dict[str, Any] = {
        "items": {
            "shortage": short.tolist(), "excess": excess.tolist(),
            "cost": cost.tolist()
        },
        "total": total,
        "baseline_total": float("nan"),
        "saving": float("nan"),
        "ratio": float("nan")
    }

    # 比べる発注の数での損と, 差と比 (比べる損の和が 0 なら比は nan)
    compared = compare_baseline(demand, baseline, shortage, holding)
    if compared is not None:
        baseline_total = float(compared.sum())
        result["items"]["baseline_cost"] = compared.tolist()
        result["baseline_total"] = baseline_total
        result["saving"] = baseline_total - total
        if baseline_total > 0:
            result["ratio"] = total / baseline_total

    return result


def check_length(
    demand: torch.Tensor, values: torch.Tensor, name: str
) -> None:
    """
    需要と売れる数の長さが同じことを確かめる関数

    Parameters
    ----------
    demand: torch.Tensor
        行ごとの実際の需要
    values: torch.Tensor
        行ごとの売れる数
    name: str
        エラーのメッセージに出す値の名前

    Returns
    ----------
    None
    """
    # 長さの確認
    if demand.shape != values.shape:
        raise ValueError(
            f"実際の需要と{name}の長さが違います "
            f"({demand.shape[0]}, {values.shape[0]})"
        )


def row_costs(
    demand: torch.Tensor, order: torch.Tensor, shortage: float,
    holding: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    行ごとの足りない数, 余った数, 損を数える関数

    Parameters
    ----------
    demand: torch.Tensor
        行ごとの実際の需要
    order: torch.Tensor
        行ごとの売れる数 (在庫の数と発注の数を足した数)
    shortage: float
        1 つ足りないときの損
    holding: float
        1 つ余ったときの損

    Returns
    ----------
    short: torch.Tensor
        足りない数 max(需要 - 売れる数, 0)
    excess: torch.Tensor
        余った数 max(売れる数 - 需要, 0)
    cost: torch.Tensor
        shortage × 足りない数 + holding × 余った数
    """
    # 足りない数と余った数は片方だけが正
    short = (demand - order).clamp(min=0.0, max=None)
    excess = (order - demand).clamp(min=0.0, max=None)
    cost = shortage * short + holding * excess

    return short, excess, cost


def compare_baseline(
    demand: torch.Tensor, baseline: Any, shortage: float, holding: float
) -> Optional[torch.Tensor]:
    """
    比べる発注の数での行ごとの損を数える関数

    Parameters
    ----------
    demand: torch.Tensor
        行ごとの実際の需要
    baseline: Any
        比べる発注の数での売れる数の 1 次元の並び, None の場合は比べない
    shortage: float
        1 つ足りないときの損
    holding: float
        1 つ余ったときの損

    Returns
    ----------
    cost: Optional[torch.Tensor]
        行ごとの損. 比べない場合は None
    """
    # 比べない場合
    if baseline is None:
        return None

    # 確認と行ごとの損
    baseline = to_values(baseline, "比べる売れる数", nonnegative=True)
    check_length(demand, baseline, "比べる売れる数")
    _, _, cost = row_costs(demand, baseline, shortage, holding)

    return cost
