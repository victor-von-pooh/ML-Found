import math
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import ml_found
from ml_found import (
    load_evaluations, price_elasticity, profit_lift, save_evaluation
)
from test_trainer import make_experiment

NAN = math.nan

# docs/DESIGN.md の弾力性の見本 (今の予測, 1 % 上げた予測, 弾力性,
# 平均, 数えた商品の数)
ELASTICITY_SAMPLES = {
    "zeros": (
        [100, 50, 0, 20, 0], [98, 50.5, 3, 0, 0], [-2, 1, NAN, -100, NAN],
        -101 / 3, 3
    ),
    "all_zero": ([0, 0], [1, 0], [NAN, NAN], NAN, 0)
}

# docs/DESIGN.md の利益の比べの見本 (値段, 原価, 候補, 上限, 予測した数,
# 期待する値)
PROFIT_SAMPLES = {
    "limit_and_tie": (
        [10, 20, 5], [6, 12, 5], [-0.1, 0, 0.1, 0.2], 0.1,
        [[120, 100, 85, 80], [60, 50, 40, 10], [30, 20, 10, 0]],
        {
            "change": [0.1, 0.0, 0.1], "price": [11, 20, 5.5],
            "profit": [425, 400, 5], "current_profit": [400, 400, 0],
            "total": 830, "current_total": 800, "gain": 30,
            "ratio": 83 / 80, "changed": 2
        }
    ),
    "no_cost": (
        [10, 8, 5], None, [-0.2, 0, 0.2], None,
        [[130, 100, 70], [10, 10, 9], [20, 18, 10]],
        {
            "change": [-0.2, 0.2, 0.0], "price": [8, 9.6, 5],
            "profit": [1040, 86.4, 90], "current_profit": [1000, 80, 90],
            "total": 1216.4, "current_total": 1170, "gain": 46.4,
            "ratio": 3041 / 2925, "changed": 2
        }
    ),
    "non_positive": (
        [10, 10], [12, 10], [0, 0.5], None, [[5, 2], [0, 0]],
        {
            "change": [0.5, 0.0], "price": [15, 10], "profit": [6, 0],
            "current_profit": [-10, 0], "total": 6, "current_total": -10,
            "gain": 16, "ratio": NAN, "changed": 1
        }
    )
}


def assert_values(values: list, expected: list) -> None:
    """
    nan どうしも等しいとして, 並びが見本と同じことを確かめる関数

    Parameters
    ----------
    values: list
        数えた値の並び
    expected: list
        見本の値の並び

    Returns
    ----------
    None
    """
    # 長さと, 値ごとの nan と数
    assert len(values) == len(expected)
    for value, target in zip(values, expected):
        assert isinstance(value, float)
        if math.isnan(target):
            assert math.isnan(value)
        else:
            assert value == pytest.approx(target, abs=1e-12)


@pytest.mark.parametrize("name", sorted(ELASTICITY_SAMPLES))
def test_elasticity_samples(name: str) -> None:
    """
    docs/DESIGN.md の弾力性の見本で, 手で数えた値が返ることを確かめる関数

    Parameters
    ----------
    name: str
        見本の名前

    Returns
    ----------
    None
    """
    # 商品ごとの弾力性, 平均, 数えた商品の数
    now, up, elasticity, mean, counted = ELASTICITY_SAMPLES[name]
    result = price_elasticity(now, up)
    assert_values(result["items"]["elasticity"], elasticity)
    assert_values([result["mean"]], [mean])
    assert result["counted"] == counted


@pytest.mark.parametrize("name", sorted(PROFIT_SAMPLES))
def test_profit_samples(name: str) -> None:
    """
    docs/DESIGN.md の利益の比べの見本で, 手で数えた値が返ることを
    確かめる関数

    Parameters
    ----------
    name: str
        見本の名前

    Returns
    ----------
    None
    """
    # 商品ごとの値
    price, cost, candidates, limit, units, expected = PROFIT_SAMPLES[name]
    result = profit_lift(price, candidates, units, cost, limit)
    for column in ["change", "price", "profit", "current_profit"]:
        assert_values(result["items"][column], expected[column])

    # 和と比べ
    for key in ["total", "current_total", "gain", "ratio"]:
        assert_values([result[key]], [expected[key]])
    assert result["changed"] == expected["changed"]


@pytest.mark.parametrize("name", sorted(PROFIT_SAMPLES))
def test_profit_order_of_candidates(name: str) -> None:
    """
    候補の並びを変えても, 同じ候補を選ぶことを確かめる関数

    Parameters
    ----------
    name: str
        見本の名前

    Returns
    ----------
    None
    """
    # 逆の順の候補と予測した数
    price, cost, candidates, limit, units, expected = PROFIT_SAMPLES[name]
    moved = [row[::-1] for row in units]
    result = profit_lift(price, candidates[::-1], moved, cost, limit)
    assert_values(result["items"]["change"], expected["change"])
    assert_values([result["ratio"]], [expected["ratio"]])


def test_profit_tie_prefers_smaller_change() -> None:
    """
    同じ利益の候補では, 割合の絶対値の小さい候補を選び, 絶対値も同じなら
    割合の小さい候補を選ぶことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 値段 10 で売上 90 が -0.1 (9 × 10) と 0.1 (11 × 90/11) で同じ
    result = profit_lift(
        [10, 10], [-0.1, 0, 0.1], [[10, 8, 90 / 11], [10, 9, 90 / 11]]
    )
    assert result["items"]["change"] == [-0.1, 0.0]
    assert result["changed"] == 1


def test_profit_without_limit_matches_brute_force() -> None:
    """
    乱数の見本で, 選んだ値段が上限の中の候補を全て数えた最も大きい利益と
    同じになり, 上限を超えないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 乱数の値段, 原価, 予測した数
    generator = np.random.default_rng(0)
    candidates = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
    price = generator.uniform(1, 100, size=50)
    cost = price * generator.uniform(0, 1.2, size=50)
    units = generator.uniform(0, 100, size=(50, len(candidates)))

    # 上限ごとの, 全ての候補を数えた最も大きい利益
    for limit in [None, 0, 0.1, 0.25]:
        result = profit_lift(price, candidates, units, cost, limit)
        allowed = [
            i for i, c in enumerate(candidates)
            if limit is None or abs(c) <= limit
        ]
        prices = price[:, None] * (1 + np.array(candidates))
        profits = (prices - cost[:, None]) * units
        best = profits[:, allowed].max(axis=1)
        assert result["items"]["profit"] == pytest.approx(best.tolist())
        assert result["gain"] >= 0
        if limit is not None:
            assert all(
                abs(c) <= limit + 1e-12 for c in result["items"]["change"]
            )


@pytest.mark.parametrize("limit", [
    0.1, np.float32(0.1), Decimal("0.1"), "float32_candidates"
])
def test_limit_is_read_as_decimal(limit: Any) -> None:
    """
    float32 と Decimal の上限も 10 進の表記どおりに比べ, 上限と同じ割合の
    候補を選べることを確かめる関数

    Parameters
    ----------
    limit: Any
        変える幅の上限 (float32_candidates は候補を float32 の配列で渡す場合)

    Returns
    ----------
    None
    """
    # float32 の 0.1 は 2 進では 0.1 より大きい
    candidates = [-0.1, 0, 0.1]
    if isinstance(limit, str):
        candidates = np.array(candidates, dtype=np.float32)
        limit = 0.1
    result = profit_lift(
        [10], candidates, [[0, 1, 100]], max_change=limit
    )
    assert result["items"]["change"] == pytest.approx([0.1])


def test_tensor_inputs() -> None:
    """
    勾配の付いたテンソルも受け, 数の並びと同じ値になることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 見本をテンソルにして数える
    price, cost, candidates, limit, units, expected = (
        PROFIT_SAMPLES["limit_and_tie"]
    )
    result = profit_lift(
        torch.tensor(price, dtype=torch.float32),
        torch.tensor(candidates, dtype=torch.float64),
        torch.tensor(units, dtype=torch.float32, requires_grad=True),
        torch.tensor(cost), limit
    )
    assert result["total"] == pytest.approx(expected["total"])
    elasticity = price_elasticity(
        torch.tensor([100.0, 50.0], requires_grad=True),
        np.array([98.0, 50.5])
    )
    assert elasticity["mean"] == pytest.approx(-0.5)


@pytest.mark.parametrize("now, up", [
    ([], []),
    ([[1.0]], [[1.0]]),
    ([1.0, 2.0], [1.0]),
    ([-1.0], [1.0]),
    ([1.0], [NAN]),
    ([1.0], [math.inf])
])
def test_elasticity_invalid_inputs(now: list, up: list) -> None:
    """
    空, 形の違い, 負の数, nan, 無限大の予測が誤りになることを確かめる関数

    Parameters
    ----------
    now: list
        今の値段での予測した数
    up: list
        1 % 上げた値段での予測した数

    Returns
    ----------
    None
    """
    # 誤りの確認
    with pytest.raises(ValueError):
        price_elasticity(now, up)


@pytest.mark.parametrize("kwargs", [
    {"price": [0]},
    {"price": [-1]},
    {"price": [NAN]},
    {"candidates": []},
    {"candidates": [-0.1, 0.1]},
    {"candidates": [0, 0, 0.1]},
    {"candidates": [-1, 0, 0.1]},
    {"candidates": [0, NAN, 0.1]},
    {"candidates": [False, 0, 0.1]},
    {"candidates": 0.1},
    {"candidates": [[0, 0.1, 0.2]]},
    {"units": [[1, 2]]},
    {"units": [[1, -2, 3]]},
    {"units": [[1, NAN, 3]]},
    {"cost": [1, 2]},
    {"cost": [-1]},
    {"max_change": -0.1},
    {"max_change": NAN},
    {"max_change": True},
    {"max_change": "0.1"}
])
def test_profit_invalid_inputs(kwargs: dict) -> None:
    """
    値段, 候補, 予測した数, 原価, 上限の誤りを確かめる関数

    Parameters
    ----------
    kwargs: dict
        正しい入力から変える引数

    Returns
    ----------
    None
    """
    # 正しい入力の一部を変えて呼ぶ
    inputs = {
        "price": [10], "candidates": [-0.1, 0, 0.1],
        "units": [[1, 2, 3]], "cost": None, "max_change": None
    }
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        profit_lift(**inputs)


def test_results_can_be_saved(tmp_path: Path) -> None:
    """
    利益の比べの商品ごとの値と和を, そのまま評価の表に書いて読めることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest の一時ディレクトリ

    Returns
    ----------
    None
    """
    # 今の利益が 0 以下の見本 (比は nan)
    price, cost, candidates, limit, units, _ = PROFIT_SAMPLES["non_positive"]
    result = profit_lift(price, candidates, units, cost, limit)
    experiment = make_experiment(tmp_path)
    metrics = {"profit_ratio": result["ratio"], "profit_gain": result["gain"]}
    save_evaluation(experiment, "price", metrics, "test")
    save_evaluation(experiment, "price_items", result["items"], "test")

    # 読み直した表
    evaluations = load_evaluations(experiment.run_dir)
    assert [e["name"] for e in evaluations] == ["price", "price_items"]
    assert evaluations[0]["rows"] == [
        {"profit_ratio": None, "profit_gain": 16.0}
    ]
    assert evaluations[1]["rows"][0] == {
        "change": 0.5, "price": 15.0, "profit": 6.0, "current_profit": -10.0
    }


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
    # 上と training の両方
    for name in ["price_elasticity", "profit_lift"]:
        assert name in ml_found.__all__
        assert name in ml_found.training.__all__
