import math
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn, optim

import ml_found
from ml_found import (
    PinballLoss, load_evaluations, order_cost, pinball_loss, save_evaluation
)
from test_price_metrics import assert_values
from test_trainer import make_experiment

NAN = math.nan

# docs/DESIGN.md の分位点の損失の見本 (実際の値, 予測, 水準,
# 水準ごとの平均, 全体の平均)
PINBALL_SAMPLES = {
    "one_level": (
        [10, 4, 7], [[8], [6], [7]], [0.5], [2 / 3], 2 / 3
    ),
    "levels": (
        [10, 4, 0], [[6, 9, 13], [2, 5, 4], [-1, 0, 2]], [0.1, 0.5, 0.9],
        [7 / 30, 1 / 3, 1 / 6], 11 / 45
    )
}

# docs/DESIGN.md の見込みの損の見本 (需要, 発注の数, 品切れの損,
# 売れ残りの損, 比べる発注の数, 期待する値)
ORDER_SAMPLES = {
    "baseline": (
        [10, 5, 8, 0], [12, 3, 8, 2], 3, 1, [7, 9, 8, 0],
        {
            "shortage": [0, 2, 0, 0], "excess": [2, 0, 0, 2],
            "cost": [2, 6, 0, 2], "baseline_cost": [9, 4, 0, 0],
            "total": 10, "baseline_total": 13, "saving": 3,
            "ratio": 10 / 13
        }
    ),
    "no_baseline": (
        [4, 6], [6, 2.5], 2, 0.5, None,
        {
            "shortage": [0, 3.5], "excess": [2, 0], "cost": [1, 7],
            "total": 8, "baseline_total": NAN, "saving": NAN, "ratio": NAN
        }
    ),
    "zero_baseline": (
        [3, 0], [3, 1], 4, 1, [3, 0],
        {
            "shortage": [0, 0], "excess": [0, 1], "cost": [0, 1],
            "baseline_cost": [0, 0], "total": 1, "baseline_total": 0,
            "saving": -1, "ratio": NAN
        }
    )
}


# 分位点の損失の誤りの入力 (実際の値, 予測, 水準)
PINBALL_INVALID = [
    ([], [], [0.5]),
    ([[1.0]], [[1.0]], [0.5]),
    ([1.0], [1.0], [0.5]),
    ([1.0], [[1.0, 2.0]], [0.5]),
    ([1.0, 2.0], [[1.0]], [0.5]),
    ([-1.0], [[1.0]], [0.5]),
    ([NAN], [[1.0]], [0.5]),
    ([1.0], [[NAN]], [0.5]),
    ([1.0], [[math.inf]], [0.5]),
    ([1.0], [[]], []),
    ([1.0], [[1.0]], [0]),
    ([1.0], [[1.0]], [1]),
    ([1.0], [[1.0]], [-0.5]),
    ([1.0], [[1.0]], [NAN]),
    ([1.0], [[1.0]], [True]),
    ([1.0], [[1.0]], ["0.5"]),
    ([1.0], [[1.0, 2.0]], [0.5, 0.5]),
    ([1.0], [[1.0]], 0.5),
    ([1.0], [[1.0]], [[0.5]])
]


@pytest.mark.parametrize("name", sorted(PINBALL_SAMPLES))
def test_pinball_samples(name: str) -> None:
    """
    docs/DESIGN.md の分位点の損失の見本で, 手で数えた値が返ることを
    確かめる関数

    Parameters
    ----------
    name: str
        見本の名前

    Returns
    ----------
    None
    """
    # 水準ごとの平均と全体の平均
    actual, quantiles, levels, losses, mean = PINBALL_SAMPLES[name]
    result = pinball_loss(actual, quantiles, levels)
    assert_values(result["levels"]["level"], levels)
    assert_values(result["levels"]["loss"], losses)
    assert_values([result["mean"]], [mean])


def test_pinball_order_of_levels() -> None:
    """
    水準の並びを変えると, 水準ごとの平均も同じ順に並び, 全体の平均は
    変わらないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 逆の順の水準と予測
    actual, quantiles, levels, losses, mean = PINBALL_SAMPLES["levels"]
    moved = [row[::-1] for row in quantiles]
    result = pinball_loss(actual, moved, levels[::-1])
    assert_values(result["levels"]["level"], levels[::-1])
    assert_values(result["levels"]["loss"], losses[::-1])
    assert_values([result["mean"]], [mean])


def test_pinball_matches_brute_force() -> None:
    """
    乱数の見本で, 水準ごとの平均が行ごとに数えた損の平均と同じになることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 乱数の実際の値と予測
    generator = np.random.default_rng(0)
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    actual = generator.uniform(0, 50, size=40)
    quantiles = generator.uniform(-5, 60, size=(40, len(levels)))
    result = pinball_loss(actual, quantiles, levels)

    # 行ごとの場合分けで数えた損
    expected = []
    for j, q in enumerate(levels):
        diff = actual - quantiles[:, j]
        loss = np.where(diff >= 0, q * diff, (q - 1) * diff)
        expected.append(loss.mean())
    assert result["levels"]["loss"] == pytest.approx(expected)
    assert result["mean"] == pytest.approx(np.mean(expected))


@pytest.mark.parametrize("levels", [
    [np.float32(0.1)], [Decimal("0.1")], np.array([0.1], dtype=np.float32),
    torch.tensor([0.1], dtype=torch.float32)
])
def test_levels_are_read_as_decimal(levels: Any) -> None:
    """
    float32 と Decimal の水準も 10 進の表記どおりに読むことを確かめる関数

    Parameters
    ----------
    levels: Any
        水準 0.1 の並び

    Returns
    ----------
    None
    """
    # 実際の値 1, 予測 0 で, 損は 0.1
    result = pinball_loss([1], [[0]], levels)
    assert result["levels"]["level"] == [0.1]
    assert result["mean"] == 0.1


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
    actual, quantiles, levels, _, mean = PINBALL_SAMPLES["levels"]
    result = pinball_loss(
        torch.tensor(actual, dtype=torch.float32),
        torch.tensor(quantiles, dtype=torch.float32, requires_grad=True),
        torch.tensor(levels, dtype=torch.float64)
    )
    assert result["mean"] == pytest.approx(mean)
    demand, order, shortage, holding, baseline, expected = (
        ORDER_SAMPLES["baseline"]
    )
    result = order_cost(
        torch.tensor(demand, dtype=torch.float32),
        torch.tensor(order, dtype=torch.float32, requires_grad=True),
        np.float32(shortage), Decimal(holding), np.array(baseline)
    )
    assert result["total"] == pytest.approx(expected["total"])
    assert result["ratio"] == pytest.approx(expected["ratio"])


def test_python_lists_are_read_as_float64() -> None:
    """
    Python の数の並びを float32 に丸めずに float64 で読み, float32 の
    上限を超える有限の値も受けることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 0.1 と 0.123456789012345 は float32 では丸められる
    assert order_cost([0.1], [0.0], 1, 1)["total"] == 0.1
    result = pinball_loss([0.123456789012345], [[0.0]], [0.5])
    assert result["mean"] == 0.5 * 0.123456789012345

    # float32 の上限を超える値
    assert order_cost([1e39], [1e39], 1, 1)["total"] == 0.0
    result = pinball_loss([0.0], [[1e39]], [0.5])
    assert result["mean"] == 0.5e39


@pytest.mark.parametrize("name", sorted(ORDER_SAMPLES))
def test_order_samples(name: str) -> None:
    """
    docs/DESIGN.md の見込みの損の見本で, 手で数えた値が返ることを
    確かめる関数

    Parameters
    ----------
    name: str
        見本の名前

    Returns
    ----------
    None
    """
    # 行ごとの値 (比べない場合は baseline_cost の列が無い)
    demand, order, shortage, holding, baseline, expected = (
        ORDER_SAMPLES[name]
    )
    result = order_cost(demand, order, shortage, holding, baseline)
    columns = ["shortage", "excess", "cost"]
    if baseline is not None:
        columns.append("baseline_cost")
    assert list(result["items"]) == columns
    for column in columns:
        assert_values(result["items"][column], expected[column])

    # 和と比べ
    for key in ["total", "baseline_total", "saving", "ratio"]:
        assert_values([result[key]], [expected[key]])


@pytest.mark.parametrize("actual, quantiles, levels", PINBALL_INVALID)
def test_pinball_invalid_inputs(
    actual: list, quantiles: list, levels: Any
) -> None:
    """
    空, 形の違い, 負の実際の値, nan, 無限大, 範囲の外と同じ水準が誤りに
    なることを確かめる関数

    Parameters
    ----------
    actual: list
        実際の値
    quantiles: list
        分位点の予測
    levels: Any
        分位点の水準

    Returns
    ----------
    None
    """
    # 誤りの確認
    with pytest.raises(ValueError):
        pinball_loss(actual, quantiles, levels)


@pytest.mark.parametrize("kwargs", [
    {"demand": []},
    {"demand": [[1.0]]},
    {"demand": [1.0, 2.0]},
    {"demand": [-1.0]},
    {"demand": [NAN]},
    {"order": [-1.0]},
    {"order": [math.inf]},
    {"order": [1.0, 2.0]},
    {"baseline": [-1.0]},
    {"baseline": [NAN]},
    {"baseline": [1.0, 2.0]},
    {"shortage_cost": 0},
    {"shortage_cost": -1},
    {"shortage_cost": NAN},
    {"shortage_cost": True},
    {"holding_cost": 0},
    {"holding_cost": math.inf},
    {"holding_cost": "1"},
    {"holding_cost": [1.0]}
])
def test_order_invalid_inputs(kwargs: dict) -> None:
    """
    需要, 発注の数, 比べる発注の数, 損の誤りを確かめる関数

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
        "demand": [1.0], "order": [2.0], "shortage_cost": 3,
        "holding_cost": 1, "baseline": None
    }
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        order_cost(**inputs)


def test_results_can_be_saved(tmp_path: Path) -> None:
    """
    分位点の損失と見込みの損の値を, そのまま評価の表に書いて読めることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest の一時ディレクトリ

    Returns
    ----------
    None
    """
    # 比べる損の和が 0 の見本 (比は nan)
    demand, order, shortage, holding, baseline, _ = (
        ORDER_SAMPLES["zero_baseline"]
    )
    cost = order_cost(demand, order, shortage, holding, baseline)
    actual, quantiles, levels, _, _ = PINBALL_SAMPLES["levels"]
    pinball = pinball_loss(actual, quantiles, levels)
    experiment = make_experiment(tmp_path)
    metrics = {
        "pinball_loss": pinball["mean"], "order_cost": cost["total"],
        "order_cost_ratio": cost["ratio"]
    }
    save_evaluation(experiment, "demand", metrics, "test")
    save_evaluation(experiment, "pinball_levels", pinball["levels"], "test")
    save_evaluation(experiment, "order_items", cost["items"], "test")

    # 読み直した表 (名前の順)
    evaluations = {
        e["name"]: e for e in load_evaluations(experiment.run_dir)
    }
    assert sorted(evaluations) == ["demand", "order_items", "pinball_levels"]
    assert evaluations["demand"]["rows"] == [{
        "pinball_loss": pytest.approx(11 / 45), "order_cost": 1.0,
        "order_cost_ratio": None
    }]
    assert evaluations["pinball_levels"]["rows"][0] == {
        "level": 0.1, "loss": pytest.approx(7 / 30)
    }
    assert evaluations["order_items"]["rows"][1] == {
        "shortage": 0.0, "excess": 1.0, "cost": 1.0, "baseline_cost": 0.0
    }


def test_public_names() -> None:
    """
    2 つの関数と学習の損失が公開の __all__ にあることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 上と training の両方
    for name in ["pinball_loss", "order_cost", "PinballLoss"]:
        assert name in ml_found.__all__
        assert name in ml_found.training.__all__


def available_devices() -> list[tuple[str, torch.dtype]]:
    """
    この計算機で使える装置と, そこで確かめる予測の型を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    devices: list[tuple[str, torch.dtype]]
        装置の名前と型の組 (float64 の無い mps は float32 だけ)
    """
    # CPU と, あれば GPU
    devices = [("cpu", torch.float64), ("cpu", torch.float32)]
    if torch.cuda.is_available():
        devices += [("cuda", torch.float64), ("cuda", torch.float32)]
    if torch.backends.mps.is_available():
        devices.append(("mps", torch.float32))

    return devices


@pytest.mark.parametrize("name", sorted(PINBALL_SAMPLES))
@pytest.mark.parametrize("device, dtype", available_devices())
def test_train_loss_samples(
    name: str, device: str, dtype: torch.dtype
) -> None:
    """
    学習の損失が見本の pinball_loss の mean と同じ値になり, 予測と同じ
    装置と型のスカラーで勾配が予測に流れることを確かめる関数

    Parameters
    ----------
    name: str
        見本の名前
    device: str
        予測の装置
    dtype: torch.dtype
        予測の型

    Returns
    ----------
    None
    """
    # 見本の予測を勾配の付くテンソルにして数える
    actual, quantiles, levels, _, mean = PINBALL_SAMPLES[name]
    prediction = torch.tensor(
        quantiles, dtype=dtype, device=device, requires_grad=True
    )
    loss = PinballLoss(levels)(
        torch.tensor(actual, dtype=torch.float64), prediction
    )
    assert loss.dim() == 0
    assert loss.device.type == device
    assert loss.dtype == dtype
    tolerance = 1e-12 if dtype == torch.float64 else 1e-6
    assert loss.item() == pytest.approx(mean, abs=tolerance)
    assert loss.item() == pytest.approx(
        pinball_loss(actual, quantiles, levels)["mean"], abs=tolerance
    )

    # 勾配は (q - 1{y < ŷ}) の符号を変えた値を要素の数で割った値
    # (差が 0 の要素は 2 つの傾きの平均 0.5 - q)
    loss.backward()
    target = torch.tensor(actual, dtype=torch.float64)[:, None]
    level = torch.tensor(levels, dtype=torch.float64)[None, :]
    values = torch.tensor(quantiles, dtype=torch.float64)
    slope = torch.where(
        values > target, 1 - level,
        torch.where(values < target, -level, 0.5 - level)
    )
    expected = slope / values.numel()
    assert prediction.grad is not None
    assert torch.allclose(
        prediction.grad.to("cpu").double(), expected, atol=tolerance
    )


def test_train_loss_inputs() -> None:
    """
    実際の値は数の並びと NumPy の配列も受け, 水準は渡した順で読み,
    state_dict に入らないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 実際の値の受け方
    actual, quantiles, levels, _, mean = PINBALL_SAMPLES["levels"]
    criterion = PinballLoss(np.array(levels, dtype=np.float32))
    assert criterion.levels == levels
    assert len(criterion.state_dict()) == 0
    prediction = torch.tensor(quantiles, dtype=torch.float64)
    for values in [actual, np.array(actual), torch.tensor(actual)]:
        assert float(criterion(values, prediction)) == pytest.approx(mean)

    # 水準と予測の列を同じ順で入れ替えても同じ値
    order = [2, 0, 1]
    swapped = PinballLoss([levels[i] for i in order])
    assert float(swapped(actual, prediction[:, order])) == pytest.approx(mean)


def test_train_loss_learns_quantiles() -> None:
    """
    学習の損失で定数の予測を学ぶと, 水準ごとの分位点に近づくことを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 0 から 1 の一様な値の分位点を学ぶ
    generator = torch.Generator().manual_seed(0)
    actual = torch.rand(2000, generator=generator)
    levels = [0.1, 0.5, 0.9]
    criterion = PinballLoss(levels)
    weight = nn.Parameter(torch.full((len(levels),), 0.5))
    optimizer = optim.SGD([weight], lr=0.5)
    for _ in range(2000):
        optimizer.zero_grad()
        prediction = weight[None, :].expand(actual.shape[0], -1)
        criterion(actual, prediction).backward()
        optimizer.step()

    # 学んだ値と標本の分位点
    expected = torch.quantile(actual, torch.tensor(levels))
    assert torch.allclose(weight.detach(), expected, atol=0.02)


@pytest.mark.parametrize("actual, quantiles, levels", PINBALL_INVALID)
def test_train_loss_invalid_inputs(
    actual: list, quantiles: list, levels: Any
) -> None:
    """
    pinball_loss と同じ入力が誤りになることを確かめる関数

    Parameters
    ----------
    actual: list
        実際の値
    quantiles: list
        分位点の予測
    levels: Any
        分位点の水準

    Returns
    ----------
    None
    """
    # 水準は作るときに, 実際の値と予測は数えるときに確かめる
    with pytest.raises(ValueError):
        criterion = PinballLoss(levels)
        criterion(
            torch.tensor(actual), torch.tensor(quantiles, dtype=torch.float64)
        )


@pytest.mark.parametrize("quantiles", [
    [[1.0]], np.array([[1.0]]), torch.tensor([[1]])
])
def test_train_loss_needs_float_tensor(quantiles: Any) -> None:
    """
    予測が浮動小数点数のテンソルでない場合は誤りになることを確かめる関数

    Parameters
    ----------
    quantiles: Any
        分位点の予測

    Returns
    ----------
    None
    """
    # 勾配を流せない予測の確認
    with pytest.raises(ValueError):
        PinballLoss([0.5])([1.0], quantiles)
