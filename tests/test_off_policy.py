import csv
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from ml_found.rl import (
    Trajectory, estimate_policy_value, evaluate_policy, save_trajectory
)
from test_format import documented_keys
from test_rl import ToyEnv


class CoinPolicy:
    def __init__(self, p_right: float, seed: int = 0):
        # 行動 1 を選ぶ確率と乱数を保存
        self.p_right = p_right
        self.rng = np.random.default_rng(seed)

    def act(self, observation: Any) -> dict[str, Any]:
        """
        観測に関わらず, 確率 p_right で行動 1 を選ぶ関数

        Parameters
        ----------
        observation: Any
            環境の観測

        Returns
        ----------
        result: dict[str, Any]
            action, status, reasons と, 選んだ行動の確率 action_prob
        """
        action = int(self.rng.random() < self.p_right)
        prob = self.p_right if action == 1 else 1.0 - self.p_right
        result = {
            "action": action, "status": "ok", "reasons": [],
            "action_prob": prob
        }

        return result


def coin_prob(p_right: float) -> Callable:
    """
    確率 p_right で行動 1 を選ぶ方策の確率の関数を作る関数

    Parameters
    ----------
    p_right: float
        行動 1 を選ぶ確率

    Returns
    ----------
    target_prob: Callable
        (観測, 行動) から確率を返す関数
    """
    def target_prob(observation: Any, action: Any) -> float:
        """
        行動を選ぶ確率を返す関数

        Parameters
        ----------
        observation: Any
            観測
        action: Any
            行動

        Returns
        ----------
        prob: float
            行動 1 は p_right, 行動 0 は 1 - p_right
        """
        prob = p_right if int(action) == 1 else 1.0 - p_right

        return prob

    return target_prob


def record(p_right: float, episodes: int, seed: int = 0) -> Trajectory:
    """
    確率で選ぶ方策で ToyEnv (5 歩, 行動 1 で報酬 1) の軌跡を記録する関数

    Parameters
    ----------
    p_right: float
        記録する方策が行動 1 を選ぶ確率
    episodes: int
        エピソードの数
    seed: int = 0
        方策の乱数の種

    Returns
    ----------
    trajectory: Trajectory
        action_prob の付いた軌跡
    """
    trajectory = Trajectory()
    evaluate_policy(
        CoinPolicy(p_right, seed), ToyEnv(), episodes, seed=0,
        trajectory=trajectory
    )

    return trajectory


def test_records_action_prob_column(tmp_path: Path) -> None:
    """
    方策が action_prob を返すと, 軌跡の表に確率の列が付くことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 表の最後の列が確率で, 説明に列の型と確率があることを書く
    path = save_trajectory(tmp_path, "coin", record(0.7, 3))
    with open(path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    info = json.loads(
        (tmp_path / "trajectories" / "coin.json").read_text(encoding="utf-8")
    )
    assert info["columns"][-1] == "action_prob"
    documented = documented_keys("### trajectories")
    for column in info["columns"]:
        assert re.sub(r"_\d+$", "_<番号>", column) in documented
    assert set(info) == documented_keys("#### trajectories/<名前>.json")
    assert info["types"]["action_prob"] == "number"
    assert info["with_action_prob"] is True
    for row in rows:
        expected = 0.7 if row["action"] == "1" else 1.0 - 0.7
        assert float(row["action_prob"]) == pytest.approx(expected)


def test_no_prob_column_without_prob(tmp_path: Path) -> None:
    """
    確率を返さない記録では列が無く, 見積もりは理由を付けて止まることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 確率の無い軌跡
    trajectory = Trajectory()
    trajectory.start_episode(0, None)
    trajectory.add([0.0], 1, 1.0, True, False)
    trajectory.end_episode()
    assert "action_prob" not in trajectory.columns()
    path = save_trajectory(tmp_path, "greedy", trajectory)
    info = json.loads(
        (tmp_path / "trajectories" / "greedy.json").read_text(
            encoding="utf-8"
        )
    )
    assert info["with_action_prob"] is False

    # 表からもメモリからも, 確率の列が無いことを知らせて止まる
    for source in (path, trajectory):
        with pytest.raises(ValueError, match="action_prob"):
            estimate_policy_value(source, coin_prob(0.5))


def test_same_policy_matches_actual_mean() -> None:
    """
    記録の方策と見積もる方策が同じなら, 見積もりが実際の平均の報酬に
    近いことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 重みは全て 1 で, 記録した報酬の平均と同じ. 本当の値は 5 * 0.7
    result = estimate_policy_value(record(0.7, 400), coin_prob(0.7))
    assert result["episodes"] == 400
    assert result["steps"] == 2000
    for key in ("value", "ordinary", "per_decision"):
        assert result[key] == pytest.approx(result["behavior_value"])
        assert result[key] == pytest.approx(3.5, abs=0.15)
    assert result["effective_sample_size"] == pytest.approx(400)
    assert result["max_weight_share"] == pytest.approx(1 / 400)
    assert result["zero_weight_episodes"] == 0
    assert result["notes"] == []


def test_other_policy_close_to_true_value() -> None:
    """
    記録と違う方策の見積もりが, その方策を動かした本当の値に近いことを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 半々で選ぶ方策の軌跡から, 0.8 で行動 1 を選ぶ方策 (本当の値は 4)
    result = estimate_policy_value(record(0.5, 2000), coin_prob(0.8))
    assert result["behavior_value"] == pytest.approx(2.5, abs=0.15)
    for key in ("value", "ordinary", "per_decision"):
        assert result[key] == pytest.approx(4.0, abs=0.25)
    assert 100 < result["effective_sample_size"] < 2000


def test_reads_saved_table(tmp_path: Path) -> None:
    """
    save_trajectory の表から読んだ見積もりが, メモリの軌跡からの見積もり
    と同じことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 観測は配列 (形 (1,)), 行動は整数で渡る
    seen = []

    def target_prob(observation: Any, action: Any) -> float:
        """
        渡された値を記録して確率を返す関数

        Parameters
        ----------
        observation: Any
            観測
        action: Any
            行動

        Returns
        ----------
        prob: float
            行動 1 は 0.6, 行動 0 は 0.4
        """
        seen.append((observation, action))
        prob = 0.6 if action == 1 else 0.4

        return prob

    trajectory = record(0.5, 50)
    path = save_trajectory(tmp_path, "coin", trajectory)
    from_table = estimate_policy_value(path, target_prob)
    from_memory = estimate_policy_value(trajectory, target_prob)
    assert from_table == from_memory
    observation, action = seen[0]
    assert isinstance(observation, np.ndarray) and observation.shape == (1,)
    assert isinstance(action, int)

    # 説明の無い表も, 列の名前と値から読む
    (tmp_path / "trajectories" / "coin.json").unlink()
    assert estimate_policy_value(str(path), target_prob) == from_table


def test_zero_target_prob_and_low_ess() -> None:
    """
    見積もる方策が選ばない行動をとったエピソードの重みが 0 になり,
    重みのばらつきが大きいことを知らせることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 必ず行動 1 を選ぶ方策. 全て 1 のエピソード (約 1/32) だけが重みを持つ
    result = estimate_policy_value(record(0.5, 640), coin_prob(1.0))
    assert result["value"] == pytest.approx(5.0)
    assert result["zero_weight_episodes"] > 500
    assert result["effective_sample_size"] < 64
    notes = " ".join(result["notes"])
    assert "確率 0" in notes
    assert "有効なサンプルの数" in notes

    # 全ての重みが 0 なら重み付きの見積もりは決まらない
    trajectory = Trajectory()
    trajectory.start_episode(0, None)
    trajectory.add([0.0], 0, 0.0, True, False, action_prob=0.5)
    trajectory.end_episode()
    result = estimate_policy_value(trajectory, coin_prob(1.0))
    assert result["value"] is None
    assert result["ordinary"] == 0.0
    assert result["effective_sample_size"] == 0.0
    assert "決まりません" in " ".join(result["notes"])


def test_discount_and_stopped_rows() -> None:
    """
    割引率と, 守りで止めた判断の行の扱いを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 2 歩の後に止めた判断の行 (行動と報酬と確率が無い)
    trajectory = Trajectory()
    trajectory.start_episode(0, None)
    trajectory.add([0.0], 1, 1.0, False, False, "ok", [], 0.5)
    trajectory.add([1.0], 1, 2.0, False, False, "ok", [], 0.5)
    trajectory.add([2.0], None, None, False, False, "stop", ["範囲"])
    trajectory.end_episode()

    # 重みは (1 / 0.5) ** 2, 割り引いた和は 1 + 0.5 * 2
    result = estimate_policy_value(trajectory, coin_prob(1.0), gamma=0.5)
    assert result["steps"] == 2
    assert result["behavior_value"] == pytest.approx(2.0)
    assert result["value"] == pytest.approx(2.0)
    assert result["ordinary"] == pytest.approx(8.0)
    assert result["per_decision"] == pytest.approx(2.0 * 1.0 + 4.0 * 0.5 * 2)
    assert "止めた判断" in " ".join(result["notes"])


def test_rejects_bad_inputs() -> None:
    """
    範囲の外の確率, 割引率, 確率の関数の戻り値で止まることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 記録する確率は 0 より大きく 1 以下で, 行動の無い歩には書けない
    trajectory = Trajectory()
    trajectory.start_episode(0, None)
    for prob in (0.0, 1.5, True, "0.5", float("nan")):
        with pytest.raises(ValueError, match="action_prob"):
            trajectory.add([0.0], 1, 1.0, True, False, action_prob=prob)
    with pytest.raises(ValueError, match="action_prob"):
        trajectory.add(
            [0.0], None, None, False, False, "stop", [], action_prob=0.5
        )

    # 割引率と, 見積もる方策の確率の範囲
    good = record(0.5, 2)
    for gamma in (-0.1, 1.5, True, float("nan")):
        with pytest.raises(ValueError, match="gamma"):
            estimate_policy_value(good, coin_prob(0.5), gamma=gamma)
    with pytest.raises(ValueError, match="target_prob"):
        estimate_policy_value(good, lambda observation, action: 1.5)
    with pytest.raises(ValueError, match="target_prob"):
        estimate_policy_value(good, lambda observation, action: None)
    with pytest.raises(ValueError, match="trajectory"):
        estimate_policy_value(123, coin_prob(0.5))
