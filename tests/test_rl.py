import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
from torch import nn

from ml_found.rl import (
    POLICY_FORMAT_VERSION, EvaluationLoader, ExperienceBuffer, Exploration,
    ReinforceStep, RolloutLoader, apply_guards, export_policy, load_policy,
    make_env, make_guards, read_policy_spec
)
from ml_found.rl.rollout import behavior_probs
from ml_found.training.trainer import Trainer
from ml_found.utils import requirements
from ml_found.utils.requirements import MissingRequirementError
from test_format import documented_keys
from test_requirements import hide_modules
from test_trainer import make_experiment


class Space:
    def __init__(
        self, n: Optional[int] = None, low: Optional[list] = None,
        high: Optional[list] = None
    ):
        # Discrete なら数, Box なら下限と上限と形を保存
        if n is not None:
            self.n = n
            self.start = 0
        else:
            self.low = np.array(low, dtype=np.float32)
            self.high = np.array(high, dtype=np.float32)
            self.shape = self.low.shape


class ToyEnv:
    def __init__(self, length: int = 5):
        # 1 エピソードの歩数と, 観測と行動の空間を保存
        self.length = length
        self.observation_space = Space(low=[-10.0], high=[10.0])
        self.action_space = Space(n=2)
        self.t = 0
        self.seeds: list[Optional[int]] = []

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        """
        エピソードを始める関数

        Parameters
        ----------
        seed: Optional[int] = None
            乱数の種

        Returns
        ----------
        observation: np.ndarray
            最初の観測
        info: dict
            空の辞書
        """
        # 渡された種を記録して歩数を戻す
        self.seeds.append(seed)
        self.t = 0
        observation = np.array([0.0], dtype=np.float32)

        return observation, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        行動 1 で報酬 1, 行動 0 で安全の違反になる 1 歩を進める関数

        Parameters
        ----------
        action: int
            行動

        Returns
        ----------
        result: tuple[np.ndarray, float, bool, bool, dict]
            観測, 報酬, 終了, 打ち切り, info
        """
        # 歩数を進めて, 最後の歩で成功を知らせる
        self.t += 1
        observation = np.array([float(self.t)], dtype=np.float32)
        info = {"cost": float(action == 0)}
        if self.t >= self.length:
            info["is_success"] = True
        result = (
            observation, float(action), self.t >= self.length, False, info
        )

        return result


def toy_model() -> nn.Module:
    """
    1 次元の観測から 2 つの行動のスコアを出すモデルを作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    model: nn.Module
        線形層
    """
    # 種を固定して作成
    torch.manual_seed(0)
    model = nn.Linear(1, 2)

    return model


def test_buffer_returns_and_batch() -> None:
    """
    割引した報酬の和とバッチの形を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    buffer = ExperienceBuffer(gamma=0.5)
    for reward in (1.0, 0.0, 2.0):
        buffer.add([0.0, 1.0], 1, reward, 0.5, violation=reward == 0.0)

    # 後ろから割引した和と, エピソードの集計
    stats = buffer.end_episode(success=True)
    assert stats == {
        "reward": 3.0, "length": 3, "violations": 1, "success": True
    }
    batch = buffer.to_batch()
    assert batch["returns"].tolist() == [1.5, 1.0, 2.0]
    assert batch["observations"].shape == (3, 2)
    assert batch["episode_success"].tolist() == [1.0]
    assert len(buffer) == 3

    # 途中のエピソードはバッチにできない
    buffer.add([0.0, 1.0], 0, 0.0)
    with pytest.raises(ValueError, match="途中"):
        buffer.to_batch()


def test_exploration_probs() -> None:
    """
    温度と一様に選ぶ割合が, 選ぶ確率に入ることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # epsilon が 1 なら一様, 温度が大きいほど一様に近い
    logits = torch.tensor([2.0, 0.0])
    uniform = behavior_probs(logits, Exploration(epsilon=1.0))
    assert torch.allclose(uniform, torch.tensor([0.5, 0.5]))
    sharp = behavior_probs(logits, Exploration())
    flat = behavior_probs(logits, Exploration(temperature=10.0))
    assert sharp[0] > flat[0] > 0.5
    with pytest.raises(ValueError):
        Exploration(temperature=0.0)
    with pytest.raises(ValueError):
        Exploration(epsilon=1.5)


def test_rollout_loader_epochs() -> None:
    """
    エピソードの数と歩数でエポックの長さを決められることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    env = ToyEnv(length=5)
    model = toy_model()

    # 5 エピソードを 2 つずつまとめると 3 つのバッチ
    loader = RolloutLoader(
        env, model, episodes=5, batch_episodes=2, seed=7,
        success_key="is_success"
    )
    batches = list(loader)
    assert len(loader) == 3
    assert [b["episode_rewards"].numel() for b in batches] == [2, 2, 1]
    assert "episode_success" in batches[0]

    # 種は最初の reset だけ
    assert env.seeds[0] == 7
    assert all(seed is None for seed in env.seeds[1:])

    # 歩数で決める場合は, 達したエピソードの終わりまで
    loader = RolloutLoader(env, model, steps=12, batch_episodes=10)
    batches = list(loader)
    assert sum(b["actions"].numel() for b in batches) == 15
    with pytest.raises(TypeError):
        len(loader)
    with pytest.raises(ValueError, match="どちらか一方"):
        RolloutLoader(env, model, episodes=1, steps=1)


def test_evaluation_loader_seeds() -> None:
    """
    評価のエピソードが毎回同じ種で初期化されることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    env = ToyEnv()
    loader = EvaluationLoader(env, toy_model(), episodes=3, seed=100)

    # 2 回の評価で同じ種の並び
    first = list(loader)
    second = list(loader)
    assert env.seeds == [100, 101, 102, 100, 101, 102]
    assert torch.equal(
        first[0]["episode_rewards"], second[0]["episode_rewards"]
    )


def test_reinforce_learns_toy_env(tmp_path: Path) -> None:
    """
    Trainer で学習すると, 報酬の出る行動を選ぶようになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(
        tmp_path,
        "optimizer:\n  name: Adam\n  params:\n    lr: 0.1\n"
        "trainer:\n  epochs: 8\n  monitor: valid/reward\n  mode: max\n"
    )
    model = toy_model()
    train_loader = RolloutLoader(
        ToyEnv(), model, episodes=8, batch_episodes=4, seed=0,
        exploration=Exploration(epsilon=0.1), success_key="is_success"
    )
    valid_loader = EvaluationLoader(
        ToyEnv(), model, episodes=2, seed=1000, success_key="is_success"
    )
    trainer = Trainer(model, experiment, ReinforceStep(entropy_coef=0.01))
    history = trainer.fit(train_loader, valid_loader)

    # エピソードの指標が記録され, 最後は全て報酬の出る行動
    for name in ("reward", "length", "violations", "success"):
        assert len(history.get(f"train/{name}")) == 8
        assert len(history.get(f"valid/{name}")) == 8
    assert history.get("valid/reward")[-1] == 5.0
    assert history.get("valid/violations")[-1] == 0.0
    assert history.get("valid/length")[-1] == 5.0
    assert (experiment.run_dir / "reward_curve.png").exists()


def test_cartpole_few_epochs(tmp_path: Path) -> None:
    """
    CartPole で数エポック学習でき, 書き出した方策が動くことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    pytest.importorskip("gymnasium")
    experiment = make_experiment(
        tmp_path,
        "optimizer:\n  name: Adam\n  params:\n    lr: 0.01\n"
        "trainer:\n  epochs: 3\n  monitor: valid/reward\n  mode: max\n"
    )
    train_env = make_env("CartPole-v1")
    valid_env = make_env("CartPole-v1")
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 2))
    train_loader = RolloutLoader(
        train_env, model, episodes=6, batch_episodes=3, seed=0
    )
    valid_loader = EvaluationLoader(valid_env, model, episodes=2, seed=500)
    trainer = Trainer(model, experiment, ReinforceStep())
    history = trainer.fit(train_loader, valid_loader)

    # エポックごとの報酬と長さ, 安全の違反の記録
    assert len(history.get("train/reward")) == 3
    assert all(value >= 1.0 for value in history.get("valid/length"))
    assert history.get("valid/violations") == [0.0, 0.0, 0.0]

    # 書き出して読み直した方策の行動は 0 か 1
    directory = export_policy(
        experiment.run_dir / "policy", model, train_env.observation_space,
        train_env.action_space
    )
    copy = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 2))
    policy = load_policy(directory, copy)
    observation, _ = valid_env.reset(seed=0)
    result = policy.act(observation)
    assert result["status"] == "ok"
    assert result["action"] in (0, 1)


def test_make_env_without_gymnasium(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    gymnasium が無い場合に make_env が入れ方を言って止まることを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        gymnasium を隠すのに使う

    Returns
    ----------
    None
    """
    hide_modules(monkeypatch, {"gymnasium"})
    with pytest.raises(MissingRequirementError, match="gymnasium>=1.0"):
        make_env("CartPole-v1")
    assert requirements.module_found("gymnasium") is False


def box_spec() -> dict:
    """
    box の行動の一式の中身を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    spec: dict
        2 次元の観測と 2 次元の行動の policy.json の中身
    """
    # 観測の 0 番目に止める条件, 行動は空間より狭い上限
    obs_space = Space(low=[-np.inf, -1.0], high=[np.inf, 1.0])
    act_space = Space(low=[-2.0, -2.0], high=[2.0, 2.0])
    guards = make_guards(
        obs_space, act_space, action_low=[-1.0, -0.5],
        action_high=[1.0, 0.5], observation_low=[-3.0, None],
        observation_high=[3.0, None], fallback="action"
    )
    spec = {
        "format_version": 1, "head": "deterministic",
        "observation_space": {
            "type": "box", "shape": [2], "low": [None, -1.0],
            "high": [None, 1.0]
        },
        "action_space": {
            "type": "box", "shape": [2], "low": [-2.0, -2.0],
            "high": [2.0, 2.0]
        },
        "guards": guards
    }

    return spec


def test_apply_guards_box() -> None:
    """
    box の行動の clip, 止める条件, 有限でない値での外れたときの動きを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    spec = box_spec()

    # 上限の中はそのまま, 外は clip
    result = apply_guards(spec, [0.0, 0.0], [0.5, 0.1])
    assert result["status"] == "ok"
    assert result["action"].tolist() == pytest.approx([0.5, 0.1])
    result = apply_guards(spec, [0.0, 0.0], [5.0, -5.0])
    assert result["status"] == "clipped"
    assert result["action"].tolist() == [1.0, -0.5]
    assert len(result["reasons"]) == 2

    # 止める条件と有限でない値は, 0 の行動 (決めた行動)
    for observation, output in (
        ([4.0, 0.0], [0.5, 0.1]), ([np.nan, 0.0], [0.5, 0.1]),
        ([0.0, 0.0], [np.inf, 0.0]), ([0.0], [0.5, 0.1])
    ):
        result = apply_guards(spec, observation, output)
        assert result["status"] == "fallback"
        assert result["action"].tolist() == [0.0, 0.0]
        assert result["reasons"]


def test_apply_guards_discrete_and_stop() -> None:
    """
    discrete の選んでよい行動と, 止めるか元の制御に戻す動きを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    obs_space = Space(low=[-1.0], high=[1.0])
    act_space = Space(n=3)
    spec = {
        "format_version": 1, "head": "categorical",
        "observation_space": {
            "type": "box", "shape": [1], "low": [-1.0], "high": [1.0]
        },
        "action_space": {"type": "discrete", "n": 3, "start": 0},
        "guards": make_guards(
            obs_space, act_space, allowed_actions=[0, 1],
            observation_low=[-0.5], observation_high=[0.5],
            fallback="handover"
        )
    }

    # 最もスコアの高い行動が選べない場合は選べる中の最良
    result = apply_guards(spec, [0.0], [0.0, 1.0, 5.0])
    assert result == {
        "action": 1, "status": "clipped", "reasons": result["reasons"]
    }
    assert apply_guards(spec, [0.0], [3.0, 1.0, 0.0])["action"] == 0

    # 止める条件に当たると元の制御に戻す (行動なし)
    result = apply_guards(spec, [0.9], [3.0, 1.0, 0.0])
    assert result["status"] == "handover"
    assert result["action"] is None

    # 止める設定では stop
    spec["guards"]["fallback"] = {"kind": "stop", "action": None}
    assert apply_guards(spec, [0.9], [3.0, 1.0, 0.0])["status"] == "stop"


def test_make_guards_errors() -> None:
    """
    空間と食い違う守りを書き出す前に止めることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    obs_space = Space(low=[-1.0], high=[1.0])
    with pytest.raises(ValueError, match="allowed_actions"):
        make_guards(obs_space, Space(n=2), allowed_actions=[2])
    with pytest.raises(ValueError, match="action_low"):
        make_guards(obs_space, Space(n=2), action_low=[0.0])
    with pytest.raises(ValueError, match="fallback_action"):
        make_guards(obs_space, Space(n=2), fallback="action")
    with pytest.raises(ValueError, match="fallback"):
        make_guards(obs_space, Space(n=2), fallback="ignore")
    with pytest.raises(ValueError, match="上限の中"):
        make_guards(
            obs_space, Space(low=[1.0], high=[2.0]), fallback="action"
        )
    with pytest.raises(ValueError, match="要素の数"):
        make_guards(obs_space, Space(n=2), observation_low=[0.0, 0.0])


def test_bundle_round_trip(tmp_path: Path) -> None:
    """
    一式を書き出して読み直すと, 同じ重みと守りで行動を返すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    obs_space = Space(low=[-np.inf, -1.0], high=[np.inf, 1.0])
    act_space = Space(low=[-2.0, -2.0], high=[2.0, 2.0])
    guards = make_guards(
        obs_space, act_space, action_low=[-1.0, -0.5],
        action_high=[1.0, 0.5], observation_low=[-3.0, None],
        observation_high=[3.0, None], fallback="stop"
    )
    directory = export_policy(
        tmp_path / "policy", model, obs_space, act_space, guards,
        metadata={"note": "テスト"}
    )

    # 版と空間と守り, 制限なしは null
    spec = read_policy_spec(directory)
    assert spec["format_version"] == POLICY_FORMAT_VERSION
    assert spec["head"] == "deterministic"
    assert spec["observation_space"]["low"] == [None, -1.0]
    assert spec["guards"]["fallback"]["kind"] == "stop"
    assert spec["metadata"] == {"note": "テスト"}

    # 別に作ったモデルに読み込んでも同じ出力, 守りも当たる
    torch.manual_seed(1)
    policy = load_policy(directory, nn.Linear(2, 2))
    observation = np.array([0.3, -0.2], dtype=np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(observation)).numpy()
    result = policy.act(observation)
    assert np.allclose(
        result["action"], np.clip(expected, [-1.0, -0.5], [1.0, 0.5])
    )
    stopped = policy.act([5.0, 0.0])
    assert stopped["status"] == "stop"
    assert stopped["action"] is None


def test_bundle_newer_version_stops(tmp_path: Path) -> None:
    """
    新しい版と版の無い一式は, 読み違えないように止めることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    directory = export_policy(
        tmp_path / "policy", nn.Linear(1, 2), Space(low=[-1.0], high=[1.0]),
        Space(n=2)
    )
    path = directory / "policy.json"
    spec = json.loads(path.read_text(encoding="utf-8"))

    # 新しい版
    spec["format_version"] = POLICY_FORMAT_VERSION + 1
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="新しい"):
        load_policy(directory, nn.Linear(1, 2))

    # 版の無いもの
    del spec["format_version"]
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="format_version"):
        read_policy_spec(directory)


def test_policy_format_documented(tmp_path: Path) -> None:
    """
    policy.json の項目と守りの項目が docs/FORMAT.md の表と同じことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    directory = export_policy(
        tmp_path / "policy", nn.Linear(1, 2), Space(low=[-1.0], high=[1.0]),
        Space(n=2)
    )
    spec = read_policy_spec(directory)

    # 一番上の項目と守りの項目
    assert set(spec) == documented_keys("### policy.json")
    assert set(spec["guards"]) == documented_keys("### guards")
    assert set(spec["action_space"]) <= documented_keys("### 空間")
    assert sorted(path.name for path in directory.iterdir()) == [
        "policy.json", "policy.pt"
    ]
