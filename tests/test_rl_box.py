from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
from torch import nn

from ml_found.rl import (
    EvaluationLoader, ExperienceBuffer, Exploration, ReinforceStep,
    RolloutLoader, evaluate_policy, export_policy, load_policy, make_guards
)
from ml_found.rl.rollout import LOG_STD_MAX, LOG_STD_MIN, gaussian_std
from ml_found.training.trainer import Trainer
from test_rl import Space
from test_trainer import make_experiment


class BoxEnv:
    def __init__(self, length: int = 5, target: float = 0.5):
        # 1 エピソードの歩数, 目標の行動と, 観測と行動の空間を保存
        self.length = length
        self.target = target
        self.observation_space = Space(low=[-10.0], high=[10.0])
        self.action_space = Space(low=[-1.0], high=[1.0])
        self.t = 0
        self.actions: list[np.ndarray] = []

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
        # 歩数の初期化
        self.t = 0
        observation = np.array([1.0], dtype=np.float32)
        info: dict = {}

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        目標の行動に近いほど報酬が大きい 1 歩を進める関数

        Parameters
        ----------
        action: np.ndarray
            連続の行動

        Returns
        ----------
        result: tuple[np.ndarray, float, bool, bool, dict]
            観測, 報酬, 終了, 打ち切り, info
        """
        # 渡された行動の記録と, 目標との差の 2 乗の符号を反転した報酬
        self.actions.append(np.array(action))
        self.t += 1
        reward = -float((float(action[0]) - self.target) ** 2)
        observation = np.array([1.0], dtype=np.float32)
        result = (observation, reward, self.t >= self.length, False, {})

        return result


class GaussianNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        # 親クラスのコンストラクタを呼び出す
        super(GaussianNet, self).__init__()

        # 観測と行動の次元数を保存
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 平均の線形層と, 観測によらない対数の標準偏差
        self.mean = nn.Linear(self.obs_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            観測

        Returns
        ----------
        mean: torch.Tensor
            行動の平均
        """
        # 線形層で行動の平均
        mean = self.mean(x)

        return mean


def box_model() -> GaussianNet:
    """
    平均が 0 の行動から始まる正規分布の方策を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    model: GaussianNet
        平均が 0, 標準偏差が 1 の方策
    """
    # 重みを 0 にして平均を 0 に
    model = GaussianNet(1, 1)
    with torch.no_grad():
        model.mean.weight.zero_()
        model.mean.bias.zero_()

    return model


def test_box_rollout_batch_and_clip() -> None:
    """
    連続の行動のバッチの形と, 環境には空間の範囲で clip した行動を渡す
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    torch.manual_seed(0)
    env = BoxEnv(length=50)
    model = box_model()
    with torch.no_grad():
        model.log_std.fill_(1.0)
    batch = next(iter(RolloutLoader(env, model, episodes=1)))

    # 行動は (歩数, 要素の数) の float32 で clip する前の値
    assert batch["actions"].dtype == torch.float32
    assert batch["actions"].shape == (50, 1)
    assert batch["actions"].abs().max() > 1.0
    assert batch["behavior_log_probs"].shape == (50,)

    # 環境に渡した行動は空間の形と範囲の中
    passed = np.stack(env.actions)
    assert passed.shape == (50, 1)
    assert passed.dtype == np.float32
    assert np.all(np.abs(passed) <= 1.0)
    expected = np.clip(batch["actions"].numpy(), -1.0, 1.0)
    assert np.allclose(passed, expected)

    # 記録した対数の確率密度は正規分布のもの
    normal = torch.distributions.Normal(0.0, float(np.e))
    log_probs = normal.log_prob(batch["actions"][:, 0])
    assert torch.allclose(log_probs, batch["behavior_log_probs"], atol=1e-5)


def test_box_ratio_is_one_and_eval_uses_mean() -> None:
    """
    更新の前の比が 1 で, 評価は平均の行動で動くことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    torch.manual_seed(0)
    model = box_model()
    batch = next(iter(RolloutLoader(BoxEnv(), model, episodes=2)))
    _, _, ratio = ReinforceStep().chosen(model, batch)
    assert torch.allclose(ratio, torch.ones_like(ratio))

    # 温度で標準偏差を広げた分布で選んだ経験は, 比が 1 より小さい要素を含む
    loader = RolloutLoader(
        BoxEnv(), model, episodes=2, exploration=Exploration(temperature=3.0)
    )
    _, _, ratio = ReinforceStep().chosen(model, next(iter(loader)))
    assert torch.all(ratio <= 1.0)
    assert torch.any(ratio < 1.0)

    # 評価は平均 (0) の行動で, 方策のエントロピーも記録
    env = BoxEnv()
    batch = next(iter(EvaluationLoader(env, model, episodes=1)))
    assert np.allclose(np.stack(env.actions), 0.0)
    outputs = ReinforceStep()(model, batch, "valid")
    normal = torch.distributions.Normal(0.0, 1.0)
    assert torch.isclose(outputs["entropy"], normal.entropy())


def test_box_loader_checks() -> None:
    """
    log_std の無いモデルと epsilon を拒むことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 標準偏差を持たないモデル
    with pytest.raises(ValueError, match="log_std"):
        RolloutLoader(BoxEnv(), nn.Linear(1, 1), episodes=1)

    # 一様に選ぶ割合は番号で選ぶ行動だけ
    with pytest.raises(ValueError, match="epsilon"):
        RolloutLoader(
            BoxEnv(), box_model(), episodes=1,
            exploration=Exploration(epsilon=0.1)
        )

    # 評価のエピソードは平均だけで集めるので log_std が無くても動くが,
    # step の検証のエントロピーには log_std が要るので分かるエラー
    batch = next(iter(EvaluationLoader(BoxEnv(), nn.Linear(1, 1))))
    assert batch["actions"].shape == (50, 1)
    with pytest.raises(ValueError, match="log_std"):
        ReinforceStep()(nn.Linear(1, 1), batch, "valid")

    # nn.Parameter でない log_std (バッファなど)
    model = nn.Linear(1, 1)
    model.register_buffer("log_std", torch.zeros(1))
    with pytest.raises(ValueError, match="nn.Parameter"):
        RolloutLoader(BoxEnv(), model, episodes=1)

    # 形が () の Box は未対応
    env = BoxEnv()
    env.action_space.shape = ()
    with pytest.raises(ValueError, match="未対応"):
        RolloutLoader(env, box_model(), episodes=1)

    # 番号と連続の行動の混ざった経験
    buffer = ExperienceBuffer()
    buffer.add([0.0], 0, 0.0)
    buffer.add([0.0], np.array([0.5]), 0.0, behavior_log_prob=-1.0)
    buffer.end_episode()
    with pytest.raises(ValueError, match="混ざって"):
        buffer.to_batch()


def test_box_learns_and_exports(tmp_path: Path) -> None:
    """
    連続の行動の方策を学習し, 守りと一緒に書き出して守り付きで動かせる
    ことを確かめる関数

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
        "optimizer:\n  name: Adam\n  params:\n    lr: 0.02\n"
        "trainer:\n  epochs: 10\n  monitor: valid/reward\n  mode: max\n"
    )
    torch.manual_seed(0)
    model = box_model()
    train_loader = RolloutLoader(
        BoxEnv(), model, episodes=8, batch_episodes=4, seed=0
    )
    valid_loader = EvaluationLoader(BoxEnv(), model, episodes=1)
    trainer = Trainer(model, experiment, ReinforceStep())
    history = trainer.fit(train_loader, valid_loader)

    # 目標 (0.5) に近い平均の行動 (ベストモデルの重みに復帰)
    rewards = history.get("valid/reward")
    assert rewards[0] < -0.5
    assert max(rewards) > -0.05
    with torch.no_grad():
        mean = float(model(torch.ones(1, 1))[0, 0])
    assert 0.3 < mean < 0.7

    # 空間より狭い行動の範囲の守りで書き出すと, 平均は守りで clip
    env = BoxEnv()
    guards = make_guards(
        env.observation_space, env.action_space, action_low=[-0.2],
        action_high=[0.2]
    )
    directory = export_policy(
        tmp_path / "policy", model, env.observation_space, env.action_space,
        guards
    )
    policy = load_policy(directory, GaussianNet(1, 1))
    result = policy.act(np.array([1.0], dtype=np.float32))
    assert result["status"] == "clipped"
    assert np.allclose(result["action"], [0.2])
    rows = evaluate_policy(policy, env, episodes=2, seed=0)
    assert [row["clipped"] for row in rows] == [5, 5]
    assert np.allclose(np.stack(env.actions), 0.2)


class MultiBoxEnv(BoxEnv):
    def __init__(self):
        # 親クラスのコンストラクタを呼び出す
        super(MultiBoxEnv, self).__init__()

        # 2 x 2 の形の行動の空間
        self.action_space = Space(
            low=[[-1.0, -2.0], [-3.0, -4.0]], high=[[1.0, 2.0], [3.0, 4.0]]
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        行動を記録して, 報酬 0 の 1 歩を進める関数

        Parameters
        ----------
        action: np.ndarray
            (2, 2) の連続の行動

        Returns
        ----------
        result: tuple[np.ndarray, float, bool, bool, dict]
            観測, 報酬, 終了, 打ち切り, info
        """
        # 渡された行動の記録と歩数の更新
        self.actions.append(np.array(action))
        self.t += 1
        observation = np.array([1.0], dtype=np.float32)
        result = (observation, 0.0, self.t >= self.length, False, {})

        return result


def test_box_multi_dim_and_log_std() -> None:
    """
    多次元の行動, 1 つの log_std, 要素の数の違い, log_std の範囲を
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 1 つの log_std を全ての要素に使い, 環境には空間の形で要素ごとに clip
    torch.manual_seed(0)
    env = MultiBoxEnv()
    model = GaussianNet(1, 4)
    with torch.no_grad():
        model.log_std = nn.Parameter(torch.full((1,), 1.5))
    batch = next(iter(RolloutLoader(env, model, episodes=1)))
    assert batch["actions"].shape == (5, 4)
    passed = np.stack(env.actions)
    assert passed.shape == (5, 2, 2)
    high = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.all(np.abs(passed) <= high)
    outputs = ReinforceStep()(model, batch, "train")
    assert torch.isfinite(outputs["loss"])

    # 要素の数が合わない log_std と平均
    model.log_std = nn.Parameter(torch.zeros(3))
    with pytest.raises(ValueError, match="log_std の要素の数"):
        next(iter(RolloutLoader(MultiBoxEnv(), model, episodes=1)))
    model = GaussianNet(1, 3)
    with pytest.raises(ValueError, match="出力の要素の数"):
        ReinforceStep()(model, batch, "train")
    with pytest.raises(ValueError, match="出力の要素の数"):
        next(iter(RolloutLoader(MultiBoxEnv(), model, episodes=1)))

    # 発散した log_std でも標準偏差は範囲の中で, 損失は有限
    model = GaussianNet(1, 4)
    with torch.no_grad():
        model.log_std.fill_(1e6)
    batch = next(iter(RolloutLoader(MultiBoxEnv(), model, episodes=1)))
    assert torch.all(torch.isfinite(batch["behavior_log_probs"]))
    outputs = ReinforceStep(entropy_coef=0.1)(model, batch, "train")
    assert torch.isfinite(outputs["loss"])

    # 範囲の外の log_std にも勾配が流れ, 更新で戻れる
    outputs["loss"].backward()
    assert torch.all(model.log_std.grad != 0)

    # 大きな値と無限大でも, 標準偏差は範囲の端の値で, 勾配は有限
    for value, expected in (
        (1e8, LOG_STD_MAX), (-1e8, LOG_STD_MIN), (float("inf"), LOG_STD_MAX),
        (float("-inf"), LOG_STD_MIN)
    ):
        model.log_std = nn.Parameter(torch.full((4,), value))
        std = gaussian_std(model, 4)
        assert torch.allclose(std, torch.full((4,), expected).exp())
        std.sum().backward()
        assert torch.all(torch.isfinite(model.log_std.grad))

        # 有限の値なら勾配も通る (無限大は更新で戻せないので 0)
        if abs(value) < float("inf"):
            assert torch.all(model.log_std.grad != 0)
    with torch.no_grad():
        model.log_std.fill_(-1e6)
    batch = next(iter(RolloutLoader(MultiBoxEnv(), model, episodes=1)))
    outputs = ReinforceStep()(model, batch, "train")
    assert torch.isfinite(outputs["loss"])
