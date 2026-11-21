from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from ml_found.rl import (
    ActorCriticStep, EvaluationLoader, ExperienceBuffer, RolloutLoader,
    export_policy, load_policy
)
from ml_found.rl.actor_critic import generalized_advantages
from ml_found.training.trainer import Trainer
from test_rl import ToyEnv
from test_rl_box import BoxEnv
from test_trainer import make_experiment


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int, n_outputs: int, gaussian: bool):
        # 親クラスのコンストラクタを呼び出す
        super(ActorCriticNet, self).__init__()

        # 観測の次元数, 出力の数, 正規分布の方策かを保存
        self.obs_dim = obs_dim
        self.n_outputs = n_outputs
        self.gaussian = gaussian

        # 方策の線形層と価値の関数の線形層
        self.policy = nn.Linear(self.obs_dim, self.n_outputs)
        self.critic = nn.Linear(self.obs_dim, 1)
        with torch.no_grad():
            self.policy.weight.zero_()
            self.policy.bias.zero_()

        # 連続の行動では観測によらない対数の標準偏差
        if self.gaussian:
            self.log_std = nn.Parameter(torch.zeros(self.n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            観測

        Returns
        ----------
        output: torch.Tensor
            行動ごとのスコアか行動の平均
        """
        # 方策の線形層
        output = self.policy(x)

        return output

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """
        状態の価値を求める関数

        Parameters
        ----------
        x: torch.Tensor
            観測

        Returns
        ----------
        values: torch.Tensor
            (バッチ, 1) の状態の価値
        """
        # 価値の関数の線形層
        values = self.critic(x)

        return values


def test_generalized_advantages() -> None:
    """
    終了と打ち切りのエピソードの利得を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    rewards = torch.tensor([1.0, 1.0, 2.0])
    values = torch.tensor([0.5, 0.25, 1.0])

    # lambda 1 の終了のエピソードは, 割引した報酬の和から価値を引いた値
    final = torch.tensor([0.0, 0.0])
    advantages = generalized_advantages(
        rewards, values, final, [2, 1], 0.5, 1.0
    )
    assert torch.allclose(advantages, torch.tensor([1.0, 0.75, 1.0]))

    # lambda 0 の打ち切りのエピソードは, 最後の観測の価値で 1 歩の TD
    final = torch.tensor([4.0, 2.0])
    advantages = generalized_advantages(
        rewards, values, final, [2, 1], 0.5, 0.0
    )
    expected = torch.tensor([
        1.0 + 0.5 * 0.25 - 0.5, 1.0 + 0.5 * 4.0 - 0.25, 2.0 + 0.5 * 2.0 - 1.0
    ])
    assert torch.allclose(advantages, expected)


def test_batch_terminated_and_final_observation() -> None:
    """
    終了か打ち切りかと, 最後の歩の後の観測をバッチに入れることを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 歩数の上限で切ったエピソードは打ち切りで, 最後の観測は次の観測
    model = nn.Linear(1, 2)
    batch = next(iter(RolloutLoader(
        ToyEnv(length=5), model, episodes=1, max_episode_steps=3, gamma=0.9
    )))
    assert batch["episode_terminated"].tolist() == [0.0]
    assert batch["final_observations"].tolist() == [[3.0]]
    assert batch["rewards"].shape == (3,)
    assert float(batch["gamma"]) == pytest.approx(0.9)

    # 環境の終了は 1, 最後の観測を渡さない場合は最後の歩の観測
    batch = next(iter(RolloutLoader(ToyEnv(length=2), model, episodes=1)))
    assert batch["episode_terminated"].tolist() == [1.0]
    buffer = ExperienceBuffer()
    buffer.add([7.0], 0, 1.0)
    buffer.end_episode(terminated=False)
    assert buffer.to_batch()["final_observations"].tolist() == [[7.0]]


def test_actor_critic_checks() -> None:
    """
    価値の関数の無いモデルと範囲の外の引数を拒むことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    model = nn.Linear(1, 2)
    batch = next(iter(RolloutLoader(ToyEnv(), model, episodes=1)))
    with pytest.raises(ValueError, match="value"):
        ActorCriticStep()(model, batch, "train")
    with pytest.raises(ValueError, match="value_coef"):
        ActorCriticStep(value_coef=-1.0)
    with pytest.raises(ValueError, match="gae_lambda"):
        ActorCriticStep(gae_lambda=1.5)


def test_actor_critic_learns_discrete(tmp_path: Path) -> None:
    """
    番号で選ぶ行動の環境で, Actor-Critic で報酬の出る行動を学ぶことを
    確かめる関数

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
    torch.manual_seed(0)
    model = ActorCriticNet(1, 2, gaussian=False)
    train_loader = RolloutLoader(
        ToyEnv(), model, episodes=8, batch_episodes=4, seed=0
    )
    valid_loader = EvaluationLoader(ToyEnv(), model, episodes=2)
    trainer = Trainer(model, experiment, ActorCriticStep())
    history = trainer.fit(train_loader, valid_loader)

    # 価値の関数の損失も記録し, 最後は全て報酬の出る行動
    assert len(history.get("train/value_loss")) == 8
    assert history.get("valid/reward")[-1] == 5.0


def test_actor_critic_learns_box_and_exports(tmp_path: Path) -> None:
    """
    連続の行動の環境で Actor-Critic で学び, 価値の関数ごと書き出して
    読み直せることを確かめる関数

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
    model = ActorCriticNet(1, 1, gaussian=True)
    train_loader = RolloutLoader(
        BoxEnv(), model, episodes=8, batch_episodes=4, seed=0
    )
    valid_loader = EvaluationLoader(BoxEnv(), model, episodes=1)
    trainer = Trainer(model, experiment, ActorCriticStep())
    history = trainer.fit(train_loader, valid_loader)
    rewards = history.get("valid/reward")
    assert rewards[0] < -0.5
    assert max(rewards) > -0.05

    # 価値の関数の重みも一式に入り, 同じ組み立てのモデルに読み直せる
    env = BoxEnv()
    directory = export_policy(
        tmp_path / "policy", model, env.observation_space, env.action_space
    )
    policy = load_policy(directory, ActorCriticNet(1, 1, gaussian=True))
    result = policy.act(np.array([1.0], dtype=np.float32))
    with torch.no_grad():
        expected = model(torch.ones(1, 1)).numpy()[0]
    assert result["status"] == "ok"
    assert np.allclose(result["action"], expected)
