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


def stepwise_advantages(
    rewards: torch.Tensor, values: torch.Tensor, final_values: torch.Tensor,
    lengths: list[int], gamma: float, gae_lambda: float
) -> torch.Tensor:
    """
    比べる元として, 1 歩ずつのテンソルの添字で利得 (GAE) を求める関数

    Parameters
    ----------
    rewards: torch.Tensor
        1 歩ごとの報酬
    values: torch.Tensor
        1 歩ごとの状態の価値
    final_values: torch.Tensor
        エピソードごとの最後の歩の後の状態の価値 (終了なら 0)
    lengths: list[int]
        エピソードごとの歩数 (バッチの中の順)
    gamma: float
        割引率
    gae_lambda: float
        利得の見積もりの重み

    Returns
    ----------
    advantages: torch.Tensor
        1 歩ごとの利得
    """
    # エピソードの後ろから, 次の状態の価値で 1 歩の TD の誤差を重ねて計算
    advantages = torch.zeros_like(rewards)
    start = 0
    for index, length in enumerate(lengths):
        next_value = final_values[index]
        running = rewards.new_zeros(())
        for t in reversed(range(start, start + length)):
            delta = rewards[t] + gamma * next_value - values[t]
            running = delta + gamma * gae_lambda * running
            advantages[t] = running
            next_value = values[t]
        start += length

    return advantages


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_generalized_advantages_matches_stepwise(
    dtype: torch.dtype, seed: int
) -> None:
    """
    乱数のバッチで, 1 歩ずつの添字の計算と同じ利得になることを確かめる
    関数

    Parameters
    ----------
    dtype: torch.dtype
        報酬と価値の型
    seed: int
        乱数の種

    Returns
    ----------
    None
    """
    # 長さ 1 を含む複数のエピソードと, 終了と打ち切りの混ざったバッチ
    generator = np.random.default_rng(seed)
    lengths = [1] + generator.integers(1, 30, size=20).tolist() + [1]
    total = sum(lengths)
    rewards = torch.tensor(generator.normal(size=total), dtype=dtype)
    values = torch.tensor(generator.normal(size=total), dtype=dtype)
    terminated = torch.tensor(
        generator.integers(0, 2, size=len(lengths)), dtype=dtype
    )
    final_values = torch.tensor(
        generator.normal(size=len(lengths)), dtype=dtype
    ) * (1.0 - terminated)

    # 割引率と重みの組ごとに, 型と値の一致
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    for gamma, gae_lambda in [(0.99, 0.95), (0.9, 0.0), (0.5, 1.0)]:
        advantages = generalized_advantages(
            rewards, values, final_values, lengths, gamma, gae_lambda
        )
        expected = stepwise_advantages(
            rewards, values, final_values, lengths, gamma, gae_lambda
        )
        assert advantages.dtype == dtype
        assert advantages.device == rewards.device
        assert torch.allclose(
            advantages, expected, rtol=tolerance, atol=tolerance
        )


def test_generalized_advantages_without_gradient() -> None:
    """
    勾配を持つ価値を渡しても, 利得が勾配を持たないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 勾配を持つ価値と, 勾配の無い利得
    rewards = torch.tensor([1.0, 0.0, 2.0])
    values = torch.tensor([0.5, 0.25, 1.0], requires_grad=True)
    final_values = torch.tensor([0.0, 3.0], requires_grad=True)
    advantages = generalized_advantages(
        rewards, values, final_values, [2, 1], 0.9, 0.95
    )
    expected = stepwise_advantages(
        rewards, values.detach(), final_values.detach(), [2, 1], 0.9, 0.95
    )
    assert not advantages.requires_grad
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

    # 環境の終了は 1, 終了で最後の観測を渡さない場合は最後の歩の観測
    batch = next(iter(RolloutLoader(ToyEnv(length=2), model, episodes=1)))
    assert batch["episode_terminated"].tolist() == [1.0]
    buffer = ExperienceBuffer()
    buffer.add([7.0], 0, 1.0)
    buffer.end_episode()
    assert buffer.to_batch()["final_observations"].tolist() == [[7.0]]


def test_truncated_episode_needs_final_observation() -> None:
    """
    打ち切りのエピソードでは最後の歩の後の観測が要り, 渡した観測を
    バッチに入れることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 最後の観測の無い打ち切りはエラーで, 途中のエピソードは残る
    buffer = ExperienceBuffer()
    buffer.add([7.0], 0, 1.0)
    with pytest.raises(ValueError, match="最後の歩の後の観測"):
        buffer.end_episode(terminated=False)
    assert len(buffer.current) == 1

    # 渡した観測を最後の観測として記録
    buffer.end_episode(terminated=False, final_observation=[8.0])
    batch = buffer.to_batch()
    assert batch["episode_terminated"].tolist() == [0.0]
    assert batch["final_observations"].tolist() == [[8.0]]


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


def test_estimate_matches_hand_calculation() -> None:
    """
    終了と打ち切りの混ざったバッチの利得と目標が, 手で求めた値と一致する
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 状態の価値が観測そのものになるモデル
    model = ActorCriticNet(1, 2, gaussian=False)
    with torch.no_grad():
        model.critic.weight.fill_(1.0)
        model.critic.bias.zero_()

    # 1 つ目は 2 歩で終了 (最後の後の価値は 0), 2 つ目は 1 歩で打ち切り
    # (最後の観測 5 の価値で見積もり)
    batch = {
        "observations": torch.tensor([[1.0], [2.0], [3.0]]),
        "rewards": torch.tensor([1.0, 1.0, 2.0]),
        "episode_lengths": torch.tensor([2.0, 1.0]),
        "episode_terminated": torch.tensor([1.0, 0.0]),
        "final_observations": torch.tensor([[9.0], [5.0]]),
        "gamma": torch.tensor(0.5)
    }
    values, advantages, targets = ActorCriticStep(gae_lambda=0.5).estimate(
        model, batch
    )

    # 手で求めた値. 1 つ目の 2 歩目は 1 + 0.5 x 0 - 2 = -1, 1 歩目は
    # (1 + 0.5 x 2 - 1) + 0.5 x 0.5 x (-1) = 0.75, 2 つ目は
    # 2 + 0.5 x 5 - 3 = 1.5. 目標は利得と価値の和
    assert torch.allclose(advantages, torch.tensor([0.75, -1.0, 1.5]))
    assert torch.allclose(targets, torch.tensor([1.75, 1.0, 4.5]))
    assert values.requires_grad
    assert not advantages.requires_grad


def test_value_loss_decreases_on_fixed_batch() -> None:
    """
    固定のバッチで数歩更新すると, 価値の関数の損失が下がることを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    torch.manual_seed(0)
    model = ActorCriticNet(1, 2, gaussian=False)
    batch = next(iter(RolloutLoader(ToyEnv(), model, episodes=2)))
    step = ActorCriticStep(gae_lambda=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # 方策と価値の関数の損失の和で更新
    losses = []
    for _ in range(50):
        outputs = step(model, batch, "train")
        optimizer.zero_grad()
        outputs["loss"].backward()
        optimizer.step()
        losses.append(float(outputs["value_loss"]))
    assert losses[-1] < 0.25 * losses[0]

    # value_coef が 0 なら価値の関数は学ばない
    model = ActorCriticNet(1, 2, gaussian=False)
    before = model.critic.weight.detach().clone()
    outputs = ActorCriticStep(value_coef=0.0)(model, batch, "train")
    outputs["loss"].backward()
    assert model.critic.weight.grad is None or torch.all(
        model.critic.weight.grad == 0
    )
    assert torch.equal(model.critic.weight, before)


class BatchNormNet(ActorCriticNet):
    def __init__(self):
        # 親クラスのコンストラクタを呼び出す
        super(BatchNormNet, self).__init__(1, 2, gaussian=False)

        # BatchNorm を挟んだ価値の関数
        self.critic = nn.Sequential(
            nn.Linear(1, 4), nn.BatchNorm1d(4), nn.Linear(4, 1)
        )


def test_estimate_uses_eval_mode_with_batch_norm() -> None:
    """
    1 エピソードのバッチでも BatchNorm のモデルで動き, 推論モードの価値で
    利得を求めることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    torch.manual_seed(0)
    model = BatchNormNet()
    batch = next(iter(RolloutLoader(
        ToyEnv(), model, episodes=1, batch_episodes=1
    )))

    # 最後の観測 1 件の価値も推論モードで求めるのでエラーにならない
    # BatchNorm の統計は, 勾配を流す学習モードの 1 回だけで更新
    tracked = int(model.critic[1].num_batches_tracked)
    outputs = ActorCriticStep()(model, batch, "train")
    assert torch.isfinite(outputs["loss"])
    assert int(model.critic[1].num_batches_tracked) == tracked + 1
    assert model.training

    # 利得と目標は推論モードの価値と一致
    _, advantages, targets = ActorCriticStep().estimate(model, batch)
    model.eval()
    with torch.no_grad():
        fixed = model.value(batch["observations"]).reshape(-1)
    model.train()
    assert torch.allclose(targets - advantages, fixed, atol=1e-6)


def hand_batch() -> dict[str, torch.Tensor]:
    """
    終了と打ち切りの混ざった, 手で組んだ学習のバッチを作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    batch: dict[str, torch.Tensor]
        3 歩 (2 歩で終了と, 1 歩で打ち切り) のバッチ. returns は
        REINFORCE の利得にすり替えると損失が変わる値
    """
    # 1 つ目は 2 歩で終了, 2 つ目は 1 歩で打ち切り (最後の観測は 5)
    batch = {
        "observations": torch.tensor([[1.0], [2.0], [3.0]]),
        "actions": torch.tensor([0, 1, 0]),
        "rewards": torch.tensor([1.0, 1.0, 2.0]),
        "returns": torch.tensor([1.5, 1.0, 2.0]),
        "behavior_probs": torch.tensor([1.0, 1.0, 1.0]),
        "behavior_log_probs": torch.tensor([0.0, 0.0, 0.0]),
        "episode_rewards": torch.tensor([2.0, 2.0]),
        "episode_lengths": torch.tensor([2.0, 1.0]),
        "episode_violations": torch.tensor([0.0, 0.0]),
        "episode_terminated": torch.tensor([1.0, 0.0]),
        "final_observations": torch.tensor([[9.0], [5.0]]),
        "gamma": torch.tensor(0.5)
    }

    return batch


def test_policy_loss_uses_estimated_advantages() -> None:
    """
    学習の損失の方策の勾配の項が, estimate() の利得 (GAE) を使うことを,
    手で組んだ損失と比べて確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 状態の価値が観測そのもので, 観測で行動の確率が変わるモデル
    model = ActorCriticNet(1, 2, gaussian=False)
    with torch.no_grad():
        model.critic.weight.fill_(1.0)
        model.critic.bias.zero_()
        model.policy.weight.copy_(torch.tensor([[0.5], [-0.5]]))
    batch = hand_batch()

    # 選んだ行動の対数確率 (探索の確率を方策の確率にそろえて比は 1) と,
    # test_estimate_matches_hand_calculation と同じ手で求めた利得
    log_probs = nn.LogSoftmax(dim=-1)(
        model(batch["observations"])
    ).detach()
    chosen = log_probs.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
    batch["behavior_probs"] = chosen.exp()
    batch["behavior_log_probs"] = chosen
    advantages = torch.tensor([0.75, -1.0, 1.5])

    # 標準化しない利得と, 価値の関数の損失を除いた方策の損失
    outputs = ActorCriticStep(
        value_coef=0.0, gae_lambda=0.5, normalize=False
    )(model, batch, "train")
    expected = -(advantages * chosen).mean()
    assert torch.allclose(outputs["loss"], expected)

    # REINFORCE の利得 (割引した報酬の和から平均を引いた値) とは違う
    reinforce = batch["returns"] - batch["returns"].mean()
    assert not torch.allclose(outputs["loss"], -(reinforce * chosen).mean())

    # 標準化した利得と, 価値の関数の損失 (価値は観測, 目標は利得と価値の和)
    outputs = ActorCriticStep(
        value_coef=0.5, gae_lambda=0.5
    )(model, batch, "train")
    normalized = (advantages - advantages.mean()) / (
        advantages.std() + 1e-8
    )
    targets = torch.tensor([1.75, 1.0, 4.5])
    value_loss = ((torch.tensor([1.0, 2.0, 3.0]) - targets) ** 2).mean()
    expected = -(normalized * chosen).mean() + 0.5 * value_loss
    assert torch.allclose(outputs["value_loss"], value_loss)
    assert torch.allclose(outputs["loss"], expected)
