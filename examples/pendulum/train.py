from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from ml_found import (
    Trainer, load_project_config, save_evaluation, start_experiment,
    write_task
)
from ml_found.rl import (
    ActorCriticStep, EvaluationLoader, Exploration, RolloutLoader,
    evaluate_policy, export_policy, load_policy, make_env, make_guards
)


class ScaledReward:
    def __init__(self, env: Any, scale: float):
        # 元の環境と, 報酬に掛ける値を保存
        self.env = env
        self.scale = scale

        # 観測と行動の空間は元の環境のもの
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, seed: Optional[int] = None) -> tuple[Any, dict]:
        """
        エピソードを始める関数

        Parameters
        ----------
        seed: Optional[int] = None
            乱数の種

        Returns
        ----------
        result: tuple[Any, dict]
            最初の観測と info
        """
        # 元の環境の初期化
        result = self.env.reset(seed=seed)

        return result

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """
        元の環境を 1 歩進め, 報酬に決めた値を掛ける関数

        Parameters
        ----------
        action: Any
            行動

        Returns
        ----------
        result: tuple[Any, float, bool, bool, dict]
            観測, 縮めた報酬, 終了, 打ち切り, info
        """
        # 元の環境の 1 歩と報酬の縮小
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        result = (
            observation, float(reward) * self.scale, terminated, truncated,
            info
        )

        return result


class ActorCriticNet(nn.Module):
    def __init__(
        self, obs_dim: int, hidden_dim: int, action_dim: int, log_std: float
    ):
        # 親クラスのコンストラクタを呼び出す
        super(ActorCriticNet, self).__init__()

        # 観測の次元数, 隠れ層の次元数, 行動の次元数を保存
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # 方策 (行動の平均) の層の積み重ね. 範囲の外の平均は, 学習では
        # 環境の行動の空間で, 書き出した方策では守りで clip
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )

        # 価値の関数の層の積み重ね
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )

        # 観測によらない, 正規分布の方策の対数の標準偏差
        self.log_std = nn.Parameter(torch.full((self.action_dim,), log_std))

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
            行動の平均 (書き出した方策の行動)
        """
        # 方策の層に適用
        mean = self.actor(x)

        return mean

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
        # 価値の関数の層に適用
        values = self.critic(x)

        return values


# プロジェクトのディレクトリの決定
project_dir = Path(__file__).resolve().parent

# コマンドライン引数 (--experiment と「項目=値」) から config を読み込んで実験を開始
cfg, diff = load_project_config(str(project_dir))
experiment = start_experiment(cfg, diff, str(project_dir))

# 何の実験か. 方策は出力の形が行動, 正解の出どころが環境の報酬
write_task(experiment, {
    "role": "effect",
    "output": "action",
    "what": "Pendulum の観測から振り子に掛けるトルクを決める方策",
    "target": {
        "kind": "reward",
        "origin": {"type": "environment", "name": cfg.env.id}
    },
    "method": "scratch"
})

# 学習用と評価用の環境は別 (Gymnasium が無ければ入れ方を示して止まる)
# 学習の環境だけ報酬を縮め (価値の関数の損失の尺度を調えるため),
# 評価の環境は元の報酬のまま (train/reward は縮めた尺度, valid/reward と
# 守り付きの評価は元の尺度)
train_env = ScaledReward(make_env(cfg.env.id), cfg.rollout.reward_scale)
valid_env = make_env(cfg.env.id)
obs_dim = train_env.observation_space.shape[0]
action_dim = train_env.action_space.shape[0]
model_args = {
    "obs_dim": obs_dim, "hidden_dim": cfg.model.hidden_dim,
    "action_dim": action_dim, "log_std": cfg.model.log_std
}
model = ActorCriticNet(**model_args)

# 1 エポックは rollout.episodes のエピソード, 評価は別の種のエピソード
train_loader = RolloutLoader(
    train_env, model, episodes=cfg.rollout.episodes,
    batch_episodes=cfg.rollout.batch_episodes, gamma=cfg.rollout.gamma,
    exploration=Exploration(cfg.rollout.temperature),
    seed=cfg.experiment.seed, safety_key=cfg.rollout.safety_key
)
valid_loader = EvaluationLoader(
    valid_env, model, episodes=cfg.evaluation.episodes,
    seed=cfg.env.valid_seed, safety_key=cfg.rollout.safety_key
)

# 価値の関数で利得を見積もる Actor-Critic で学習
step = ActorCriticStep(
    value_coef=cfg.loss.value_coef, entropy_coef=cfg.loss.entropy_coef,
    gae_lambda=cfg.loss.gae_lambda
)
trainer = Trainer(model, experiment, step)
history = trainer.fit(train_loader, valid_loader)

# トルクを空間より狭い範囲に clip し, 振り子の角速度が範囲を外れたら
# トルク 0 にする守りで書き出し
torque = cfg.guards.torque_limit
speed = cfg.guards.speed_limit
guards = make_guards(
    valid_env.observation_space, valid_env.action_space,
    action_low=[-torque] * action_dim, action_high=[torque] * action_dim,
    observation_low=[None, None, -speed],
    observation_high=[None, None, speed], fallback="action"
)
policy_dir = export_policy(
    experiment.run_dir / "policy", model, valid_env.observation_space,
    valid_env.action_space, guards,
    metadata={
        "env_id": cfg.env.id, "algorithm": "ActorCritic",
        "model": {
            "class": f"{ActorCriticNet.__module__}.ActorCriticNet",
            "args": model_args
        }
    }
)

# 書き出した一式を metadata の引数で組み立て直したモデルに読み直し,
# 守り付きで動かした結果を守る指標の表に保存
policy = load_policy(policy_dir, ActorCriticNet(**model_args))
observation, _ = valid_env.reset(seed=cfg.env.valid_seed)
first_action = policy.act(observation)
rows = evaluate_policy(
    policy, valid_env, episodes=cfg.evaluation.episodes,
    seed=cfg.env.valid_seed
)
save_evaluation(
    experiment, "guarded_episodes", rows, "simulator",
    description="書き出した方策を読み直し, 守りを当てて動かしたエピソード",
    guard=True, epoch=trainer.checkpoint.best_epoch
)
best_epoch = trainer.checkpoint.best_epoch
best_reward = history.get("valid/reward")[best_epoch - 1]
print(f"保存先: {experiment.run_dir}")
print(f"ベストモデルのエポック: {best_epoch}")
print(f"ベストモデルの評価のエピソードの報酬の和: {best_reward:.1f}")
print(f"方策の一式: {policy_dir}")
print(f"最初の観測での行動: {first_action}")
print(
    "守り付きの評価の報酬の和の平均: "
    f"{sum(row['reward'] for row in rows) / len(rows):.1f}, "
    f"clipped {sum(row['clipped'] for row in rows)}, "
    f"fallback {sum(row['fallback'] for row in rows)}"
)
