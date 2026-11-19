from pathlib import Path

import torch
from torch import nn

from ml_found import Trainer, load_project_config, start_experiment
from ml_found.rl import (
    EvaluationLoader, Exploration, ReinforceStep, RolloutLoader,
    export_policy, load_policy, make_env, make_guards
)


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int):
        # 親クラスのコンストラクタを呼び出す
        super(PolicyNet, self).__init__()

        # 観測の次元数, 隠れ層の次元数, 行動の数を保存
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        # 線形層と活性化関数の積み重ね
        self.model = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            観測

        Returns
        ----------
        logits: torch.Tensor
            行動ごとのスコア
        """
        # 線形層と活性化関数に順に適用
        logits = self.model(x)

        return logits


# プロジェクトのディレクトリの決定
project_dir = Path(__file__).resolve().parent

# コマンドライン引数 (--experiment と「項目=値」) から config を読み込んで実験を開始
cfg, diff = load_project_config(str(project_dir))
experiment = start_experiment(cfg, diff, str(project_dir))

# 学習用と評価用の環境は別 (Gymnasium が無ければ入れ方を示して止まる)
train_env = make_env(cfg.env.id)
valid_env = make_env(cfg.env.id)
obs_dim = train_env.observation_space.shape[0]
n_actions = int(train_env.action_space.n)
model = PolicyNet(obs_dim, cfg.model.hidden_dim, n_actions)

# 1 エポックは rollout.episodes のエピソード, 評価は別の種のエピソード
train_loader = RolloutLoader(
    train_env, model, episodes=cfg.rollout.episodes,
    batch_episodes=cfg.rollout.batch_episodes, gamma=cfg.rollout.gamma,
    exploration=Exploration(
        cfg.rollout.temperature, cfg.rollout.epsilon
    ),
    seed=cfg.experiment.seed, safety_key=cfg.rollout.safety_key
)
valid_loader = EvaluationLoader(
    valid_env, model, episodes=cfg.evaluation.episodes,
    seed=cfg.env.valid_seed, safety_key=cfg.rollout.safety_key
)

# 学習
step = ReinforceStep(entropy_coef=cfg.loss.entropy_coef)
trainer = Trainer(model, experiment, step)
history = trainer.fit(train_loader, valid_loader)

# 止める条件 (台車の位置と棒の角度) と, 外れたら元の制御に戻す守りで
# 書き出し
limits = [cfg.guards.position_limit, None, cfg.guards.angle_limit, None]
guards = make_guards(
    train_env.observation_space, train_env.action_space,
    observation_low=[None if value is None else -value for value in limits],
    observation_high=limits, fallback="handover"
)
policy_dir = export_policy(
    experiment.run_dir / "policy", model, train_env.observation_space,
    train_env.action_space, guards,
    metadata={
        "env_id": cfg.env.id, "algorithm": "REINFORCE",
        "model": {"hidden_dim": cfg.model.hidden_dim}
    }
)

# 書き出した一式を読み直し, 1 つの観測で守りを当てた行動の確認
policy = load_policy(policy_dir, PolicyNet(obs_dim, cfg.model.hidden_dim,
                                           n_actions))
observation, _ = valid_env.reset(seed=cfg.env.valid_seed)
best_epoch = trainer.checkpoint.best_epoch
best_reward = history.get("valid/reward")[best_epoch - 1]
print(f"保存先: {experiment.run_dir}")
print(f"ベストモデルのエポック: {best_epoch}")
print(f"ベストモデルの評価のエピソードの報酬の和: {best_reward:.1f}")
print(f"方策の一式: {policy_dir}")
print(f"最初の観測での行動: {policy.act(observation)}")
