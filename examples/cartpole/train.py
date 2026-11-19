from pathlib import Path

import torch
from torch import nn

from ml_found import (
    Trainer, load_project_config, save_evaluation, start_experiment,
    write_task
)
from ml_found.rl import (
    EvaluationLoader, Exploration, ReinforceStep, RolloutLoader,
    evaluate_policy, export_policy, load_policy, make_env, make_guards
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

# 何の実験か. 方策は出力の形が行動, 正解の出どころが環境の報酬
write_task(experiment, {
    "role": "effect",
    "output": "action",
    "what": "CartPole の観測から押す向きを選ぶ方策",
    "target": {
        "kind": "reward",
        "origin": {"type": "environment", "name": cfg.env.id}
    },
    "method": "scratch"
})

# 学習用と評価用の環境は別 (Gymnasium が無ければ入れ方を示して止まる)
train_env = make_env(cfg.env.id)
valid_env = make_env(cfg.env.id)
obs_dim = train_env.observation_space.shape[0]
n_actions = int(train_env.action_space.n)
model_args = {
    "obs_dim": obs_dim, "hidden_dim": cfg.model.hidden_dim,
    "n_actions": n_actions
}
model = PolicyNet(**model_args)

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
        "model": {"class": "train.PolicyNet", "args": model_args}
    }
)

# 書き出した一式を metadata の引数で組み立て直したモデルに読み直し,
# 守り付きで動かした結果を守る指標の表に
policy = load_policy(policy_dir, PolicyNet(**model_args))
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
    f"handover {sum(row['handover'] for row in rows)}"
)
