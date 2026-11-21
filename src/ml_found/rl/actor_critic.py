import torch
from torch import nn

from ml_found.rl.reinforce import ReinforceStep, episode_metrics


def state_values(
    model: nn.Module, observations: torch.Tensor
) -> torch.Tensor:
    """
    モデルの価値の関数で, 観測ごとの状態の価値を求める関数

    Parameters
    ----------
    model: nn.Module
        value(観測) で状態の価値を返すモデル
    observations: torch.Tensor
        観測のバッチ

    Returns
    ----------
    values: torch.Tensor
        (バッチ,) の状態の価値
    """
    # 価値の関数の無いモデルと, 大きさの合わない出力はエラー
    if not callable(getattr(model, "value", None)):
        raise ValueError(
            "ActorCriticStep のモデルは, 観測から状態の価値を返す value "
            "を持ってください"
        )
    values = model.value(observations).reshape(-1)
    if values.numel() != observations.shape[0]:
        raise ValueError(
            f"value の出力の要素の数 {values.numel()} が観測の数 "
            f"{observations.shape[0]} と違います"
        )

    return values


def generalized_advantages(
    rewards: torch.Tensor, values: torch.Tensor, final_values: torch.Tensor,
    lengths: list[int], gamma: float, gae_lambda: float
) -> torch.Tensor:
    """
    エピソードごとに, 価値の関数で見積もった利得 (GAE) を求める関数

    Parameters
    ----------
    rewards: torch.Tensor
        1 歩ごとの報酬
    values: torch.Tensor
        1 歩ごとの状態の価値 (勾配は流さない)
    final_values: torch.Tensor
        エピソードごとの最後の歩の後の状態の価値 (終了なら 0)
    lengths: list[int]
        エピソードごとの歩数 (バッチの中の順)
    gamma: float
        割引率
    gae_lambda: float
        利得の見積もりの重み (0 なら 1 歩の TD, 1 ならモンテカルロ)

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


class ActorCriticStep(ReinforceStep):
    def __init__(
        self, value_coef: float = 0.5, entropy_coef: float = 0.0,
        gae_lambda: float = 0.95, normalize: bool = True
    ):
        # 親クラスのコンストラクタを呼び出す
        super(ActorCriticStep, self).__init__(entropy_coef, normalize)

        # 値の範囲の確認
        if value_coef < 0.0:
            raise ValueError(f"value_coef は 0 以上にしてください: {value_coef}")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError(
                f"gae_lambda は 0 以上 1 以下にしてください: {gae_lambda}"
            )

        # 価値の損失の重みと, 利得の見積もりの重みを保存
        self.value_coef = value_coef
        self.gae_lambda = gae_lambda

        # 価値の関数の損失
        self.value_loss = nn.MSELoss()

    def estimate(
        self, model: nn.Module, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        状態の価値と, 利得と, 価値の関数の目標を求める関数

        Parameters
        ----------
        model: nn.Module
            value(観測) で状態の価値を返すモデル
        batch: dict[str, torch.Tensor]
            RolloutLoader のバッチ

        Returns
        ----------
        values: torch.Tensor
            1 歩ごとの状態の価値 (勾配を流す)
        advantages: torch.Tensor
            1 歩ごとの利得 (勾配は流さない)
        targets: torch.Tensor
            1 歩ごとの価値の関数の目標 (利得と価値の和. 勾配は流さない)
        """
        # 1 歩ごとの価値と, 打ち切りのエピソードだけ最後の観測の価値
        values = state_values(model, batch["observations"])
        with torch.no_grad():
            final_values = state_values(
                model, batch["final_observations"]
            ) * (1.0 - batch["episode_terminated"])
            lengths = [
                int(length) for length in batch["episode_lengths"].tolist()
            ]
            advantages = generalized_advantages(
                batch["rewards"], values.detach(), final_values, lengths,
                float(batch["gamma"]), self.gae_lambda
            )
            targets = advantages + values.detach()

        return values, advantages, targets

    def __call__(
        self, model: nn.Module, batch: dict[str, torch.Tensor], phase: str
    ) -> dict:
        """
        Actor-Critic (価値の関数で利得を見積もる方法) の損失と,
        エピソードの指標を求める関数

        Parameters
        ----------
        model: nn.Module
            観測から行動ごとのスコア (連続の行動では正規分布の平均. log_std
            も持つ) を返し, value(観測) で状態の価値を返すモデル
        batch: dict[str, torch.Tensor]
            RolloutLoader (学習) か EvaluationLoader (検証) のバッチ
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            学習では方策の勾配の損失と価値の関数の損失の和, 検証では報酬の
            和の平均の符号を反転した値を loss に入れ, 報酬, 歩数, 安全の
            違反の指標, エントロピー (entropy), 学習では価値の関数の損失
            (value_loss) を加える
        """
        # 検証は ReinforceStep と同じ指標
        if phase != "train":
            outputs = super(ActorCriticStep, self).__call__(
                model, batch, phase
            )
            return outputs

        # エピソードの指標と, 選んだ行動の対数確率, エントロピー, 比
        outputs = episode_metrics(batch)
        chosen, entropy, ratio = self.chosen(model, batch)

        # 価値の関数で見積もった利得 (normalize の場合は標準化) で方策の
        # 勾配, 価値の関数は利得と価値の和を目標に学習
        values, advantages, targets = self.estimate(model, batch)
        if self.normalize and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )
        policy_loss = -(ratio * advantages * chosen).mean()
        value_loss = self.value_loss(values, targets)
        loss = policy_loss + self.value_coef * value_loss
        loss = loss - self.entropy_coef * entropy
        outputs["loss"] = loss
        outputs["entropy"] = entropy.detach()
        outputs["value_loss"] = value_loss.detach()

        return outputs
