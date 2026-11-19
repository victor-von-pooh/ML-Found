import torch
from torch import nn


def episode_metrics(batch: dict[str, torch.Tensor]) -> dict:
    """
    バッチのエピソードごとの集計から指標を求める関数

    Parameters
    ----------
    batch: dict[str, torch.Tensor]
        RolloutLoader か EvaluationLoader のバッチ

    Returns
    ----------
    metrics: dict
        エピソードの報酬の和 (reward), 歩数 (length), 安全の違反の数
        (violations) の平均, 成功の割合 (success. 環境が知らせた場合),
        エピソードの数 (batch_size)
    """
    # エピソードの数で重み付けして平均するための batch_size の追加
    metrics = {
        "reward": batch["episode_rewards"].mean(),
        "length": batch["episode_lengths"].mean(),
        "violations": batch["episode_violations"].mean(),
        "batch_size": int(batch["episode_rewards"].numel())
    }
    if "episode_success" in batch:
        metrics["success"] = batch["episode_success"].mean()

    return metrics


class ReinforceStep:
    def __init__(self, entropy_coef: float = 0.0, normalize: bool = True):
        # 値の範囲の確認
        if entropy_coef < 0.0:
            raise ValueError(
                f"entropy_coef は 0 以上にしてください: {entropy_coef}"
            )

        # エントロピーの重みと, 利得を標準化するかを保存
        self.entropy_coef = entropy_coef
        self.normalize = normalize

        # 行動ごとの対数確率の計算
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def advantages(self, returns: torch.Tensor) -> torch.Tensor:
        """
        割引した報酬の和から, 平均を基準 (baseline) にした利得を求める関数

        Parameters
        ----------
        returns: torch.Tensor
            1 歩ごとの割引した報酬の和

        Returns
        ----------
        advantage: torch.Tensor
            平均を引いた値, normalize の場合は標準偏差でも割った値
        """
        # バッチの平均を基準にした分散の低減
        advantage = returns - returns.mean()
        if self.normalize and returns.numel() > 1:
            advantage = advantage / (returns.std() + 1e-8)

        return advantage

    def eval_chosen_probs(
        self, model: nn.Module, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        推論モードの方策で, 選んだ行動の確率を求める関数

        Parameters
        ----------
        model: nn.Module
            方策のモデル
        batch: dict[str, torch.Tensor]
            RolloutLoader のバッチ

        Returns
        ----------
        probs: torch.Tensor
            1 歩ごとの選んだ行動の確率 (勾配は流さない)
        """
        # 推論モードにして勾配を記録せずに計算し, 元のモードに復帰
        training = model.training
        model.eval()
        try:
            with torch.no_grad():
                log_probs = self.log_softmax(model(batch["observations"]))
        finally:
            model.train(training)
        probs = log_probs.gather(
            1, batch["actions"].unsqueeze(1)
        ).squeeze(1).exp()

        return probs

    def entropy(
        self, model: nn.Module, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        バッチの観測ごとの方策のエントロピーの平均を, 勾配なしで求める関数

        Parameters
        ----------
        model: nn.Module
            方策のモデル
        batch: dict[str, torch.Tensor]
            EvaluationLoader のバッチ

        Returns
        ----------
        entropy: torch.Tensor
            1 歩ごとのエントロピーの平均 (勾配は流さない)
        """
        # 評価の指標なので勾配を記録せずに計算
        with torch.no_grad():
            log_probs = self.log_softmax(model(batch["observations"]))
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

        return entropy

    def __call__(
        self, model: nn.Module, batch: dict[str, torch.Tensor], phase: str
    ) -> dict:
        """
        REINFORCE (基準付き) の損失と, エピソードの指標を求める関数

        Parameters
        ----------
        model: nn.Module
            観測を受け取り, 行動ごとのスコアを返す方策のモデル
        batch: dict[str, torch.Tensor]
            RolloutLoader (学習) か EvaluationLoader (検証) のバッチ
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            学習では方策の勾配の損失, 検証では報酬の和の平均の符号を
            反転した値を loss に入れ, 報酬, 歩数, 安全の違反の指標と
            方策のエントロピー (entropy) を加える
        """
        # エピソードの指標
        outputs = episode_metrics(batch)

        # 検証は報酬の和の平均の符号を反転した値が loss. 報酬が上限で
        # 並んでも動く指標として, 評価で通った観測の方策のエントロピーの平均
        if phase != "train":
            outputs["loss"] = -batch["episode_rewards"].mean()
            outputs["entropy"] = self.entropy(model, batch)
            return outputs

        # 選んだ行動の対数確率とエントロピー
        log_probs = self.log_softmax(model(batch["observations"]))
        chosen = log_probs.gather(
            1, batch["actions"].unsqueeze(1)
        ).squeeze(1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

        # 探索で選んだ確率との比 (1 で打ち切る) で, 方策と違う分布で
        # 集めた経験の重みの補正. 行動を選んだときと同じ推論モードの
        # 確率で比べるので, Dropout があっても探索の無い設定では比は 1
        ratio = (
            self.eval_chosen_probs(model, batch)
            / batch["behavior_probs"]
        ).clamp(max=1.0)

        # 基準付きの方策の勾配の損失
        advantage = self.advantages(batch["returns"])
        loss = -(ratio * advantage * chosen).mean()
        loss = loss - self.entropy_coef * entropy
        outputs["loss"] = loss
        outputs["entropy"] = entropy.detach()

        return outputs
