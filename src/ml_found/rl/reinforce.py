import torch
from torch import nn

from ml_found.rl.rollout import gaussian_std


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


def is_box_batch(batch: dict[str, torch.Tensor]) -> bool:
    """
    バッチの行動が連続の行動 (Box) かを確かめる関数

    Parameters
    ----------
    batch: dict[str, torch.Tensor]
        RolloutLoader か EvaluationLoader のバッチ

    Returns
    ----------
    is_box: bool
        行動が小数のテンソルの場合は True
    """
    # 番号は long, 連続の行動は float32
    is_box = batch["actions"].is_floating_point()

    return is_box


def gaussian_policy(
    model: nn.Module, observations: torch.Tensor, size: int
) -> torch.distributions.Normal:
    """
    観測ごとの連続の行動の正規分布を作る関数

    Parameters
    ----------
    model: nn.Module
        観測から行動の平均を返し, log_std を持つモデル
    observations: torch.Tensor
        観測のバッチ
    size: int
        行動の要素の数

    Returns
    ----------
    distribution: torch.distributions.Normal
        (バッチ, 要素の数) の正規分布
    """
    # 平らにした平均と, 観測によらない標準偏差
    mean = model(observations).reshape(observations.shape[0], -1)
    if mean.shape[1] != size:
        raise ValueError(
            f"モデルの出力の要素の数 {mean.shape[1]} が行動の要素の数 "
            f"{size} と違います"
        )
    std = gaussian_std(model, size).to(mean)
    distribution = torch.distributions.Normal(mean, std.expand_as(mean))

    return distribution


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

    def eval_chosen_log_probs(
        self, model: nn.Module, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        推論モードの正規分布の方策で, 選んだ連続の行動の対数の確率密度を
        求める関数

        Parameters
        ----------
        model: nn.Module
            観測から行動の平均を返し, log_std を持つモデル
        batch: dict[str, torch.Tensor]
            RolloutLoader のバッチ

        Returns
        ----------
        log_probs: torch.Tensor
            1 歩ごとの対数の確率密度 (勾配は流さない)
        """
        # 推論モードにして勾配を記録せずに計算し, 元のモードに復帰
        training = model.training
        model.eval()
        try:
            with torch.no_grad():
                actions = batch["actions"]
                distribution = gaussian_policy(
                    model, batch["observations"], actions.shape[1]
                )
                log_probs = distribution.log_prob(actions).sum(dim=-1)
        finally:
            model.train(training)

        return log_probs

    def chosen(
        self, model: nn.Module, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        選んだ行動の対数確率, エントロピー, 探索との比を求める関数

        Parameters
        ----------
        model: nn.Module
            方策のモデル
        batch: dict[str, torch.Tensor]
            RolloutLoader のバッチ

        Returns
        ----------
        chosen: torch.Tensor
            1 歩ごとの選んだ行動の対数確率 (連続の行動では確率密度)
        entropy: torch.Tensor
            1 歩ごとのエントロピーの平均
        ratio: torch.Tensor
            推論モードの方策と実際に選んだ確率の比 (1 で打ち切る.
            勾配は流さない)
        """
        # 連続の行動は正規分布の確率密度 (要素ごとの和) で, 比は対数の差
        if is_box_batch(batch):
            actions = batch["actions"]
            distribution = gaussian_policy(
                model, batch["observations"], actions.shape[1]
            )
            chosen = distribution.log_prob(actions).sum(dim=-1)
            entropy = distribution.entropy().sum(dim=-1).mean()
            ratio = (
                self.eval_chosen_log_probs(model, batch)
                - batch["behavior_log_probs"]
            ).exp().clamp(max=1.0)
            return chosen, entropy, ratio

        # 番号で選ぶ行動は softmax の確率
        log_probs = self.log_softmax(model(batch["observations"]))
        chosen = log_probs.gather(
            1, batch["actions"].unsqueeze(1)
        ).squeeze(1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
        ratio = (
            self.eval_chosen_probs(model, batch)
            / batch["behavior_probs"]
        ).clamp(max=1.0)

        return chosen, entropy, ratio

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
            if is_box_batch(batch):
                distribution = gaussian_policy(
                    model, batch["observations"], batch["actions"].shape[1]
                )
                entropy = distribution.entropy().sum(dim=-1).mean()
                return entropy
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
            観測を受け取り, 行動ごとのスコア (連続の行動では正規分布の
            平均) を返す方策のモデル (連続の行動では log_std も持つ)
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

        # 選んだ行動の対数確率とエントロピーと, 探索で選んだ確率との比
        # (1 で打ち切る) による, 方策と違う分布で集めた経験の重みの補正.
        # 行動を選んだときと同じ推論モードの確率で比べるので, Dropout が
        # あっても探索の無い設定では比は 1
        chosen, entropy, ratio = self.chosen(model, batch)

        # 基準付きの方策の勾配の損失
        advantage = self.advantages(batch["returns"])
        loss = -(ratio * advantage * chosen).mean()
        loss = loss - self.entropy_coef * entropy
        outputs["loss"] = loss
        outputs["entropy"] = entropy.detach()

        return outputs
