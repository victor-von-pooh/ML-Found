import numpy as np
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


def eval_state_values(
    model: nn.Module, observations: torch.Tensor
) -> torch.Tensor:
    """
    推論モードのモデルで, 観測ごとの状態の価値を勾配なしで求める関数

    Parameters
    ----------
    model: nn.Module
        value(観測) で状態の価値を返すモデル
    observations: torch.Tensor
        観測のバッチ

    Returns
    ----------
    values: torch.Tensor
        (バッチ,) の状態の価値 (勾配は流さない)
    """
    # 推論モードにして勾配を記録せずに計算し, 元のモードに復帰
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            values = state_values(model, observations)
    finally:
        model.train(training)

    return values


def _to_array(tensor: torch.Tensor) -> np.ndarray:
    """
    テンソルを, 勾配を切った CPU の 64 ビットの浮動小数点数の配列にする関数

    Parameters
    ----------
    tensor: torch.Tensor
        1 次元のテンソル

    Returns
    ----------
    array: np.ndarray
        同じ値の 64 ビットの浮動小数点数の配列
    """
    # 勾配を切り, CPU の 64 ビットに変換
    array = tensor.detach().to(device="cpu", dtype=torch.float64).numpy()

    return array


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
    # 勾配を切り, CPU の 64 ビットの配列に変換
    reward_array = _to_array(rewards)
    value_array = _to_array(values)
    final_array = _to_array(final_values)

    # 次の状態の価値 (エピソードの最後の歩は最後の歩の後の状態の価値) と,
    # 重ねる重み (エピソードの最後の歩で 0 にして, 前のエピソードと分ける).
    # 歩数 0 のエピソードは歩が無いので除外
    length_array = np.asarray(lengths, dtype=np.int64)
    ends = np.cumsum(length_array)[length_array > 0] - 1
    next_values = np.empty_like(value_array)
    next_values[:-1] = value_array[1:]
    next_values[ends] = final_array[length_array > 0]
    decays = np.full_like(value_array, gamma * gae_lambda)
    decays[ends] = 0.0

    # 1 歩の TD の誤差をまとめて求め, バッチの後ろから重ねて計算
    deltas = reward_array + gamma * next_values - value_array
    summed = []
    running = 0.0
    for delta, decay in zip(deltas[::-1].tolist(), decays[::-1].tolist()):
        running = delta + decay * running
        summed.append(running)

    # 前からの順に戻し, 元のデバイスと型のテンソルに変換
    advantages = torch.tensor(
        summed[::-1], dtype=rewards.dtype, device=rewards.device
    )

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
            1 歩ごとの学習モードの状態の価値 (勾配を流す)
        advantages: torch.Tensor
            1 歩ごとの利得 (勾配は流さない)
        targets: torch.Tensor
            1 歩ごとの価値の関数の目標 (利得と推論モードの価値の和.
            勾配は流さない)
        """
        # 勾配を流す学習モードの価値と, 利得と目標に使う推論モードの価値
        # (行動を選んだときと同じモード. BatchNorm の統計を進めず,
        # Dropout の雑音を入れないため). 最後の観測の価値は打ち切りだけ
        values = state_values(model, batch["observations"])
        fixed = eval_state_values(model, batch["observations"])
        final_values = eval_state_values(
            model, batch["final_observations"]
        ) * (1.0 - batch["episode_terminated"])

        # エピソードごとの利得と, 利得と価値の和の目標
        lengths = [
            int(length) for length in batch["episode_lengths"].tolist()
        ]
        advantages = generalized_advantages(
            batch["rewards"], fixed, final_values, lengths,
            float(batch["gamma"]), self.gae_lambda
        )
        targets = advantages + fixed

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
            学習では方策の勾配の損失 + value_coef × 価値の関数の損失 -
            entropy_coef × エントロピー, 検証では報酬の和の平均の符号を
            反転した値を loss に入れ, 報酬, 歩数, 安全の違反の指標と,
            方策のエントロピー (entropy. 学習ではバッチの観測ごとの平均,
            検証では評価で通った観測ごとの平均), 学習では価値の関数の
            損失 (value_loss) を加える
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
