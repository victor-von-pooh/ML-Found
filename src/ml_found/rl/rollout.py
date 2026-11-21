import math
from typing import Any, Iterator, Optional

import numpy as np
import torch
from torch import nn

from ml_found.rl.buffer import ExperienceBuffer

# 連続の行動の方策の対数の標準偏差の範囲 (発散して inf や nan にしないため)
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class Exploration:
    def __init__(self, temperature: float = 1.0, epsilon: float = 0.0):
        # 値の範囲の確認
        if not temperature > 0.0:
            raise ValueError(
                f"temperature は 0 より大きくしてください: {temperature}"
            )
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon は 0 以上 1 以下にしてください: {epsilon}")

        # 方策の分布の温度と, 一様に選ぶ割合を保存
        self.temperature = temperature
        self.epsilon = epsilon

    def __repr__(self) -> str:
        """
        設定を文字にする関数

        Parameters
        ----------
        None

        Returns
        ----------
        text: str
            温度と一様に選ぶ割合
        """
        # ログに残す形
        text = (
            f"Exploration(temperature={self.temperature}, "
            f"epsilon={self.epsilon})"
        )

        return text


def behavior_probs(
    logits: torch.Tensor, exploration: Exploration
) -> torch.Tensor:
    """
    探索の設定を入れた, 行動を選ぶ確率を求める関数

    Parameters
    ----------
    logits: torch.Tensor
        方策のモデルの出力 (行動ごとのスコア. 最後の次元が行動)
    exploration: Exploration
        探索の設定

    Returns
    ----------
    probs: torch.Tensor
        温度で割った softmax と一様な分布を epsilon で混ぜた確率
    """
    # 温度で割った分布と一様な分布の混合
    softmax = nn.Softmax(dim=-1)
    probs = softmax(logits / exploration.temperature)
    n_actions = logits.shape[-1]
    probs = (1.0 - exploration.epsilon) * probs + (
        exploration.epsilon / n_actions
    )

    return probs


def model_scores(model: nn.Module, observation: Any) -> torch.Tensor:
    """
    推論モードのモデルで, 1 件の観測の出力を勾配なしで求める関数

    Parameters
    ----------
    model: nn.Module
        方策のモデル
    observation: Any
        環境の観測

    Returns
    ----------
    output: torch.Tensor
        1 件分の出力 (discrete なら行動ごとのスコア, box なら行動の平均)
    """
    # モデルと同じデバイスで, 推論モードにして勾配を記録せずに計算
    # (1 件の観測の BatchNorm と, Dropout の違いで確率がずれるのを避ける
    # ため. 元のモードに復帰)
    device = model_device(model)
    obs = torch.as_tensor(
        np.asarray(observation, dtype=np.float32), device=device
    )
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(obs.unsqueeze(0))[0]
    finally:
        model.train(training)

    return output


def gaussian_std(model: nn.Module, size: int) -> torch.Tensor:
    """
    連続の行動の方策の標準偏差を, モデルの log_std から求める関数

    Parameters
    ----------
    model: nn.Module
        log_std (行動の要素ごとか 1 つの対数の標準偏差) を持つモデル
    size: int
        行動の要素の数

    Returns
    ----------
    std: torch.Tensor
        要素が size 個の標準偏差 (log_std は LOG_STD_MIN 以上
        LOG_STD_MAX 以下に制限)
    """
    # log_std の無いモデルと, 要素の数が合わない log_std はエラー
    if not isinstance(getattr(model, "log_std", None), torch.Tensor):
        raise ValueError(
            "連続の行動 (Box) の方策のモデルは, 正規分布の対数の標準偏差 "
            "log_std (nn.Parameter) を持ってください"
        )
    log_std = model.log_std.reshape(-1)
    if log_std.numel() not in (1, size):
        raise ValueError(
            f"log_std の要素の数 {log_std.numel()} は 1 か行動の要素の数 "
            f"{size} にしてください"
        )

    # 発散して標準偏差が 0 か無限大にならないように, 決めた範囲に制限
    log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
    std = log_std.exp().expand(size)

    return std


def select_action(
    model: nn.Module, observation: Any, exploration: Exploration,
    greedy: bool
) -> tuple[int, float]:
    """
    観測から番号で選ぶ行動を 1 つ選ぶ関数

    Parameters
    ----------
    model: nn.Module
        観測を受け取り, 行動ごとのスコアを返すモデル
    observation: Any
        環境の観測
    exploration: Exploration
        探索の設定 (greedy の場合は使わない)
    greedy: bool
        最もスコアの高い行動を選ぶ場合は True (評価のエピソード)

    Returns
    ----------
    action: int
        選んだ行動の番号 (0 から)
    prob: float
        その行動を選んだ確率 (greedy の場合は 1)
    """
    # 評価は最良の行動, 学習は探索を入れた分布から選択
    logits = model_scores(model, observation)
    if greedy:
        action = int(logits.argmax().item())
        prob = 1.0
        return action, prob
    probs = behavior_probs(logits.float(), exploration)
    action = int(torch.multinomial(probs, 1).item())
    prob = float(probs[action].item())

    return action, prob


def select_box_action(
    model: nn.Module, observation: Any, exploration: Exploration,
    greedy: bool
) -> tuple[np.ndarray, float]:
    """
    観測から連続の行動を, 正規分布の方策で 1 つ選ぶ関数

    Parameters
    ----------
    model: nn.Module
        観測を受け取り行動の平均を返し, log_std を持つモデル
    observation: Any
        環境の観測
    exploration: Exploration
        探索の設定 (温度を標準偏差に掛ける. greedy の場合は使わない)
    greedy: bool
        平均の行動を選ぶ場合は True (評価のエピソード)

    Returns
    ----------
    action: np.ndarray
        選んだ行動 (平らな float32. 空間の範囲で clip する前の値)
    log_prob: float
        その行動を選んだ確率密度の対数 (greedy の場合は 0)
    """
    # 評価は平均の行動, 学習は温度を掛けた標準偏差の正規分布から選択
    mean = model_scores(model, observation).float().reshape(-1)
    if greedy:
        action = mean.cpu().numpy().astype(np.float32)
        log_prob = 0.0
        return action, log_prob
    with torch.no_grad():
        std = gaussian_std(model, mean.numel()).to(mean) * (
            exploration.temperature
        )
        distribution = torch.distributions.Normal(mean, std)
        sample = distribution.sample()
        log_prob = float(distribution.log_prob(sample).sum().item())
    action = sample.cpu().numpy().astype(np.float32)

    return action, log_prob


def to_env_action(action: Any, space: Any) -> Any:
    """
    選んだ行動を, 環境に渡す形にする関数

    Parameters
    ----------
    action: Any
        選んだ行動 (番号か, 平らな連続の行動)
    space: Any
        行動の空間 (box の場合は low, high, shape を持つ)

    Returns
    ----------
    env_action: Any
        番号はそのまま, 連続の行動は空間の範囲で clip して形を戻した値
    """
    # 番号で選ぶ行動はそのまま
    if isinstance(action, (int, np.integer)):
        env_action = action
        return env_action

    # 要素の数の確認と, 空間の範囲 (書き出しの守りの範囲の外側) での clip
    low = np.asarray(space.low, dtype=np.float64).reshape(-1)
    high = np.asarray(space.high, dtype=np.float64).reshape(-1)
    if np.size(action) != low.size:
        raise ValueError(
            f"モデルの出力の要素の数 {np.size(action)} が行動の要素の数 "
            f"{low.size} と違います"
        )
    clipped = np.clip(np.asarray(action, dtype=np.float64), low, high)
    env_action = clipped.astype(np.float32).reshape(tuple(space.shape))

    return env_action


def model_device(model: nn.Module) -> torch.device:
    """
    モデルのパラメータがあるデバイスを返す関数

    Parameters
    ----------
    model: nn.Module
        方策のモデル

    Returns
    ----------
    device: torch.device
        最初のパラメータのデバイス, パラメータが無い場合は cpu
    """
    # パラメータの無いモデルは cpu で計算
    parameter = next(model.parameters(), None)
    device = (
        parameter.device if parameter is not None else torch.device("cpu")
    )

    return device


def is_violation(info: Any, safety_key: Optional[str]) -> bool:
    """
    環境の info から安全の違反を読み取る関数

    Parameters
    ----------
    info: Any
        環境の step が返した info
    safety_key: Optional[str]
        違反を表す info のキー, None の場合は数えない

    Returns
    ----------
    violated: bool
        値が 0 より大きい (True を含む) 場合は True
    """
    # キーが無い場合と数えない設定は違反なし
    if safety_key is None or not isinstance(info, dict):
        violated = False
        return violated
    value = info.get(safety_key, 0)
    violated = bool(np.any(np.asarray(value, dtype=np.float64) > 0))

    return violated


def run_episode(
    env: Any, model: nn.Module, buffer: ExperienceBuffer,
    exploration: Exploration, greedy: bool, seed: Optional[int],
    max_steps: Optional[int], safety_key: Optional[str],
    success_key: Optional[str], kind: str = "discrete"
) -> dict[str, Any]:
    """
    Gymnasium の形の環境で 1 エピソードを動かし, 経験を貯める関数

    Parameters
    ----------
    env: Any
        reset(seed=...) が (観測, info) を, step(行動) が (観測, 報酬,
        終了, 打ち切り, info) を返す環境
    model: nn.Module
        方策のモデル
    buffer: ExperienceBuffer
        経験を貯める入れ物
    exploration: Exploration
        探索の設定
    greedy: bool
        最もスコアの高い行動を選ぶ場合は True
    seed: Optional[int]
        reset に渡す乱数の種, None の場合は環境の乱数の続き
    max_steps: Optional[int]
        1 エピソードの歩数の上限, None の場合は環境に任せる
    safety_key: Optional[str]
        安全の違反を表す info のキー
    success_key: Optional[str]
        成功を表す info のキー (最後の歩の値を読む)
    kind: str = "discrete"
        行動の空間の種類 (discrete か box)

    Returns
    ----------
    stats: dict[str, Any]
        エピソードの報酬の和, 歩数, 安全の違反の数, 成功
    """
    # 環境の初期化と, 終了か打ち切りまでの行動の選択
    observation, info = env.reset(seed=seed)
    steps = 0
    done = False
    ended = False
    while not done:
        # 連続の行動は clip する前の値を記録 (選んだ確率密度と合わせるため)
        if kind == "box":
            action, log_prob = select_box_action(
                model, observation, exploration, greedy
            )
        else:
            action, prob = select_action(
                model, observation, exploration, greedy
            )
            log_prob = None
        next_observation, reward, terminated, truncated, info = env.step(
            to_env_action(action, env.action_space)
        )
        if kind == "box":
            buffer.add(
                observation, action, float(reward),
                violation=is_violation(info, safety_key),
                behavior_log_prob=log_prob
            )
        else:
            buffer.add(
                observation, action, float(reward), prob,
                is_violation(info, safety_key)
            )
        observation = next_observation
        steps += 1
        ended = bool(terminated)
        done = ended or bool(truncated) or (
            max_steps is not None and steps >= max_steps
        )

    # 成功を知らせるキーがある場合だけ, 最後の歩の値を成功として記録
    success = None
    if success_key is not None and isinstance(info, dict):
        if success_key in info:
            success = bool(info[success_key])
    # 打ち切りの後の価値を見積もれるように, 終了かと最後の観測も記録
    stats = buffer.end_episode(success, ended, observation)

    return stats


def action_kind(env: Any) -> str:
    """
    環境の行動の空間が, 番号で選ぶ形 (Discrete) か連続 (Box) かを返す関数

    Parameters
    ----------
    env: Any
        Gymnasium の形の環境

    Returns
    ----------
    kind: str
        discrete か box (行動の空間を持たない環境は discrete)
    """
    # 行動の空間が無い場合は番号で選ぶ形
    space = getattr(env, "action_space", None)
    if space is None:
        kind = "discrete"
        return kind

    # 0 から始まる Discrete (MultiBinary なども n を持つので, 形が () で
    # あることも確認)
    shape = tuple(getattr(space, "shape", ()) or ())
    if isinstance(getattr(space, "n", None), (int, np.integer)):
        if shape != ():
            raise ValueError(
                "強化学習の補助は, 行動を番号で選ぶ環境 (Discrete) か連続の"
                f"行動の環境 (Box) だけに対応します: {space!r}"
            )
        if int(getattr(space, "start", 0)) != 0:
            raise ValueError(
                "強化学習の補助は, 行動の番号が 0 から始まる環境だけに"
                f"対応します: {space!r}"
            )
        kind = "discrete"
        return kind

    # 下限と上限と形を持つ Box (形が () の Box は未対応)
    if not (hasattr(space, "low") and hasattr(space, "high")):
        raise ValueError(
            "強化学習の補助は, 行動を番号で選ぶ環境 (Discrete) か連続の"
            f"行動の環境 (Box) だけに対応します: {space!r}"
        )
    if not shape:
        raise ValueError(
            "形が () の Box の行動は未対応です. 形を (1,) にしてください: "
            f"{space!r}"
        )
    kind = "box"

    return kind


def check_box_model(model: nn.Module, exploration: Exploration) -> None:
    """
    連続の行動の学習に使うモデルと探索の設定を確かめる関数

    Parameters
    ----------
    model: nn.Module
        方策のモデル
    exploration: Exploration
        探索の設定

    Returns
    ----------
    None
    """
    # 標準偏差は学習するモデルの log_std, 一様に選ぶ割合は番号で選ぶ行動だけ
    if not isinstance(getattr(model, "log_std", None), nn.Parameter):
        raise ValueError(
            "連続の行動 (Box) の方策のモデルは, 正規分布の対数の標準偏差 "
            "log_std を nn.Parameter で持ってください"
        )
    if exploration.epsilon > 0.0:
        raise ValueError(
            "連続の行動 (Box) では epsilon は使えません (探索は temperature "
            f"で標準偏差を広げます): {exploration.epsilon}"
        )


def check_count(name: str, value: Any, optional: bool = False) -> None:
    """
    回数の引数が 1 以上の整数 (真偽値を除く) であることを確かめる関数

    Parameters
    ----------
    name: str
        引数の名前
    value: Any
        引数の値
    optional: bool = False
        None を許す場合は True

    Returns
    ----------
    None
    """
    # bool は int の子なので除外
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} は 1 以上の整数にしてください: {value!r}")


class RolloutLoader:
    def __init__(
        self, env: Any, model: nn.Module, episodes: Optional[int] = None,
        steps: Optional[int] = None, batch_episodes: int = 1,
        gamma: float = 0.99, exploration: Optional[Exploration] = None,
        seed: Optional[int] = None, max_episode_steps: Optional[int] = None,
        safety_key: Optional[str] = "cost",
        success_key: Optional[str] = None
    ):
        # エポックの長さはエピソードの数か歩数のどちらか一方
        if (episodes is None) == (steps is None):
            raise ValueError("episodes と steps のどちらか一方を指定してください")
        check_count("episodes", episodes, optional=True)
        check_count("steps", steps, optional=True)
        check_count("batch_episodes", batch_episodes)
        check_count("max_episode_steps", max_episode_steps, optional=True)
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or (
            not 0.0 <= gamma <= 1.0
        ):
            raise ValueError(f"gamma は 0 以上 1 以下にしてください: {gamma!r}")
        self.kind = action_kind(env)
        if self.kind == "box":
            check_box_model(model, exploration or Exploration())

        # 環境, モデル, エポックの長さ, まとめるエピソードの数を保存
        self.env = env
        self.model = model
        self.episodes = episodes
        self.steps = steps
        self.batch_episodes = batch_episodes

        # 割引率, 探索の設定, 乱数の種, 歩数の上限, info のキーを保存
        self.gamma = gamma
        self.exploration = exploration or Exploration()
        self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.safety_key = safety_key
        self.success_key = success_key

        # 最初の reset だけ種を渡すための印
        self.seeded = False

    def __len__(self) -> int:
        """
        1 エポックのバッチの数を返す関数 (エピソードの数で決める場合だけ)

        Parameters
        ----------
        None

        Returns
        ----------
        n_batches: int
            1 エポックのバッチの数
        """
        # 歩数で決める場合はエピソードの長さで変わるのでエラー
        if self.episodes is None:
            raise TypeError("steps で決めた RolloutLoader の長さは分かりません")
        n_batches = math.ceil(self.episodes / self.batch_episodes)

        return n_batches

    def next_seed(self) -> Optional[int]:
        """
        次の reset に渡す乱数の種を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        seed: Optional[int]
            最初の reset だけ種, その後は None (環境の乱数の続き)
        """
        # 最初だけ種を渡して, 後は環境の乱数を続けて使用
        seed = None if self.seeded else self.seed
        self.seeded = True

        return seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        今の方策でエピソードを集め, batch_episodes ごとにバッチを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        batches: Iterator[dict[str, torch.Tensor]]
            ExperienceBuffer.to_batch の形のバッチ
        """
        # 推論モード (trainer.eval_train の評価し直し) では, 学習の環境の
        # 乱数を進めて結果を変えてしまうのでエラー
        if not self.model.training:
            raise ValueError(
                "RolloutLoader は学習だけに使えます. trainer.eval_train は "
                "false にし, 評価には EvaluationLoader を使ってください"
            )

        # エピソードの数か歩数がエポックの長さに達するまでの収集
        # 歩数で決める場合は, 達したエピソードの終わりまで
        buffer = ExperienceBuffer(self.gamma)
        n_episodes = 0
        n_steps = 0
        while True:
            stats = run_episode(
                self.env, self.model, buffer, self.exploration, False,
                self.next_seed(), self.max_episode_steps, self.safety_key,
                self.success_key, self.kind
            )
            n_episodes += 1
            n_steps += stats["length"]
            finished = (
                n_episodes >= self.episodes if self.episodes is not None
                else n_steps >= self.steps
            )

            # まとめる数に達したか, エポックの最後でバッチを返却
            if finished or len(buffer.episodes) >= self.batch_episodes:
                yield buffer.to_batch()
                buffer.clear()
            if finished:
                break


class EvaluationLoader:
    def __init__(
        self, env: Any, model: nn.Module, episodes: int = 10,
        seed: int = 0, max_episode_steps: Optional[int] = None,
        safety_key: Optional[str] = "cost",
        success_key: Optional[str] = None
    ):
        # エピソードの数, 乱数の種, 歩数の上限と, 行動の形の確認
        check_count("episodes", episodes)
        check_count("max_episode_steps", max_episode_steps, optional=True)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed は整数にしてください: {seed!r}")
        self.kind = action_kind(env)

        # 評価用の環境, モデル, エピソードの数, 乱数の種を保存
        self.env = env
        self.model = model
        self.episodes = episodes
        self.seed = seed

        # 歩数の上限と info のキーを保存
        self.max_episode_steps = max_episode_steps
        self.safety_key = safety_key
        self.success_key = success_key

    def __len__(self) -> int:
        """
        1 回の評価のバッチの数を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        n_batches: int
            全てのエピソードを 1 つのバッチにまとめるので 1
        """
        # 評価のエピソードはまとめて 1 つのバッチ
        n_batches = 1

        return n_batches

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        毎回同じ種のエピソードを, 最良の行動で動かしてバッチを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        batches: Iterator[dict[str, torch.Tensor]]
            全ての評価のエピソードをまとめた 1 つのバッチ
        """
        # エポックの間で比べられるように, エピソードごとに seed + 番号で初期化
        buffer = ExperienceBuffer()
        for index in range(self.episodes):
            run_episode(
                self.env, self.model, buffer, Exploration(), True,
                self.seed + index, self.max_episode_steps, self.safety_key,
                self.success_key, self.kind
            )
        batch = buffer.to_batch()

        yield batch
