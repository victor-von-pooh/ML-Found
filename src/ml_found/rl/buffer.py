import math
from typing import Any, Optional, Union

import numpy as np
import torch


class ExperienceBuffer:
    def __init__(self, gamma: float = 0.99):
        # 割引率の確認と保存
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma は 0 以上 1 以下にしてください: {gamma}")
        self.gamma = gamma

        # 入れ物の用意
        self.clear()

    def clear(self) -> None:
        """
        貯めた経験を全て捨てる関数

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 1 歩ごとの記録 (終えたエピソードの分)
        self.observations: list[np.ndarray] = []
        self.actions: list[Union[int, np.ndarray]] = []
        self.rewards: list[float] = []
        self.behavior_log_probs: list[float] = []
        self.returns: list[float] = []

        # 途中のエピソードの記録
        self.current: list[
            tuple[np.ndarray, Union[int, np.ndarray], float, float]
        ] = []
        self.current_violations = 0

        # 終えたエピソードごとの集計
        self.episodes: list[dict[str, Any]] = []

    def __len__(self) -> int:
        """
        終えたエピソードの歩数の合計を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        steps: int
            終えたエピソードの歩数の合計
        """
        # 途中のエピソードは除外
        steps = len(self.actions)

        return steps

    def add(
        self, observation: Any, action: Any, reward: float,
        behavior_prob: float = 1.0, violation: bool = False,
        behavior_log_prob: Optional[float] = None
    ) -> None:
        """
        1 歩分の経験を途中のエピソードに加える関数

        Parameters
        ----------
        observation: Any
            行動を選んだときの観測
        action: Any
            選んだ行動 (番号か, 連続の行動の配列)
        reward: float
            行動で得た報酬
        behavior_prob: float = 1.0
            探索を含めて, 実際にその行動を選んだ確率
        violation: bool = False
            その歩で安全の違反があったか
        behavior_log_prob: Optional[float] = None
            選んだ確率 (連続の行動では確率密度) の対数. 指定した場合は
            behavior_prob の代わりに使う

        Returns
        ----------
        None
        """
        # 観測は float32 の配列, 行動は番号か平らな float32 の配列,
        # 選んだ確率は対数にして保存
        obs = np.asarray(observation, dtype=np.float32)
        if isinstance(action, (int, np.integer)):
            value = int(action)
        else:
            value = np.asarray(action, dtype=np.float32).reshape(-1)
        if behavior_log_prob is None:
            behavior_log_prob = (
                math.log(behavior_prob) if behavior_prob > 0.0 else -math.inf
            )
        self.current.append(
            (obs, value, float(reward), float(behavior_log_prob))
        )
        self.current_violations += int(bool(violation))

    def end_episode(
        self, success: Optional[bool] = None
    ) -> dict[str, Any]:
        """
        途中のエピソードを終え, 割引した報酬の和を求めて貯める関数

        終了 (terminated) と打ち切り (truncated. TimeLimit や
        max_episode_steps) を区別せず, 最後の歩の後の報酬を 0 として
        求める (モンテカルロの REINFORCE). 打ち切りの後の価値を見積もって
        足す方法は, 価値の関数を持つ方法 (Actor-Critic など) で行う

        Parameters
        ----------
        success: Optional[bool] = None
            環境が成功を知らせた場合はその値, 知らせない場合は None

        Returns
        ----------
        stats: dict[str, Any]
            エピソードの報酬の和 (reward), 歩数 (length), 安全の違反の数
            (violations), 成功 (success)
        """
        # 歩の無いエピソードはエラー
        if not self.current:
            raise ValueError("歩の無いエピソードは終えられません")

        # 後ろから割引した報酬の和の計算
        returns = []
        running = 0.0
        for _, _, reward, _ in reversed(self.current):
            running = reward + self.gamma * running
            returns.append(running)
        returns.reverse()

        # 1 歩ごとの記録への移動
        for (obs, action, reward, log_prob), value in zip(
            self.current, returns
        ):
            self.observations.append(obs)
            self.actions.append(action)
            self.rewards.append(reward)
            self.behavior_log_probs.append(log_prob)
            self.returns.append(value)

        # エピソードの集計を残して途中の記録を空に
        stats = {
            "reward": sum(step[2] for step in self.current),
            "length": len(self.current),
            "violations": self.current_violations,
            "success": success
        }
        self.episodes.append(stats)
        self.current = []
        self.current_violations = 0

        return stats

    def to_batch(self) -> dict[str, torch.Tensor]:
        """
        終えたエピソードの経験を, step に渡すテンソルの辞書にする関数

        Parameters
        ----------
        None

        Returns
        ----------
        batch: dict[str, torch.Tensor]
            1 歩ごとの observations, actions (番号は long, 連続の行動は
            (歩数, 要素の数) の float32), returns, behavior_log_probs,
            behavior_probs (番号の行動の確率. 連続の行動では確率密度で,
            小さい値は 0 にアンダーフローするので使わず,
            behavior_log_probs を使う) と, エピソードごとの
            episode_rewards, episode_lengths, episode_violations (環境が
            成功を知らせた場合は episode_success)
        """
        # 途中のエピソードと空の入れ物はエラー
        if self.current:
            raise ValueError("途中のエピソードがあります. end_episode を呼んでください")
        if not self.episodes:
            raise ValueError("終えたエピソードがありません")

        # 1 歩ごととエピソードごとのテンソルに変換
        log_probs = torch.tensor(self.behavior_log_probs, dtype=torch.float32)
        batch = {
            "observations": torch.from_numpy(np.stack(self.observations)),
            "actions": self.actions_tensor(),
            "returns": torch.tensor(self.returns, dtype=torch.float32),
            "behavior_log_probs": log_probs,
            "behavior_probs": log_probs.exp(),
            "episode_rewards": torch.tensor(
                [stats["reward"] for stats in self.episodes],
                dtype=torch.float32
            ),
            "episode_lengths": torch.tensor(
                [stats["length"] for stats in self.episodes],
                dtype=torch.float32
            ),
            "episode_violations": torch.tensor(
                [stats["violations"] for stats in self.episodes],
                dtype=torch.float32
            )
        }

        # 全てのエピソードで成功が分かる場合だけ成功の列を追加
        successes = [stats["success"] for stats in self.episodes]
        if all(value is not None for value in successes):
            batch["episode_success"] = torch.tensor(
                successes, dtype=torch.float32
            )

        return batch

    def actions_tensor(self) -> torch.Tensor:
        """
        貯めた行動をテンソルにする関数

        Parameters
        ----------
        None

        Returns
        ----------
        actions: torch.Tensor
            番号なら (歩数,) の long, 連続の行動なら (歩数, 要素の数) の
            float32
        """
        # 番号と連続の行動が混ざる場合はエラー
        is_index = [isinstance(action, int) for action in self.actions]
        if all(is_index):
            actions = torch.tensor(self.actions, dtype=torch.long)
            return actions
        if any(is_index):
            raise ValueError("番号の行動と連続の行動が混ざっています")
        actions = torch.from_numpy(np.stack(self.actions))

        return actions
