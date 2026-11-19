from typing import Any, Optional

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
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.behavior_probs: list[float] = []
        self.returns: list[float] = []

        # 途中のエピソードの記録
        self.current: list[tuple[np.ndarray, int, float, float]] = []
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
        self, observation: Any, action: int, reward: float,
        behavior_prob: float = 1.0, violation: bool = False
    ) -> None:
        """
        1 歩分の経験を途中のエピソードに加える関数

        Parameters
        ----------
        observation: Any
            行動を選んだときの観測
        action: int
            選んだ行動
        reward: float
            行動で得た報酬
        behavior_prob: float = 1.0
            探索を含めて, 実際にその行動を選んだ確率
        violation: bool = False
            その歩で安全の違反があったか

        Returns
        ----------
        None
        """
        # 観測は float32 の配列にして保存
        obs = np.asarray(observation, dtype=np.float32)
        self.current.append(
            (obs, int(action), float(reward), float(behavior_prob))
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
        for (obs, action, reward, prob), value in zip(self.current, returns):
            self.observations.append(obs)
            self.actions.append(action)
            self.rewards.append(reward)
            self.behavior_probs.append(prob)
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
            1 歩ごとの observations, actions, returns, behavior_probs と,
            エピソードごとの episode_rewards, episode_lengths,
            episode_violations (環境が成功を知らせた場合は episode_success)
        """
        # 途中のエピソードと空の入れ物はエラー
        if self.current:
            raise ValueError("途中のエピソードがあります. end_episode を呼んでください")
        if not self.episodes:
            raise ValueError("終えたエピソードがありません")

        # 1 歩ごととエピソードごとのテンソルに変換
        batch = {
            "observations": torch.from_numpy(np.stack(self.observations)),
            "actions": torch.tensor(self.actions, dtype=torch.long),
            "returns": torch.tensor(self.returns, dtype=torch.float32),
            "behavior_probs": torch.tensor(
                self.behavior_probs, dtype=torch.float32
            ),
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
