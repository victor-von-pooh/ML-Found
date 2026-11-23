from typing import Any, Optional

import numpy as np

from ml_found.rl.env import make_env
from ml_found.rl.trajectory import Trajectory, env_id

# 守りの結果の種類 (apply_guards の status. 表の列の順)
STATUSES = ("ok", "clipped", "fallback", "stop", "handover")

# 行動を返さずにエピソードを終える守りの結果
STOP_STATUSES = ("stop", "handover")


def check_episodes(episodes: Any) -> None:
    """
    エピソード数が 1 以上の整数 (真偽値を除く) であることを確かめる関数

    Parameters
    ----------
    episodes: Any
        エピソード数

    Returns
    ----------
    None
    """
    # bool は int の子なので除外
    if (
        isinstance(episodes, bool)
        or not isinstance(episodes, (int, np.integer)) or episodes < 1
    ):
        raise ValueError(
            f"episodes は 1 以上の整数にしてください: {episodes!r}"
        )


def check_optional_int(seed: Any, where: str) -> None:
    """
    乱数の種と歩数の上限が 0 以上の整数か None であることを確かめる関数

    Parameters
    ----------
    seed: Any
        確かめる値
    where: str
        誤りのメッセージに出す引数の名前

    Returns
    ----------
    None
    """
    if seed is None:
        return
    if (
        isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError(
            f"{where} は 0 以上の整数か None にしてください: {seed!r}"
        )


def check_trajectory(trajectory: Any) -> None:
    """
    軌跡が Trajectory で, まだ何も記録していないことを確かめる関数

    Parameters
    ----------
    trajectory: Any
        記録する軌跡か None

    Returns
    ----------
    None
    """
    # エピソードの番号が重ならないように, 1 回の評価に 1 つの軌跡
    if trajectory is None:
        return
    if not isinstance(trajectory, Trajectory):
        raise ValueError(
            f"trajectory は Trajectory か None にしてください: {trajectory!r}"
        )
    if trajectory.episodes or trajectory.current or (
        trajectory.dropped_episodes
    ):
        raise ValueError(
            "trajectory には何も記録していない Trajectory を渡してください "
            "(1 回の評価に 1 つ)"
        )


def run_guarded_episode(
    policy: Any, env: Any, seed: Optional[int], max_steps: Optional[int],
    trajectory: Optional[Trajectory] = None, episode: int = 0
) -> dict[str, Any]:
    """
    守りを当てた方策で 1 エピソードを動かし, 守りの結果を数える関数

    Parameters
    ----------
    policy: Any
        act(観測) が action と status の辞書を返す方策 (load_policy の方策)
    env: Any
        Gymnasium の形の環境
    seed: Optional[int]
        reset に渡す乱数の種, None の場合は環境の乱数の続き
    max_steps: Optional[int]
        1 エピソードの歩数の上限, None の場合は環境に任せる
    trajectory: Optional[Trajectory] = None
        1 歩ごとの観測, 行動, 報酬, 終わったか, 守りの結果を記録する軌跡,
        None の場合は記録しない
    episode: int = 0
        軌跡に書くエピソードの番号

    Returns
    ----------
    row: dict[str, Any]
        seed, reward (報酬の和), length (環境を進めた歩数), end (終えた
        理由), status ごとの数 (act を呼んだ回数. 止めた判断も含む)
    """
    # 環境の初期化と, 守りの結果の数の用意
    observation, _ = env.reset(seed=None if seed is None else int(seed))
    if trajectory is not None:
        trajectory.start_episode(episode, seed)
    counts = {status: 0 for status in STATUSES}
    total = 0.0
    length = 0
    end = None

    # 1 歩ごとの守りを当てた行動の決定. stop と handover は行動を返さない
    # ので, そこでエピソードを終了 (止めたか元の制御に戻した扱い)
    while end is None:
        result = policy.act(observation)
        status = result.get("status") if isinstance(result, dict) else None
        if status not in counts:
            raise ValueError(
                "方策の act は status が "
                f"{', '.join(STATUSES)} のどれかの辞書を返してください: "
                f"{result!r}"
            )
        counts[status] += 1
        reasons = result.get("reasons")
        if status in STOP_STATUSES:
            # 止めた判断も, 行動と報酬の無い行として軌跡に記録
            end = status
            if trajectory is not None:
                trajectory.add(
                    observation, None, None, False, False, status, reasons
                )
            continue
        if result.get("action") is None:
            raise ValueError(
                f"方策の act が status {status} で行動を返しませんでした: "
                f"{result!r}"
            )
        next_observation, reward, terminated, truncated, _ = env.step(
            result["action"]
        )
        value = float(np.asarray(reward, dtype=np.float64).item())
        total += value
        length += 1
        limited = max_steps is not None and length >= max_steps
        if terminated:
            end = "terminated"
        elif truncated:
            end = "truncated"
        elif limited:
            end = "max_steps"

        # 軌跡には行動の前の観測を記録. 歩数の上限で切った歩も打ち切り.
        # 方策が行動を選んだ確率 (action_prob) を返した場合はそれも記録
        if trajectory is not None:
            trajectory.add(
                observation, result["action"], value, bool(terminated),
                bool(truncated) or (limited and not terminated), status,
                reasons, result.get("action_prob")
            )
        observation = next_observation
    if trajectory is not None:
        trajectory.end_episode()
    row = {
        "seed": None if seed is None else int(seed),
        "reward": total,
        "length": length,
        "end": end,
        **counts
    }

    return row


def evaluate_policy(
    policy: Any, env_or_id: Any, episodes: int, seed: Optional[int] = None,
    max_steps: Optional[int] = None, trajectory: Optional[Trajectory] = None
) -> list[dict[str, Any]]:
    """
    書き出した方策を守り付きで環境で動かし, エピソードごとの結果を返す関数

    Parameters
    ----------
    policy: Any
        load_policy の方策 (act(観測) が action, status, reasons と,
        確率で選ぶ方策では任意で action_prob を返す)
    env_or_id: Any
        Gymnasium の形の環境か, Gymnasium に登録された環境の名前.
        名前の場合は make_env で作り (追加の依存 rl が要る), 終えたら閉じる
    episodes: int
        エピソード数
    seed: Optional[int] = None
        最初のエピソードの乱数の種. エピソードごとに seed, seed + 1, ...
        None の場合は環境の乱数の続き
    max_steps: Optional[int] = None
        1 エピソードの歩数の上限, None の場合は環境に任せる
    trajectory: Optional[Trajectory] = None
        1 歩ごとの観測, 行動, 報酬, 終わったか, 守りの結果を記録する
        空の軌跡 (save_trajectory で書き出す), None の場合は記録しない

    Returns
    ----------
    rows: list[dict[str, Any]]
        1 エピソードが 1 行の辞書. episode (0 から), seed, reward
        (報酬の和), length (歩数), end (terminated, truncated, max_steps,
        stop, handover), ok, clipped, fallback, stop, handover
        (守りの結果ごとの数). save_evaluation にそのまま渡せる
    """
    # 何も動かす前に, 回数と種の確認
    check_episodes(episodes)
    check_optional_int(seed, "seed")
    check_optional_int(max_steps, "max_steps")
    if max_steps is not None and max_steps < 1:
        raise ValueError(
            f"max_steps は 1 以上の整数か None にしてください: {max_steps!r}"
        )
    check_trajectory(trajectory)

    # 名前なら環境を作り (Gymnasium が無ければ入れ方を示して停止),
    # 終えたら閉じる処理 (渡された環境はそのまま)
    owned = isinstance(env_or_id, str)
    env = make_env(env_or_id) if owned else env_or_id
    rows = []
    try:
        # 軌跡の空間と出どころ (要素の数が上限を超えれば止まる)
        if trajectory is not None:
            trajectory.set_spaces(
                getattr(env, "observation_space", None),
                getattr(env, "action_space", None)
            )
            trajectory.check_spaces()
            trajectory.source = "evaluate_policy"
            trajectory.env_id = env_id(env)

        # エピソードごとに seed + 番号で, 守り付きで動かす
        for index in range(int(episodes)):
            episode_seed = None if seed is None else int(seed) + index
            row = run_guarded_episode(
                policy, env, episode_seed, max_steps, trajectory, index
            )
            rows.append({"episode": index, **row})
    finally:
        if owned:
            env.close()

    return rows
