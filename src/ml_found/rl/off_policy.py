import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from ml_found.rl.trajectory import (
    ACTION_PREFIX, GUARD_COLUMNS, MAX_TRAJECTORY_BYTES,
    MAX_TRAJECTORY_INFO_BYTES, OBSERVATION_PREFIX, PROB_COLUMN, Trajectory,
    is_prefixed, space_info
)
from ml_found.utils.files import read_plain_file

# 見積もりに要る列 (観測と行動の他)
REQUIRED_COLUMNS = ("episode", "step", "reward", "terminated", "truncated")

# 有効なサンプルの数がエピソードの数のこの割合より少なければ知らせる
LOW_ESS_RATIO = 0.1

# 守りで止めた判断の結果 (行動と報酬の無い行)
STOP_GUARDS = ("stop", "handover")


def column_index(column: str, prefix: str) -> int:
    """
    観測か行動の列の, 要素の番号を返す関数

    Parameters
    ----------
    column: str
        列の名前 (prefix か prefix_<番号>)
    prefix: str
        obs か action

    Returns
    ----------
    index: int
        prefix_<番号> の番号, prefix そのものは 0
    """
    index = 0 if column == prefix else int(column[len(prefix) + 1:])

    return index


def element_columns(columns: list[str], prefix: str) -> list[str]:
    """
    表の列から, 観測か行動の列を要素の順に選ぶ関数

    Parameters
    ----------
    columns: list[str]
        表の列
    prefix: str
        obs か action

    Returns
    ----------
    elements: list[str]
        prefix か prefix_<番号> の列 (番号の順)
    """
    # 1 つの値の列と番号の列が並ぶ表, 列の無い表は読まない
    elements = sorted(
        (column for column in columns if is_prefixed(column, prefix)),
        key=lambda column: column_index(column, prefix)
    )
    if not elements:
        raise ValueError(
            f"軌跡の表に {prefix} か {prefix}_<番号> の列がありません"
        )
    if prefix in elements and len(elements) > 1:
        raise ValueError(
            f"軌跡の表に {prefix} と {prefix}_<番号> の列が並んでいます"
        )

    return elements


def read_table_file(path: Path) -> tuple[list[dict], Optional[dict]]:
    """
    save_trajectory が書いた表 (CSV) と, あれば説明 (JSON) を読む関数

    Parameters
    ----------
    path: Path
        trajectories/<名前>.csv

    Returns
    ----------
    result: tuple[list[dict], Optional[dict]]
        見出しの列の名前から値の文字の行と, 説明の中身 (無いか読めない
        場合は None)
    """
    # リンクでない普通のファイルを, 大きさの上限までだけ読む
    raw = read_plain_file(path.parent, path.name, MAX_TRAJECTORY_BYTES)
    if raw is None:
        raise ValueError(
            "軌跡の表を読めません (無いファイル, リンク, 普通でないファイル, "
            f"{MAX_TRAJECTORY_BYTES} バイトを超える表は読みません): {path}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"軌跡の表が UTF-8 ではありません: {path}") from exc
    rows = list(csv.DictReader(io.StringIO(text, newline="")))

    # 説明は補足なので, 読めなければ無いものとして扱う
    info = None
    raw_info = read_plain_file(
        path.parent, f"{path.stem}.json", MAX_TRAJECTORY_INFO_BYTES
    )
    if raw_info is not None:
        try:
            info = json.loads(raw_info.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            info = None
    if not isinstance(info, dict):
        info = None
    result = (rows, info)

    return result


def table_from_trajectory(
    trajectory: Trajectory
) -> tuple[list[dict], dict]:
    """
    記録した軌跡 (メモリの Trajectory) を, 表の行と説明の形にする関数

    Parameters
    ----------
    trajectory: Trajectory
        記録を終えたエピソードを持つ軌跡

    Returns
    ----------
    result: tuple[list[dict], dict]
        save_trajectory が書くのと同じ文字の行と, 説明の types,
        observation_space, complete の項目
    """
    # 書き出しと同じ文字にして, 表と同じ読み方にする
    text = trajectory.header() + "".join(
        trajectory.episode_lines(steps) for steps in trajectory.episodes
    )
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    info = {
        "types": trajectory.column_types(),
        "observation_space": space_info(trajectory.observation_space),
        "complete": trajectory.dropped_episodes == 0
    }
    result = (rows, info)

    return result


def parse_number(text: str, column: str, integer: bool) -> Any:
    """
    表の値の文字を数にする関数

    Parameters
    ----------
    text: str
        値の文字
    column: str
        誤りのメッセージに出す列の名前
    integer: bool
        整数として読むか

    Returns
    ----------
    value: Any
        整数か小数
    """
    try:
        value = int(text) if integer else float(text)
    except ValueError as exc:
        raise ValueError(
            f"軌跡の表の {column} の値を数として読めません: {text!r}"
        ) from exc

    return value


def parse_bool(text: str, column: str) -> bool:
    """
    表の真偽値の文字 (true か false) を読む関数

    Parameters
    ----------
    text: str
        値の文字
    column: str
        誤りのメッセージに出す列の名前

    Returns
    ----------
    value: bool
        真偽値
    """
    if text not in ("true", "false"):
        raise ValueError(
            f"軌跡の表の {column} は true か false にしてください: {text!r}"
        )
    value = text == "true"

    return value


def element_value(
    row: dict, columns: list[str], integer: bool, shape: Optional[list]
) -> Any:
    """
    行の観測か行動の列を, 方策に渡す値にする関数

    Parameters
    ----------
    row: dict
        表の 1 行
    columns: list[str]
        観測か行動の列 (要素の順)
    integer: bool
        番号の空間 (Discrete) の値か
    shape: Optional[list]
        値の形 (説明の空間の shape), 分からない場合は None

    Returns
    ----------
    value: Any
        1 つの値の列は数, 番号の列は形を戻した配列 (形が分からなければ
        平らな配列). 全ての列が空なら None
    """
    # 空の値 (欠け) は行動の無い歩と形の合わない観測だけ
    texts = [row.get(column) or "" for column in columns]
    if all(text == "" for text in texts):
        value = None
        return value
    numbers = [
        parse_number(text, column, integer)
        for text, column in zip(texts, columns)
    ]
    if columns[0] in (OBSERVATION_PREFIX, ACTION_PREFIX):
        value = numbers[0]
        return value
    array = np.asarray(numbers, dtype=np.int64 if integer else np.float32)
    if shape is not None and int(np.prod(shape)) == array.size:
        array = array.reshape(tuple(int(size) for size in shape))
    value = array

    return value


def check_columns(columns: list[str]) -> None:
    """
    見積もりに要る列が表にあることを確かめる関数

    Parameters
    ----------
    columns: list[str]
        表の列

    Returns
    ----------
    None
    """
    # 記録した方策の確率の無い表 (最もスコアの高い行動で選ぶ評価など) は
    # 重みが決まらないので止まる
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"軌跡の表に要る列がありません: {missing}")
    if PROB_COLUMN not in columns:
        raise ValueError(
            f"軌跡の表に記録した方策が行動を選んだ確率の列 {PROB_COLUMN} "
            "がありません. 重点サンプリングには, 確率で行動を選ぶ方策 "
            "(act が action_prob を返す方策など) で記録した軌跡が要ります"
        )


def group_episodes(
    rows: list[dict], info: Optional[dict]
) -> tuple[list[list[dict]], list[str]]:
    """
    表の行を読み, エピソードごとの歩 (観測, 行動, 報酬, 確率) にまとめる
    関数

    Parameters
    ----------
    rows: list[dict]
        見出しの列の名前から値の文字の行
    info: Optional[dict]
        表の説明, 無い場合は None

    Returns
    ----------
    result: tuple[list[list[dict]], list[str]]
        エピソードごとの歩の並びと, 利用者に知らせること
    """
    # 列の確かめと, 観測と行動の列と型 (説明が無ければ値から)
    if not rows:
        raise ValueError("軌跡の表に行がありません")
    columns = list(rows[0].keys())
    check_columns(columns)
    observation_columns = element_columns(columns, OBSERVATION_PREFIX)
    action_columns = element_columns(columns, ACTION_PREFIX)
    info = info or {}
    types = info.get("types") if isinstance(info.get("types"), dict) else {}
    observation_integer = types.get(observation_columns[0]) == "integer"
    action_integer = types.get(action_columns[0]) == "integer" or (
        not types and all(
            (row.get(action_columns[0]) or "0").lstrip("-").isdigit()
            for row in rows
        )
    )
    space = info.get("observation_space")
    shape = space.get("shape") if isinstance(space, dict) else None
    if not isinstance(shape, list):
        shape = None

    # 行をエピソードの番号の続く並びにまとめ, 止めた判断の行は飛ばす
    notes: list[str] = []
    episodes: list[list[dict]] = []
    last = None
    skipped = 0
    for row in rows:
        episode = parse_number(row["episode"], "episode", True)
        if episode != last:
            episodes.append([])
            last = episode
        action = element_value(row, action_columns, action_integer, None)
        guard = row.get(GUARD_COLUMNS[0]) or ""
        if action is None:
            if guard not in STOP_GUARDS:
                raise ValueError(
                    "軌跡の表に, 守りで止めた判断でないのに行動の無い行が"
                    f"あります (episode {row['episode']}, step "
                    f"{row['step']})"
                )
            skipped += 1
            continue
        step = make_step(
            row, action, element_value(
                row, observation_columns, observation_integer, shape
            )
        )
        episodes[-1].append(step)

    # 終わりの印の無いエピソード (利用者の表の途中で切ったもの) を数える
    open_episodes = sum(not ended for ended in episode_ends(rows))
    if skipped:
        notes.append(
            f"守りで止めた判断の行 ({skipped} 行) は, 行動と報酬が無いので"
            "重みと報酬に入れません (そこでエピソードを終えたとします)"
        )
    if open_episodes:
        notes.append(
            f"終了か打ち切りの印の無いエピソードが {open_episodes} ありま"
            "す. 途中までの報酬で見積もります"
        )
    if info.get("complete") is False:
        notes.append(
            "評価で走らせたエピソードの一部が表に入っていません (説明の "
            "complete が false). 表にあるエピソードだけで見積もります"
        )
    result = (episodes, notes)

    return result


def make_step(row: dict, action: Any, observation: Any) -> dict:
    """
    表の 1 行を, 見積もりに使う 1 歩にする関数

    Parameters
    ----------
    row: dict
        表の 1 行 (行動のある行)
    action: Any
        行動
    observation: Any
        行動を選ぶ前の観測

    Returns
    ----------
    step: dict
        observation, action, reward, prob
    """
    # 行動のある行は報酬と確率を持つ
    if not row.get("reward"):
        raise ValueError(
            "軌跡の表に, 行動があるのに報酬の無い行があります "
            f"(episode {row['episode']}, step {row['step']})"
        )
    text = row.get(PROB_COLUMN) or ""
    if text == "":
        raise ValueError(
            f"軌跡の表に, 行動があるのに {PROB_COLUMN} の無い行があります "
            f"(episode {row['episode']}, step {row['step']}). 全ての歩で"
            "確率を返す方策で記録してください"
        )
    prob = parse_number(text, PROB_COLUMN, False)
    if not 0.0 < prob <= 1.0:
        raise ValueError(
            f"軌跡の表の {PROB_COLUMN} は 0 より大きく 1 以下にしてください "
            f"(episode {row['episode']}, step {row['step']}): {text!r}"
        )
    step = {
        "observation": observation, "action": action,
        "reward": parse_number(row["reward"], "reward", False), "prob": prob
    }

    return step


def episode_ends(rows: list[dict]) -> list[bool]:
    """
    エピソードごとに, 最後の行が終わりの印を持つかを返す関数

    Parameters
    ----------
    rows: list[dict]
        表の行

    Returns
    ----------
    ends: list[bool]
        最後の行が terminated か truncated か, 守りで止めた判断なら True
    """
    # エピソードの番号が変わる前の行が, そのエピソードの最後の行
    ends = []
    for index, row in enumerate(rows):
        following = rows[index + 1] if index + 1 < len(rows) else None
        if following is not None and following["episode"] == row["episode"]:
            continue
        ended = parse_bool(row["terminated"], "terminated") or parse_bool(
            row["truncated"], "truncated"
        ) or (row.get(GUARD_COLUMNS[0]) or "") in STOP_GUARDS
        ends.append(ended)

    return ends


def target_prob_of(
    target_prob: Callable, observation: Any, action: Any
) -> float:
    """
    見積もる方策がその観測でその行動を選ぶ確率を, 確かめて返す関数

    Parameters
    ----------
    target_prob: Callable
        target_prob(観測, 行動) が確率を返す関数
    observation: Any
        観測
    action: Any
        記録した方策がとった行動

    Returns
    ----------
    prob: float
        0 以上 1 以下の確率
    """
    # 数でない値と範囲の外の値は止まる (重みを読み違えないため)
    value = target_prob(observation, action)
    try:
        prob = float(np.asarray(value, dtype=np.float64).item())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"target_prob は 0 以上 1 以下の数を返してください: {value!r}"
        ) from exc
    if not 0.0 <= prob <= 1.0:
        raise ValueError(
            f"target_prob は 0 以上 1 以下の数を返してください: {value!r}"
        )

    return prob


def episode_weights(
    steps: list[dict], target_prob: Callable, gamma: float
) -> dict[str, Any]:
    """
    1 つのエピソードの重み (確率の比の積) と割り引いた報酬の和を計算する
    関数

    Parameters
    ----------
    steps: list[dict]
        エピソードの歩
    target_prob: Callable
        見積もる方策の確率の関数
    gamma: float
        割引率

    Returns
    ----------
    result: dict[str, Any]
        weight (エピソード全体の重み), return (割り引いた報酬の和),
        per_decision (歩ごとの重みを掛けた割り引いた報酬の和),
        zero_steps (見積もる方策の確率が 0 の歩の数)
    """
    # 歩ごとに, そこまでの比の積を報酬に掛けて足す
    weight = 1.0
    total = 0.0
    per_decision = 0.0
    zero_steps = 0
    discount = 1.0
    for step in steps:
        prob = target_prob_of(target_prob, step["observation"], step["action"])
        if prob == 0.0:
            zero_steps += 1
        weight *= prob / step["prob"]
        total += discount * step["reward"]
        per_decision += discount * weight * step["reward"]
        discount *= gamma
    result = {
        "weight": weight, "return": total, "per_decision": per_decision,
        "zero_steps": zero_steps
    }

    return result


def summarize_weights(
    results: list[dict], notes: list[str]
) -> dict[str, Any]:
    """
    エピソードごとの重みから, 3 つの見積もりと重みのばらつきの印を
    まとめる関数

    Parameters
    ----------
    results: list[dict]
        episode_weights の結果の並び
    notes: list[str]
        利用者に知らせることを足す並び

    Returns
    ----------
    summary: dict[str, Any]
        value, ordinary, per_decision, behavior_value,
        effective_sample_size, max_weight_share, zero_weight_episodes
    """
    # 重み付きの見積もりは重みの和で割る (全ての重みが 0 なら決まらない)
    weights = np.array([row["weight"] for row in results], dtype=np.float64)
    returns = np.array([row["return"] for row in results], dtype=np.float64)
    count = len(results)
    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "重みが大きすぎて計算できません (記録した方策の確率が非常に"
            "小さい歩が続いています)"
        )
    total = float(weights.sum())
    value = float((weights * returns).sum() / total) if total > 0 else None

    # 有効なサンプルの数 (Kish) と, 最も大きい重みの割合
    if total > 0:
        scaled = weights / weights.max()
        ess = float(scaled.sum() ** 2 / (scaled ** 2).sum())
        share = float(weights.max() / total)
    else:
        ess = 0.0
        share = None
    zero_episodes = int(np.sum(weights == 0.0))
    zero_steps = sum(row["zero_steps"] for row in results)
    if zero_steps:
        notes.append(
            f"見積もる方策が選ばない行動 (確率 0) を記録した方策が "
            f"{zero_steps} 歩でとりました. その {zero_episodes} エピソード"
            "の重みは 0 です"
        )
    if value is None:
        notes.append(
            "全てのエピソードの重みが 0 なので, 重み付きの見積もり (value)"
            " は決まりません. 見積もる方策に近い方策で記録してください"
        )
    elif ess < LOW_ESS_RATIO * count:
        notes.append(
            f"有効なサンプルの数 {ess:.1f} がエピソードの数 {count} の "
            f"{LOW_ESS_RATIO:.0%} より少なく, 重みのばらつきが大きいので"
            "見積もりは当てになりません"
        )
    summary = {
        "value": value,
        "ordinary": float((weights * returns).mean()),
        "per_decision": float(
            np.mean([row["per_decision"] for row in results])
        ),
        "behavior_value": float(returns.mean()),
        "effective_sample_size": ess,
        "max_weight_share": share,
        "zero_weight_episodes": zero_episodes
    }

    return summary


def estimate_policy_value(
    trajectory: Union[Trajectory, str, Path], target_prob: Callable,
    gamma: float = 1.0
) -> dict[str, Any]:
    """
    記録した軌跡から, 環境で動かさずに方策の良さ (エピソードの割り引いた
    報酬の和の期待値) を重点サンプリングで見積もる関数

    Parameters
    ----------
    trajectory: Union[Trajectory, str, Path]
        save_trajectory が書いた表 (trajectories/<名前>.csv. 隣の説明も
        読む) の場所か, 記録した軌跡 (Trajectory). 記録した方策が行動を
        選んだ確率の列 action_prob が要る
    target_prob: Callable
        target_prob(観測, 行動) が, 見積もる方策がその観測でその行動を
        選ぶ確率 (0 以上 1 以下) を返す関数. 観測と行動は, 1 つの値の列は
        数 (番号の空間は整数), 番号の列は配列 (観測は空間の形)
    gamma: float = 1.0
        割引率 (0 以上 1 以下)

    Returns
    ----------
    result: dict[str, Any]
        value (重み付きの重点サンプリングの見積もり. 決まらなければ None),
        ordinary (普通の重点サンプリング), per_decision (歩ごとの重点
        サンプリング), behavior_value (記録した方策の報酬の和の平均),
        episodes, steps, effective_sample_size (有効なサンプルの数),
        max_weight_share (最も大きい重みの割合), zero_weight_episodes
        (重みが 0 のエピソードの数), notes (知らせること)
    """
    # 割引率と関数の確認
    if isinstance(gamma, bool) or not isinstance(
        gamma, (int, float)
    ) or not 0.0 <= float(gamma) <= 1.0 or math.isnan(float(gamma)):
        raise ValueError(f"gamma は 0 以上 1 以下にしてください: {gamma!r}")
    if not callable(target_prob):
        raise ValueError(
            "target_prob は (観測, 行動) から確率を返す関数にしてください: "
            f"{target_prob!r}"
        )

    # 表か記録した軌跡を, エピソードごとの歩にする
    if isinstance(trajectory, Trajectory):
        rows, info = table_from_trajectory(trajectory)
    elif isinstance(trajectory, (str, Path)):
        rows, info = read_table_file(Path(trajectory))
    else:
        raise ValueError(
            "trajectory は Trajectory か軌跡の表 (CSV) の場所にしてくださ"
            f"い: {trajectory!r}"
        )
    episodes, notes = group_episodes(rows, info)

    # エピソードごとの重みと, 見積もりのまとめ
    results = [
        episode_weights(steps, target_prob, float(gamma))
        for steps in episodes
    ]
    summary = summarize_weights(results, notes)
    result = {
        **summary,
        "episodes": len(episodes),
        "steps": sum(len(steps) for steps in episodes),
        "notes": notes
    }

    return result
