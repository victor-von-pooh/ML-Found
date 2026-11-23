import csv
import io
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ml_found.experiment.results import (
    check_name, check_optional_text, new_temp, run_directory,
    write_json_file
)
from ml_found.rl.bundle import space_to_dict
from ml_found.training.status import now_iso

# 軌跡の表を置くディレクトリの名前
TRAJECTORIES_DIR = "trajectories"

# 軌跡の表の説明 (trajectories/<名前>.json) の形式の版
TRAJECTORY_FORMAT_VERSION = 1

# 軌跡の表 (trajectories/<名前>.csv) の大きさの上限 (公開. 読む側も使う).
# 書くときはこの大きさに収まるエピソードまでを書く
MAX_TRAJECTORY_BYTES = 64 * 1024 * 1024

# 軌跡の表の説明の大きさの上限 (読む側が使う)
MAX_TRAJECTORY_INFO_BYTES = 1024 * 1024

# 1 つの軌跡が貯める歩の数の既定の上限 (メモリを使い切らないため)
DEFAULT_MAX_ROWS = 100_000

# 観測と行動の要素の数の上限 (列の数. 画像のような観測は表にしない)
MAX_ELEMENTS = 256

# 観測と行動の列の名前の前の部分
OBSERVATION_PREFIX = "obs"
ACTION_PREFIX = "action"

# 守りの結果の列 (守りを当てた評価だけ)
GUARD_COLUMNS = ("guard", "guard_reasons")

# 守りの理由をつなぐ文字
REASON_SEPARATOR = "; "

# 数として書く NumPy の配列の種類 (真偽値, 整数, 小数)
NUMERIC_KINDS = "biuf"


def check_max_rows(max_rows: Any, where: str) -> None:
    """
    軌跡が貯める歩の数の上限が 1 以上の整数であることを確かめる関数

    Parameters
    ----------
    max_rows: Any
        歩の数の上限
    where: str
        誤りのメッセージに出す引数の名前

    Returns
    ----------
    None
    """
    # bool は int の子なので除外
    if (
        isinstance(max_rows, bool) or not isinstance(max_rows, int)
        or max_rows < 1
    ):
        raise ValueError(
            f"{where} は 1 以上の整数にしてください: {max_rows!r}"
        )


def env_id(env: Any) -> Optional[str]:
    """
    環境の名前 (Gymnasium の登録の名前) を取得する関数

    Parameters
    ----------
    env: Any
        Gymnasium の形の環境

    Returns
    ----------
    name: Optional[str]
        spec.id (CartPole-v1 など), 無い場合は None
    """
    # make で作った環境だけが spec を持つ
    name = getattr(getattr(env, "spec", None), "id", None)
    if not isinstance(name, str):
        name = None

    return name


def space_size(space: Any) -> int:
    """
    空間から, 表の列にする要素の数を決める関数

    Parameters
    ----------
    space: Any
        Gymnasium の形の空間 (Discrete か Box), 分からない場合は None

    Returns
    ----------
    size: int
        Discrete と形が () の Box は 0 (1 つの列), 他の Box は要素の数,
        決められない場合は -1
    """
    # 形が () のもの (Discrete など) は 1 つの列, 形があれば要素の数
    shape = getattr(space, "shape", None)
    if space is None or shape is None:
        size = -1
        return size
    try:
        size = int(np.prod(tuple(shape), dtype=np.int64)) if shape else 0
    except (TypeError, ValueError):
        size = -1

    return size


def value_size(value: Any) -> int:
    """
    値から, 表の列にする要素の数を決める関数

    Parameters
    ----------
    value: Any
        観測か行動

    Returns
    ----------
    size: int
        0 次元の値は 0 (1 つの列), 配列は要素の数
    """
    # 0 次元は 1 つの列, 配列は平らにした要素の数
    array = np.asarray(value)
    size = 0 if array.ndim == 0 else int(array.size)

    return size


def column_names(prefix: str, size: int) -> list[str]:
    """
    観測か行動の列の名前を作る関数

    Parameters
    ----------
    prefix: str
        列の名前の前の部分 (obs か action)
    size: int
        要素の数 (0 は 1 つの列)

    Returns
    ----------
    names: list[str]
        0 なら [prefix], 他は prefix_0, prefix_1, ...
    """
    # 要素の数の上限の確認 (画像のような観測で列が増えすぎないため)
    if size > MAX_ELEMENTS:
        raise ValueError(
            f"{prefix} の要素の数 {size} が軌跡の表の上限 {MAX_ELEMENTS} を"
            "超えます. 観測を小さくする Wrapper を使うか, 軌跡を記録しない"
            "でください"
        )
    names = [prefix] if size == 0 else [f"{prefix}_{i}" for i in range(size)]

    return names


def format_number(value: Any) -> str:
    """
    数を CSV の値にする関数

    Parameters
    ----------
    value: Any
        数 (Python か NumPy の数)

    Returns
    ----------
    text: str
        整数はそのまま, 小数は型の最短の書き方, nan は空, 無限大は
        inf か -inf, 真偽値は 0 か 1
    """
    # 真偽値と整数はそのまま, 小数は NumPy の型の最短の書き方
    if isinstance(value, (bool, np.bool_)):
        text = str(int(value))
        return text
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
        return text
    number = float(value)
    if math.isnan(number):
        text = ""
    elif math.isinf(number):
        text = "inf" if number > 0 else "-inf"
    elif isinstance(value, np.floating):
        text = str(value)
    else:
        text = repr(number)

    return text


def flat_values(value: Any, size: int) -> Optional[list[str]]:
    """
    観測か行動を, 列ごとの CSV の値にする関数

    Parameters
    ----------
    value: Any
        観測か行動, 無い場合は None
    size: int
        列の要素の数 (0 は 1 つの列)

    Returns
    ----------
    values: Optional[list[str]]
        列ごとの値. 形が合わないか, 数でない場合は None (空の値にする)
    """
    # 行動の無い歩 (守りで止めた判断) と, 形の合わない値は空
    if value is None:
        values = None
        return values
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        values = None
        return values
    if array.dtype.kind not in NUMERIC_KINDS or value_size(array) != size:
        values = None
        return values
    values = [format_number(item) for item in array.reshape(-1)]

    return values


def is_prefixed(column: str, prefix: str) -> bool:
    """
    列が観測か行動の列 (prefix か prefix_<番号>) かを判定する関数

    Parameters
    ----------
    column: str
        列の名前
    prefix: str
        obs か action

    Returns
    ----------
    matched: bool
        prefix そのものか, prefix_ の後が数字だけの場合は True
    """
    rest = column[len(prefix) + 1:]
    matched = column == prefix or (
        column.startswith(f"{prefix}_") and rest.isdigit()
    )

    return matched


def format_bool(value: bool) -> str:
    """
    真偽値を CSV の値にする関数

    Parameters
    ----------
    value: bool
        真偽値

    Returns
    ----------
    text: str
        true か false
    """
    text = "true" if value else "false"

    return text


class Trajectory:
    def __init__(
        self, max_rows: int = DEFAULT_MAX_ROWS,
        observation_space: Any = None, action_space: Any = None
    ):
        # 貯める歩の数の上限の確認
        check_max_rows(max_rows, "max_rows")

        # 歩の数の上限と, 観測と行動の空間を保存
        self.max_rows = max_rows
        self.observation_space = observation_space
        self.action_space = action_space

        # 何から記録したか, 環境の名前 (記録する側が埋める)
        self.source: Optional[str] = None
        self.env_id: Optional[str] = None

        # 観測と行動の列の要素の数 (最初の歩か空間から決める)
        self.observation_size: Optional[int] = None
        self.action_size: Optional[int] = None

        # 終えたエピソードの歩とその数, 途中のエピソード, 上限に達したか,
        # 入れなかったエピソードの数, 守りの結果があるか
        self.episodes: list[list[dict]] = []
        self.n_rows = 0
        self.current: Optional[list[dict]] = None
        self.current_episode = 0
        self.current_seed: Optional[int] = None
        self.full = False
        self.dropped_episodes = 0
        self.guarded = False

    def set_spaces(self, observation_space: Any, action_space: Any) -> None:
        """
        観測と行動の空間を, まだ無い場合だけ保存する関数

        Parameters
        ----------
        observation_space: Any
            観測の空間
        action_space: Any
            行動の空間

        Returns
        ----------
        None
        """
        if self.observation_space is None:
            self.observation_space = observation_space
        if self.action_space is None:
            self.action_space = action_space

    def check_spaces(self) -> None:
        """
        空間の要素の数が表の列の上限に収まることを, 記録の前に確かめる関数

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 空間から決められない場合は, 最初の値で確かめる
        for prefix, space in (
            (OBSERVATION_PREFIX, self.observation_space),
            (ACTION_PREFIX, self.action_space)
        ):
            size = space_size(space)
            if size >= 0:
                column_names(prefix, size)

    def start_episode(self, episode: int, seed: Optional[int]) -> None:
        """
        エピソードの記録を始める関数

        Parameters
        ----------
        episode: int
            エピソードの番号 (0 から)
        seed: Optional[int]
            エピソードを始めた乱数の種, 環境の乱数の続きなら None

        Returns
        ----------
        None
        """
        # 前のエピソードを終えずに始めた場合は, 前のものを捨てる
        self.current = []
        self.current_episode = int(episode)
        self.current_seed = None if seed is None else int(seed)

    def resolve_sizes(self, observation: Any, action: Any) -> None:
        """
        観測と行動の列の要素の数を, 空間か最初の値から決める関数

        Parameters
        ----------
        observation: Any
            最初の観測
        action: Any
            最初の行動 (守りで止めた判断では None)

        Returns
        ----------
        None
        """
        # 空間から決められなければ, 最初の値の形から
        # (上限を超える場合は決めずに止まる)
        if self.observation_size is None:
            size = space_size(self.observation_space)
            if size < 0:
                size = value_size(observation)
            column_names(OBSERVATION_PREFIX, size)
            self.observation_size = size
        if self.action_size is None:
            size = space_size(self.action_space)
            if size < 0 and action is not None:
                size = value_size(action)
            if size >= 0:
                column_names(ACTION_PREFIX, size)
                self.action_size = size

    def add(
        self, observation: Any, action: Any, reward: Optional[float],
        terminated: bool, truncated: bool, guard: Optional[str] = None,
        reasons: Optional[list] = None
    ) -> None:
        """
        1 歩 (か守りで止めた判断) を記録する関数

        Parameters
        ----------
        observation: Any
            行動を選ぶ前の観測
        action: Any
            環境に渡した行動, 守りで止めた判断では None
        reward: Optional[float]
            行動の後の報酬, 環境を進めていない場合は None
        terminated: bool
            環境が終了を知らせたか
        truncated: bool
            打ち切り (環境か歩数の上限) で終えたか
        guard: Optional[str] = None
            守りの結果 (ok, clipped, fallback, stop, handover)
        reasons: Optional[list] = None
            守りが働いた理由

        Returns
        ----------
        None
        """
        # 始めていない場合は次の番号から. 上限に達した後のエピソードは
        # 先頭から続く表にするため全て捨てる
        if self.current is None:
            self.start_episode(len(self.episodes), None)
        self.resolve_sizes(observation, action)
        if self.full:
            return
        if self.n_rows + len(self.current) >= self.max_rows:
            self.full = True
            self.current = []
            return

        # 値はその場で CSV の文字にし, 観測を後で書き換えられても変わらない形.
        # 行動があれば resolve_sizes が要素の数を決めている
        if guard is not None:
            self.guarded = True
        text_reasons = REASON_SEPARATOR.join(str(r) for r in reasons or [])
        self.current.append({
            "observation": flat_values(observation, self.observation_size),
            "action": None if action is None else flat_values(
                action, self.action_size
            ),
            "reward": None if reward is None else float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "guard": guard,
            "guard_reasons": text_reasons
        })

    def end_episode(self) -> None:
        """
        エピソードの記録を終え, 上限に収まれば貯める関数

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 上限に達したエピソードは途中までを残さずに捨てる
        if self.current is None:
            return
        if self.full:
            self.dropped_episodes += 1
        else:
            for index, row in enumerate(self.current):
                row["episode"] = self.current_episode
                row["step"] = index
                row["seed"] = self.current_seed
            self.episodes.append(self.current)
            self.n_rows += len(self.current)
        self.current = None

    def columns(self) -> list[str]:
        """
        表の列の名前を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        columns: list[str]
            episode, step, 観測, 行動, reward, terminated, truncated, seed,
            守りを当てた場合は guard と guard_reasons
        """
        # 行動の無い軌跡 (全て止めた判断) は 1 つの列
        observation = column_names(
            OBSERVATION_PREFIX, self.observation_size or 0
        )
        action = column_names(ACTION_PREFIX, self.action_size or 0)
        columns = [
            "episode", "step", *observation, *action, "reward",
            "terminated", "truncated", "seed"
        ]
        if self.guarded:
            columns.extend(GUARD_COLUMNS)

        return columns

    def column_types(self) -> dict[str, str]:
        """
        列ごとの値の型を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        types: dict[str, str]
            列の名前から integer, number, boolean, string
        """
        # 観測と行動は, 番号の空間 (Discrete) なら整数, 他は小数
        observation_type = "integer" if is_discrete(
            self.observation_space
        ) else "number"
        action_type = "integer" if is_discrete(self.action_space) else (
            "number"
        )
        types = {}
        for column in self.columns():
            if column in ("episode", "step", "seed"):
                types[column] = "integer"
            elif is_prefixed(column, OBSERVATION_PREFIX):
                types[column] = observation_type
            elif is_prefixed(column, ACTION_PREFIX):
                types[column] = action_type
            elif column == "reward":
                types[column] = "number"
            elif column in ("terminated", "truncated"):
                types[column] = "boolean"
            else:
                types[column] = "string"

        return types

    def episode_lines(self, steps: list[dict]) -> str:
        """
        1 つのエピソードの歩を CSV の行にする関数

        Parameters
        ----------
        steps: list[dict]
            エピソードの歩

        Returns
        ----------
        text: str
            CSV の行 (見出しを除く)
        """
        # 形の合わない観測と, 行動の無い歩は空の値
        empty_observation = [""] * max(self.observation_size or 0, 1)
        action_size = self.action_size or 0
        empty_action = [""] * max(action_size, 1)
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        for row in steps:
            action = row["action"]
            line = [
                str(row["episode"]), str(row["step"]),
                *(row["observation"] or empty_observation),
                *(action or empty_action),
                "" if row["reward"] is None else format_number(row["reward"]),
                format_bool(row["terminated"]),
                format_bool(row["truncated"]),
                "" if row["seed"] is None else str(row["seed"])
            ]
            if self.guarded:
                line.extend([row["guard"] or "", row["guard_reasons"]])
            writer.writerow(line)
        text = buffer.getvalue()

        return text


def is_discrete(space: Any) -> bool:
    """
    空間が番号で選ぶ形 (Discrete) かを判定する関数

    Parameters
    ----------
    space: Any
        空間

    Returns
    ----------
    discrete: bool
        n を持ち形が () の場合は True
    """
    shape = tuple(getattr(space, "shape", ()) or ())
    discrete = isinstance(getattr(space, "n", None), (int, np.integer)) and (
        shape == ()
    )

    return discrete


def space_info(space: Any) -> Optional[dict]:
    """
    空間を説明に書く辞書にする関数

    Parameters
    ----------
    space: Any
        空間

    Returns
    ----------
    info: Optional[dict]
        space_to_dict の形 (Discrete か Box), 書けない空間は None
    """
    # Discrete と Box の他 (Dict など) と, 分からない空間は書かない
    if space is None:
        info = None
        return info
    try:
        info = space_to_dict(space)
    except (TypeError, ValueError):
        info = None

    return info


def write_csv(path: Path, header: str, chunks: list[str]) -> None:
    """
    読み手が書きかけを読まないように CSV を書く関数

    Parameters
    ----------
    path: Path
        書くファイル
    header: str
        見出しの行
    chunks: list[str]
        エピソードごとの行

    Returns
    ----------
    None
    """
    # 一時ファイルに書いてから置き換え, 失敗した場合は一時ファイルの削除
    temp = new_temp(path)
    try:
        with open(temp, "w", encoding="utf-8", newline="") as file:
            file.write(header)
            for chunk in chunks:
                file.write(chunk)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_trajectory(
    target: Any, name: str, trajectory: Trajectory,
    epoch: Optional[int] = None, description: Optional[str] = None
) -> Path:
    """
    評価の軌跡を trajectories/<名前>.csv (1 歩が 1 行の表) と説明の
    trajectories/<名前>.json に書く関数

    Parameters
    ----------
    target: Any
        start_experiment が返した実験か, 実験のディレクトリのパス
    name: str
        軌跡の名前 (英数字と _ . @ - だけ. 先頭は . 以外)
    trajectory: Trajectory
        EvaluationLoader の trajectory か, evaluate_policy に渡した軌跡
    epoch: Optional[int] = None
        評価した重みのエポック
    description: Optional[str] = None
        何の軌跡か

    Returns
    ----------
    path: Path
        書いた表のファイル
    """
    # 実験, 名前, 軌跡, エポック, 説明の確認
    run_dir = run_directory(target)
    check_name(name, "軌跡")
    if not isinstance(trajectory, Trajectory):
        raise ValueError(
            "trajectory には Trajectory (EvaluationLoader の trajectory か, "
            f"evaluate_policy に渡した軌跡) を渡してください: {trajectory!r}"
        )
    if epoch is not None and (
        not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0
    ):
        raise ValueError(f"epoch は 0 以上の整数にしてください: {epoch!r}")
    check_optional_text(description, "description")

    # 大きさの上限に収まるエピソードまでを, エピソードの単位で
    columns = trajectory.columns()
    header = ",".join(columns) + "\n"
    size = len(header.encode("utf-8"))
    chunks = []
    rows = 0
    for steps in trajectory.episodes:
        chunk = trajectory.episode_lines(steps)
        size += len(chunk.encode("utf-8"))
        if size > MAX_TRAJECTORY_BYTES:
            break
        chunks.append(chunk)
        rows += len(steps)
    dropped = trajectory.dropped_episodes + (
        len(trajectory.episodes) - len(chunks)
    )

    # 実験のディレクトリの外に書かないように, リンクの置き場には書かない
    folder = run_dir / TRAJECTORIES_DIR
    if folder.is_symlink():
        raise ValueError(f"軌跡の置き場がリンクです: {folder}")
    folder.mkdir(exist_ok=True)
    path = folder / f"{name}.csv"
    write_csv(path, header, chunks)
    info = {
        "format_version": TRAJECTORY_FORMAT_VERSION,
        "name": name,
        "file": path.name,
        "source": trajectory.source,
        "env": trajectory.env_id,
        "epoch": epoch,
        "description": description,
        "columns": columns,
        "types": trajectory.column_types(),
        "observation_columns": [
            column for column in columns
            if is_prefixed(column, OBSERVATION_PREFIX)
        ],
        "action_columns": [
            column for column in columns
            if is_prefixed(column, ACTION_PREFIX)
        ],
        "observation_space": space_info(trajectory.observation_space),
        "action_space": space_info(trajectory.action_space),
        "guarded": trajectory.guarded,
        "episodes": len(chunks),
        "rows": rows,
        "complete": dropped == 0,
        "dropped_episodes": dropped,
        "created_at": now_iso()
    }
    write_json_file(folder / f"{name}.json", info)

    return path
