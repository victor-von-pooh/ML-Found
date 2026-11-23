import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch

from ml_found.rl import (
    TRAJECTORY_FORMAT_VERSION, EvaluationLoader, ReinforceStep,
    RolloutLoader, Trajectory, evaluate_policy, save_trajectory
)
from ml_found.training.trainer import Trainer
from ml_found.rl import trajectory as trajectory_module
from ml_found.rl.trajectory import MAX_ELEMENTS
from test_format import documented_keys
from test_policy_eval import RightModel, guarded_policy
from test_rl import Space, ToyEnv, toy_model
from test_rl_box import BoxEnv, box_model
from test_trainer import make_experiment


def read_table(path: Path) -> list[dict]:
    """
    軌跡の表を CSV の行の辞書として読む関数

    Parameters
    ----------
    path: Path
        trajectories/<名前>.csv

    Returns
    ----------
    rows: list[dict]
        見出しの列の名前から値の文字
    """
    with open(path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    return rows


def read_info(run_dir: Path, name: str) -> dict:
    """
    軌跡の表の説明を読む関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ
    name: str
        軌跡の名前

    Returns
    ----------
    info: dict
        trajectories/<名前>.json の中身
    """
    text = (run_dir / "trajectories" / f"{name}.json").read_text(
        encoding="utf-8"
    )
    info = json.loads(text)

    return info


class WideEnv(ToyEnv):
    def __init__(self, size: int):
        # 親クラスのコンストラクタを呼び出す
        super(WideEnv, self).__init__()

        # 要素の多い観測の空間を保存
        self.size = size
        self.observation_space = Space(
            low=[-1.0] * size, high=[1.0] * size
        )

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        """
        要素の多い観測でエピソードを始める関数

        Parameters
        ----------
        seed: Optional[int] = None
            乱数の種

        Returns
        ----------
        observation: np.ndarray
            0 の並び
        info: dict
            空の辞書
        """
        super(WideEnv, self).reset(seed)
        observation = np.zeros(self.size, dtype=np.float32)

        return observation, {}


def test_loader_does_not_record_by_default() -> None:
    """
    EvaluationLoader が既定では軌跡を記録しないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    loader = EvaluationLoader(ToyEnv(), RightModel(), episodes=2, seed=3)
    next(iter(loader))
    assert loader.trajectory is None


def test_loader_records_and_saves(tmp_path: Path) -> None:
    """
    EvaluationLoader の軌跡が, FORMAT.md の列と説明の形で書けることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 5 歩で終了する環境を 2 エピソード, 行動はいつも 1 で報酬 1
    loader = EvaluationLoader(
        ToyEnv(), RightModel(), episodes=2, seed=3, record_trajectory=True
    )
    next(iter(loader))
    path = save_trajectory(
        tmp_path, "valid", loader.trajectory, epoch=4, description="検証"
    )
    assert path == tmp_path / "trajectories" / "valid.csv"

    # 見出しと 1 歩 1 行の中身. 観測は行動の前 (0, 1, ..., 4)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "episode,step,obs_0,action,reward,terminated,truncated,seed"
    )
    rows = read_table(path)
    assert len(rows) == 10
    assert [row["episode"] for row in rows] == ["0"] * 5 + ["1"] * 5
    assert [row["step"] for row in rows[:5]] == ["0", "1", "2", "3", "4"]
    assert [row["obs_0"] for row in rows[:5]] == [
        "0.0", "1.0", "2.0", "3.0", "4.0"
    ]
    assert {row["action"] for row in rows} == {"1"}
    assert {row["reward"] for row in rows} == {"1.0"}
    assert [row["terminated"] for row in rows[:5]] == (
        ["false"] * 4 + ["true"]
    )
    assert {row["truncated"] for row in rows} == {"false"}
    assert [row["seed"] for row in (rows[0], rows[5])] == ["3", "4"]

    # 説明の項目は FORMAT.md の表と同じ
    info = read_info(tmp_path, "valid")
    assert set(info) == documented_keys("#### trajectories/<名前>.json")
    assert info["format_version"] == TRAJECTORY_FORMAT_VERSION
    assert info["source"] == "evaluation_loader"
    assert info["epoch"] == 4 and info["description"] == "検証"
    assert info["observation_columns"] == ["obs_0"]
    assert info["action_columns"] == ["action"]
    assert info["types"]["obs_0"] == "number"
    assert info["types"]["action"] == "integer"
    assert info["types"]["terminated"] == "boolean"
    assert info["observation_space"]["type"] == "box"
    assert info["action_space"] == {"type": "discrete", "n": 2, "start": 0}
    assert info["guarded"] is False
    assert info["episodes"] == 2 and info["rows"] == 10
    assert info["complete"] is True and info["dropped_episodes"] == 0


def test_csv_columns_match_format() -> None:
    """
    表の列の決まり (FORMAT.md の表) と, 軌跡の列の名前が合うことを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    keys = documented_keys("### trajectories")
    assert keys == {
        "episode", "step", "obs", "obs_<番号>", "action", "action_<番号>",
        "reward", "terminated", "truncated", "seed", "guard",
        "guard_reasons"
    }


def test_loader_steps_limit_is_truncation(tmp_path: Path) -> None:
    """
    max_episode_steps で切った最後の歩が打ち切りになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    loader = EvaluationLoader(
        ToyEnv(length=10), RightModel(), episodes=1, seed=0,
        max_episode_steps=3, record_trajectory=True
    )
    next(iter(loader))
    rows = read_table(save_trajectory(tmp_path, "cut", loader.trajectory))
    assert [row["truncated"] for row in rows] == ["false", "false", "true"]
    assert {row["terminated"] for row in rows} == {"false"}


def test_loader_replaces_trajectory_each_evaluation() -> None:
    """
    評価のたびに軌跡を置き換える (貯め続けない) ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    loader = EvaluationLoader(
        ToyEnv(), RightModel(), episodes=2, record_trajectory=True
    )
    next(iter(loader))
    first = loader.trajectory
    next(iter(loader))
    assert loader.trajectory is not first
    assert loader.trajectory.n_rows == 10


def test_box_actions_are_split_and_clipped(tmp_path: Path) -> None:
    """
    連続の行動が要素ごとの列に分かれ, clip した値で書かれることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 平均を範囲の外 (3.0) にして, 環境に渡す行動は上限の 1.0
    model = box_model()
    with torch.no_grad():
        model.mean.weight.zero_()
        model.mean.bias.fill_(3.0)
    loader = EvaluationLoader(
        BoxEnv(length=2), model, episodes=1, record_trajectory=True
    )
    next(iter(loader))
    save_trajectory(tmp_path, "box", loader.trajectory)
    rows = read_table(tmp_path / "trajectories" / "box.csv")
    assert [row["action_0"] for row in rows] == ["1.0", "1.0"]
    info = read_info(tmp_path, "box")
    assert info["action_columns"] == ["action_0"]
    assert info["types"]["action_0"] == "number"


def test_guarded_trajectory_keeps_stop(tmp_path: Path) -> None:
    """
    守り付きの評価の軌跡が, 守りの結果と止めた判断の行を持つことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 観測は 0, 1, 2, ... なので, 3 を見たところで止める
    policy = guarded_policy(tmp_path, 2.5, "stop")
    trajectory = Trajectory()
    rows = evaluate_policy(
        policy, ToyEnv(), episodes=2, seed=7, trajectory=trajectory
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = save_trajectory(run_dir, "guarded", trajectory, epoch=1)
    table = read_table(path)

    # 1 エピソードは 3 歩と止めた判断の 1 行
    assert len(table) == 8
    first = table[:4]
    assert [row["guard"] for row in first] == ["ok", "ok", "ok", "stop"]
    assert first[3]["obs_0"] == "3.0"
    assert first[3]["action"] == "" and first[3]["reward"] == ""
    assert first[3]["terminated"] == "false"
    assert first[3]["truncated"] == "false"
    assert first[3]["guard_reasons"] != ""
    assert first[0]["guard_reasons"] == ""

    # 報酬の和と歩数は evaluate_policy の行と同じ
    rewards = [float(row["reward"]) for row in first if row["reward"]]
    assert sum(rewards) == rows[0]["reward"]
    assert len(rewards) == rows[0]["length"]
    info = read_info(run_dir, "guarded")
    assert info["guarded"] is True and info["source"] == "evaluate_policy"
    assert info["columns"][-2:] == ["guard", "guard_reasons"]
    assert info["types"]["guard"] == "string"


def test_evaluate_policy_max_steps_is_truncation(tmp_path: Path) -> None:
    """
    evaluate_policy の max_steps で切った歩が打ち切りになることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    policy = guarded_policy(tmp_path, 100.0, "stop")
    trajectory = Trajectory()
    evaluate_policy(
        policy, ToyEnv(length=10), episodes=1, max_steps=2,
        trajectory=trajectory
    )
    path = save_trajectory(tmp_path, "cut", trajectory)
    table = read_table(path)
    assert [row["truncated"] for row in table] == ["false", "true"]
    assert table[0]["seed"] == ""


def test_evaluate_policy_needs_empty_trajectory(tmp_path: Path) -> None:
    """
    evaluate_policy が記録済みの軌跡と Trajectory でない値を断ることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    policy = guarded_policy(tmp_path, 100.0, "stop")
    trajectory = Trajectory()
    evaluate_policy(policy, ToyEnv(), episodes=1, trajectory=trajectory)
    with pytest.raises(ValueError, match="何も記録していない"):
        evaluate_policy(policy, ToyEnv(), episodes=1, trajectory=trajectory)
    with pytest.raises(ValueError, match="Trajectory か None"):
        evaluate_policy(policy, ToyEnv(), episodes=1, trajectory=[])


def test_rows_limit_keeps_whole_episodes(tmp_path: Path) -> None:
    """
    歩の数の上限で, 先頭から収まるエピソードだけを残すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 5 歩のエピソードを 3 つ, 上限 12 歩なので 2 つだけ入る
    loader = EvaluationLoader(
        ToyEnv(), RightModel(), episodes=3, record_trajectory=True,
        max_trajectory_rows=12
    )
    next(iter(loader))
    save_trajectory(tmp_path, "limited", loader.trajectory)
    table = read_table(tmp_path / "trajectories" / "limited.csv")
    assert {row["episode"] for row in table} == {"0", "1"}
    info = read_info(tmp_path, "limited")
    assert info["episodes"] == 2 and info["rows"] == 10
    assert info["complete"] is False and info["dropped_episodes"] == 1


def test_bytes_limit_keeps_whole_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    表の大きさの上限で, 先頭から収まるエピソードだけを書くことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        上限を小さくする

    Returns
    ----------
    None
    """
    loader = EvaluationLoader(
        ToyEnv(), RightModel(), episodes=3, record_trajectory=True
    )
    next(iter(loader))
    full = save_trajectory(tmp_path, "full", loader.trajectory)
    lines = full.read_text(encoding="utf-8").splitlines(keepends=True)
    limit = len("".join(lines[:11]).encode("utf-8"))
    monkeypatch.setattr(trajectory_module, "MAX_TRAJECTORY_BYTES", limit)
    path = save_trajectory(tmp_path, "small", loader.trajectory)
    assert path.stat().st_size == limit
    info = read_info(tmp_path, "small")
    assert info["episodes"] == 2 and info["dropped_episodes"] == 1
    assert info["complete"] is False


def test_values_and_shapes() -> None:
    """
    値の書き方 (nan と無限大, 1 つの値の列) と形の誤りを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 空間の分からない軌跡は, 最初の値の形から列を決める
    trajectory = Trajectory()
    trajectory.start_episode(0, None)
    trajectory.add(
        np.array([np.nan, np.inf], dtype=np.float32), 2, 0.5, False, False
    )
    trajectory.add(np.array([1e-5, -np.inf]), 1, -1.0, True, False)
    trajectory.end_episode()
    assert trajectory.columns()[:5] == [
        "episode", "step", "obs_0", "obs_1", "action"
    ]
    lines = trajectory.episode_lines(trajectory.episodes[0]).splitlines()
    assert lines == [
        "0,0,nan,inf,2,0.5,false,false,",
        "0,1,1e-05,-inf,1,-1.0,true,false,"
    ]

    # 1 つの値の観測は obs の列, 形の合わない観測は空
    scalar = Trajectory()
    scalar.add(np.int64(3), 0, 1.0, False, False)
    scalar.add(np.array([1, 2]), 0, 1.0, True, False)
    scalar.end_episode()
    assert scalar.columns()[2] == "obs"
    lines = scalar.episode_lines(scalar.episodes[0]).splitlines()
    assert [line.split(",")[2] for line in lines] == ["3", ""]

    # 上限の確認
    with pytest.raises(ValueError, match="max_rows"):
        Trajectory(max_rows=0)
    with pytest.raises(ValueError, match="max_trajectory_rows"):
        EvaluationLoader(ToyEnv(), RightModel(), max_trajectory_rows=True)
    with pytest.raises(ValueError, match="record_trajectory"):
        EvaluationLoader(ToyEnv(), RightModel(), record_trajectory=1)


def test_wide_observation_is_rejected() -> None:
    """
    要素の数が上限を超える観測を記録しないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 空間で分かる場合は, 記録を始める前に止まる
    with pytest.raises(ValueError, match="上限"):
        EvaluationLoader(
            WideEnv(MAX_ELEMENTS + 1), RightModel(), episodes=1,
            record_trajectory=True
        )

    # 空間の分からない値は, 最初の歩で止まる
    trajectory = Trajectory()
    with pytest.raises(ValueError, match="上限"):
        trajectory.add(np.zeros(MAX_ELEMENTS + 1), 0, 1.0, False, False)


def test_save_trajectory_checks(tmp_path: Path) -> None:
    """
    save_trajectory の名前, 軌跡, エポック, 置き場の確認を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    trajectory = Trajectory()
    with pytest.raises(ValueError, match="軌跡の名前"):
        save_trajectory(tmp_path, "../x", trajectory)
    with pytest.raises(ValueError, match="Trajectory"):
        save_trajectory(tmp_path, "x", [])
    with pytest.raises(ValueError, match="epoch"):
        save_trajectory(tmp_path, "x", trajectory, epoch=-1)

    # 置き場がリンクなら書かない
    outside = tmp_path / "outside"
    outside.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "trajectories").symlink_to(outside)
    with pytest.raises(ValueError, match="リンク"):
        save_trajectory(run_dir, "x", trajectory)
    assert list(outside.iterdir()) == []

    # 何も記録していない軌跡は見出しだけの表
    empty = tmp_path / "empty"
    empty.mkdir()
    path = save_trajectory(empty, "x", trajectory)
    assert path.read_text(encoding="utf-8").count("\n") == 1
    assert read_info(empty, "x")["rows"] == 0


def test_best_trajectory_with_callback(tmp_path: Path) -> None:
    """
    学習中にベストモデルを更新した検証の軌跡を, 登録した関数で書き出せる
    ことを確かめる関数 (DESIGN.md の例)

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(
        tmp_path,
        "trainer:\n  epochs: 2\n  progress_bar: false\n"
        "  monitor: valid/reward\n  mode: max\n"
    )
    model = toy_model()
    train_loader = RolloutLoader(ToyEnv(), model, episodes=2, seed=0)
    valid_loader = EvaluationLoader(
        ToyEnv(), model, episodes=2, seed=100, record_trajectory=True
    )
    saved = []

    def keep_best(info: dict) -> None:
        """
        ベストモデルを更新したエポックの検証の軌跡を書く関数

        Parameters
        ----------
        info: dict
            エポックの結果

        Returns
        ----------
        None
        """
        if info["improved"]:
            save_trajectory(
                experiment, "valid_best", valid_loader.trajectory,
                epoch=info["epoch"]
            )
            saved.append(info["epoch"])

    trainer = Trainer(model, experiment, ReinforceStep())
    trainer.add_callback(keep_best)
    trainer.fit(train_loader, valid_loader)
    assert saved and saved[0] == 1
    info = read_info(Path(experiment.run_dir), "valid_best")
    assert info["epoch"] == saved[-1]
    assert info["episodes"] == 2 and info["rows"] == 10


def test_bytes_limit_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    記録の間も, 書き出しと同じ文字数の上限で貯めるのを止めることを
    確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        上限を小さくする

    Returns
    ----------
    None
    """
    # 上限を見出しと 7 行の文字数にすると, 5 歩のエピソードは 1 つだけ入る
    loader = EvaluationLoader(
        ToyEnv(), RightModel(), episodes=3, record_trajectory=True
    )
    next(iter(loader))
    full = loader.trajectory
    text = full.header() + "".join(
        full.episode_lines(steps) for steps in full.episodes
    )
    lines = text.splitlines(keepends=True)
    limit = len("".join(lines[:8]).encode("utf-8"))
    monkeypatch.setattr(trajectory_module, "MAX_TRAJECTORY_BYTES", limit)
    next(iter(loader))
    small = loader.trajectory
    assert small.full is True
    assert small.n_rows == 5 and small.dropped_episodes == 2
    assert small.n_bytes + len(small.header().encode("utf-8")) <= limit


class DictSpace:
    def __init__(self):
        # 形を持たない空間 (Gymnasium の Dict と同じ)
        self.shape = None
        self.spaces = {"position": Space(low=[0.0], high=[1.0])}


class TupleEnv(ToyEnv):
    def __init__(self):
        # 親クラスのコンストラクタを呼び出す
        super(TupleEnv, self).__init__()

        # 表の列にできない観測の空間を保存
        self.observation_space = DictSpace()


def test_non_table_space_is_rejected(tmp_path: Path) -> None:
    """
    Discrete と Box の他の空間を, 記録を始める前に断ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    with pytest.raises(ValueError, match="Discrete か Box"):
        EvaluationLoader(
            TupleEnv(), RightModel(), episodes=1, record_trajectory=True
        )
    policy = guarded_policy(tmp_path, 100.0, "stop")
    env = TupleEnv()
    with pytest.raises(ValueError, match="Discrete か Box"):
        evaluate_policy(policy, env, episodes=1, trajectory=Trajectory())
    assert env.seeds == []

    # 記録しない評価と, 形が () の Box は使える
    EvaluationLoader(TupleEnv(), RightModel(), episodes=1)
    scalar = Trajectory(
        observation_space=Space(low=1.0, high=2.0), action_space=Space(n=2)
    )
    scalar.check_spaces()


def test_failed_write_keeps_old_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    表の置き換えに失敗した場合に, 元の表と説明を残し一時ファイルを消すことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        表の置き換えを失敗させる

    Returns
    ----------
    None
    """
    loader = EvaluationLoader(
        ToyEnv(), RightModel(), episodes=1, record_trajectory=True
    )
    next(iter(loader))
    save_trajectory(tmp_path, "valid", loader.trajectory, epoch=1)
    folder = tmp_path / "trajectories"
    old_csv = (folder / "valid.csv").read_bytes()
    old_info = (folder / "valid.json").read_bytes()

    # 説明の後に表を置き換えるところで失敗させる
    replace = trajectory_module.os.replace

    def failing_replace(source: Path, target: Path) -> None:
        """
        表の置き換えだけを失敗させる関数

        Parameters
        ----------
        source: Path
            一時ファイル
        target: Path
            置き換える先

        Returns
        ----------
        None
        """
        if Path(target).suffix == ".csv":
            raise OSError("置き換えの失敗")
        replace(source, target)

    monkeypatch.setattr(trajectory_module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="置き換えの失敗"):
        save_trajectory(tmp_path, "valid", loader.trajectory, epoch=2)
    assert (folder / "valid.csv").read_bytes() == old_csv
    assert (folder / "valid.json").read_bytes() == old_info
    assert sorted(p.name for p in folder.iterdir()) == [
        "valid.csv", "valid.json"
    ]

    # 元の無い名前では, 説明も残さない
    with pytest.raises(OSError, match="置き換えの失敗"):
        save_trajectory(tmp_path, "fresh", loader.trajectory)
    assert sorted(p.name for p in folder.iterdir()) == [
        "valid.csv", "valid.json"
    ]
