import json
import os
from pathlib import Path

import pytest

from ml_found import load_samples, load_task, save_sample
from ml_found.experiment import results, task
from ml_found.experiment.results import read_samples_index
from ml_found.experiment.task import TASK_FILE
from ml_found.training import status
from ml_found.training.status import RunStatus, read_run_status
from ml_found.utils import files
from test_trainer import make_experiment

# FIFO を作れない環境 (Windows など) では FIFO の確かめを飛ばす印
NEEDS_FIFO = pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="FIFO を作れない"
)


def put_bad_file(path: Path, kind: str, content: bytes) -> None:
    """
    読まないはずのファイル (リンク, ディレクトリ, FIFO) を置く関数

    Parameters
    ----------
    path: Path
        置く場所
    kind: str
        link (実験の外の同じ中身のファイルへのリンク), dir, fifo のどれか
    content: bytes
        リンクの先のファイルの中身

    Returns
    ----------
    None
    """
    if kind == "link":
        outside = path.parent.parent / f"outside_{path.name}"
        outside.write_bytes(content)
        path.symlink_to(outside)
    elif kind == "dir":
        path.mkdir()
    else:
        os.mkfifo(path)


BAD_KINDS = ["link", "dir", pytest.param("fifo", marks=NEEDS_FIFO)]


@pytest.mark.parametrize("dir_fd", [True, False])
def test_read_plain_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dir_fd: bool
) -> None:
    """
    共通の読み方が, 上限までの普通のファイルだけを読むことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        dir_fd の使い方を差し替える
    dir_fd: bool
        開いたディレクトリから開くか (False は Windows などの読み方)

    Returns
    ----------
    None
    """
    if not dir_fd:
        monkeypatch.setattr(files, "can_use_dir_fd", lambda: False)
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "edge").write_bytes(b"x" * 4)
    (folder / "large").write_bytes(b"x" * 5)
    put_bad_file(folder / "link", "link", b"x")
    readable, fd = files.open_plain_dir(folder)
    assert readable
    try:
        assert files.read_plain_file(folder, "edge", 4, fd) == b"xxxx"
        assert files.read_plain_file(folder, "large", 4, fd) is None
        assert files.read_plain_file(folder, "link", 4, fd) is None
        assert files.read_plain_file(folder, "missing", 4, fd) is None
    finally:
        if fd is not None:
            os.close(fd)

    # リンクのディレクトリは開かない
    (tmp_path / "linked").symlink_to(folder, target_is_directory=True)
    assert files.open_plain_dir(tmp_path / "linked") == (False, None)
    assert files.open_plain_dir(tmp_path / "missing") == (False, None)


def test_stop_plain_file(tmp_path: Path) -> None:
    """
    普通のファイルの STOP の要求と, now の読み方を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 毎回確かめるように, 間隔を 0 にした状態
    run = RunStatus(tmp_path, progress_interval=0.0)
    stop = tmp_path / "STOP"
    assert not run.stop_requested()
    assert not run.stop_now_requested()
    stop.touch()
    assert run.stop_requested()
    assert not run.stop_now_requested()
    stop.write_text("﻿ NOW \n", encoding="utf-8")
    assert run.stop_now_requested()

    # 上限までは中身を確かめ, 超える場合はエポックの区切りで止める要求
    stop.write_bytes(b"now".ljust(status.MAX_STOP_BYTES))
    assert run.stop_now_requested()
    stop.write_bytes(b"now".ljust(status.MAX_STOP_BYTES + 1))
    assert run.stop_requested()
    assert not run.stop_now_requested()


@pytest.mark.parametrize("kind", BAD_KINDS)
def test_stop_not_plain_file(tmp_path: Path, kind: str) -> None:
    """
    リンク, ディレクトリ, FIFO の STOP を, 開かずにエポックの区切りで止める
    要求とし, 中身を now と読まないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    kind: str
        置く STOP の種類

    Returns
    ----------
    None
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    put_bad_file(run_dir / "STOP", kind, b"now\n")
    run = RunStatus(run_dir, progress_interval=0.0)
    assert run.stop_requested()
    assert not run.stop_now_requested()


def test_stop_dangling_link(tmp_path: Path) -> None:
    """
    先の無いリンクの STOP も, エポックの区切りで止める要求とすることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    (tmp_path / "STOP").symlink_to(tmp_path / "missing")
    run = RunStatus(tmp_path, progress_interval=0.0)
    assert run.stop_requested()
    assert not run.stop_now_requested()


def test_read_plain_file_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    調べた後に置き換わったファイルを, 1 回だけ読み直すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        置き換わりの見分けを差し替える

    Returns
    ----------
    None
    """
    (tmp_path / "run.json").write_bytes(b"{}")
    calls = []
    limit = [1]
    samestat = os.path.samestat

    def replaced_first(first: os.stat_result, second: os.stat_result) -> bool:
        """
        初めの limit[0] 回だけ置き換わったものとする関数

        Parameters
        ----------
        first: os.stat_result
            調べた結果
        second: os.stat_result
            開いたものの結果

        Returns
        ----------
        same: bool
            同じファイルかどうか
        """
        calls.append(None)
        same = len(calls) > limit[0] and samestat(first, second)

        return same

    # 1 回の置き換わりは読み直して読め, 2 回続くと読まない
    monkeypatch.setattr(os.path, "samestat", replaced_first)
    assert files.read_plain_file(tmp_path, "run.json", 10) == b"{}"
    assert len(calls) == 2
    calls.clear()
    limit[0] = 2
    assert files.read_plain_file(tmp_path, "run.json", 10) is None
    assert len(calls) == 2


@pytest.mark.parametrize("kind", BAD_KINDS)
def test_run_status_not_plain_file(tmp_path: Path, kind: str) -> None:
    """
    リンク, ディレクトリ, FIFO の run.json を, 止まらずに無いものとする
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    kind: str
        置く run.json の種類

    Returns
    ----------
    None
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    put_bad_file(
        run_dir / "run.json", kind, json.dumps({"status": "running"}).encode()
    )
    assert read_run_status(run_dir) is None


def test_run_status_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    上限を超える run.json を無いものとすることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        大きさの上限を差し替える

    Returns
    ----------
    None
    """
    text = json.dumps({"status": "running"}).encode()
    (tmp_path / "run.json").write_bytes(text)
    monkeypatch.setattr(status, "MAX_RUN_BYTES", len(text))
    assert read_run_status(tmp_path) == {"status": "running"}
    (tmp_path / "run.json").write_bytes(text + b" ")
    assert read_run_status(tmp_path) is None


@pytest.mark.parametrize("kind", BAD_KINDS)
def test_load_task_not_plain_file(tmp_path: Path, kind: str) -> None:
    """
    リンク, ディレクトリ, FIFO の task.yaml を, 止まらずに無いものとする
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    kind: str
        置く task.yaml の種類

    Returns
    ----------
    None
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    put_bad_file(run_dir / TASK_FILE, kind, b"format_version: 1\n")
    assert load_task(run_dir) is None


def test_load_task_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    上限を超える task.yaml を無いものとすることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        大きさの上限を差し替える

    Returns
    ----------
    None
    """
    text = b"format_version: 1\nrole: predict\n"
    (tmp_path / TASK_FILE).write_bytes(text)
    monkeypatch.setattr(task, "MAX_TASK_BYTES", len(text))
    assert load_task(tmp_path)["role"] == "predict"
    (tmp_path / TASK_FILE).write_bytes(text + b" ")
    assert load_task(tmp_path) is None


@pytest.mark.parametrize("kind", BAD_KINDS + ["large"])
def test_samples_index_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """
    リンク, ディレクトリ, FIFO, 大きすぎる一覧を, 止まらずに読めない一覧と
    することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        大きさの上限を差し替える
    kind: str
        置く一覧の種類 (large は上限を超える普通のファイル)

    Returns
    ----------
    None
    """
    # 上限ちょうどの一覧は読める
    experiment = make_experiment(tmp_path)
    root = experiment.run_dir / "samples"
    save_sample(experiment, "note", "文", epoch=1)
    index = root / "index.json"
    content = index.read_bytes()
    monkeypatch.setattr(results, "MAX_SAMPLES_INDEX_BYTES", len(content))
    assert len(read_samples_index(index)[0]) == 1

    # 読まない一覧は, load_samples では一覧が無いもの, save_sample では誤り
    index.unlink()
    if kind == "large":
        index.write_bytes(content + b" ")
    else:
        put_bad_file(index, kind, content)
    with pytest.raises(ValueError, match="見本の一覧"):
        read_samples_index(index)
    assert load_samples(experiment.run_dir) == []
    with pytest.raises(ValueError, match="見本の一覧"):
        save_sample(experiment, "note", "文", epoch=2)
    if kind == "link":
        assert (root.parent / "outside_index.json").read_bytes() == content


def test_samples_linked_folder(tmp_path: Path) -> None:
    """
    samples/ 自体がリンクの実験の一覧を読まず, 外に書かないことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 外の samples/ の形のディレクトリへのリンク
    outside = make_experiment(tmp_path / "outside")
    save_sample(outside, "note", "文", epoch=1)
    (outside.run_dir / "samples" / "epoch_010.png").write_bytes(b"")
    experiment = make_experiment(tmp_path / "run")
    (experiment.run_dir / "samples").symlink_to(
        outside.run_dir / "samples", target_is_directory=True
    )
    index = experiment.run_dir / "samples" / "index.json"
    with pytest.raises(ValueError, match="リンク"):
        read_samples_index(index)
    assert load_samples(experiment.run_dir) == []
    with pytest.raises(ValueError, match="リンク"):
        save_sample(experiment, "other", "文", epoch=1)
    assert not (outside.run_dir / "samples" / "other").exists()

    # 無い samples/ は空の一覧
    assert read_samples_index(tmp_path / "samples" / "index.json") == (
        [], results.SAMPLES_FORMAT_VERSION
    )


def test_save_sample_broken_index_creates_nothing(tmp_path: Path) -> None:
    """
    壊れた一覧の実験で, save_sample が見本のディレクトリを作らずに止まる
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    root = experiment.run_dir / "samples"
    root.mkdir()
    (root / "index.json").write_text('{"samples": 3}', encoding="utf-8")
    with pytest.raises(ValueError, match="見本の一覧"):
        save_sample(experiment, "note", "文", epoch=1)
    assert not (root / "note").exists()
    (root / "index.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        save_sample(experiment, "note", "文", epoch=1)
    assert not (root / "note").exists()


def test_save_sample_first_time(tmp_path: Path) -> None:
    """
    一覧が無い実験の初めての保存で, samples/ と一覧を作ることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    root = experiment.run_dir / "samples"
    assert not root.exists()
    path = save_sample(experiment, "note", "文", epoch=1)
    assert path == root / "note" / "epoch_0001.txt"
    samples, version = read_samples_index(root / "index.json")
    assert version == results.SAMPLES_FORMAT_VERSION
    assert [item["file"] for item in samples] == ["note/epoch_0001.txt"]
