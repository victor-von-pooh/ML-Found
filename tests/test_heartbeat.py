import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest
from torch import nn

from ml_found.training import heartbeat as heartbeat_module
from ml_found.training.heartbeat import (
    HEARTBEAT_FILE, HEARTBEAT_INTERVAL, PostHeartbeat, end_heartbeats
)
from ml_found.training.trainer import Trainer
from test_trainer import make_experiment, make_loader, regression_step

# 子のプロセスで心拍を始め, 書き直した心拍がファイルに届いてから,
# 引数の終わり方で終わるスクリプト
CHILD_SCRIPT = """
import json
import os
import signal
import sys
import time
from pathlib import Path
from ml_found.training.heartbeat import HEARTBEAT_FILE, PostHeartbeat

heartbeat = PostHeartbeat(sys.argv[1], interval=0.05)
heartbeat.start()
path = Path(sys.argv[1]) / HEARTBEAT_FILE
while json.loads(path.read_text(encoding="utf-8"))["beat"] < 2:
    time.sleep(0.01)
if sys.argv[2] == "kill":
    os.kill(os.getpid(), signal.SIGKILL)
if sys.argv[2] == "error":
    raise RuntimeError("後の処理のエラー")
"""


def read_heartbeat(run_dir: Path) -> dict:
    """
    実験のディレクトリの heartbeat.json を読み込む関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ

    Returns
    ----------
    data: dict
        ファイルの内容
    """
    data = json.loads((run_dir / HEARTBEAT_FILE).read_text(encoding="utf-8"))

    return data


def wait_beat(run_dir: Path, beat: int, timeout: float = 5.0) -> dict:
    """
    心拍の連番が決めた数に届くまで待つ関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ
    beat: int
        待つ連番
    timeout: float = 5.0
        待つ秒数の上限

    Returns
    ----------
    data: dict
        届いたときのファイルの内容
    """
    # 置き換えで書くので, 読むたびに古いか新しいかのどちらか
    deadline = time.monotonic() + timeout
    data = read_heartbeat(run_dir)
    while data["beat"] < beat:
        assert time.monotonic() < deadline, data
        time.sleep(0.01)
        data = read_heartbeat(run_dir)

    return data


@pytest.fixture(autouse=True)
def end_all() -> Iterator[None]:
    """
    テストの後に, 残った心拍の書き直しを止める関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    yield
    end_heartbeats()


def test_fit_starts_heartbeat(tmp_path: Path) -> None:
    """
    fit を終えたときから心拍を書き, end_post で終えることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 2\n")
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader(), make_loader())

    # 後の処理の間は running で, 書いているのは学習のプロセス
    run = json.loads((experiment.run_dir / "run.json").read_text())
    data = read_heartbeat(experiment.run_dir)
    assert data["state"] == "running"
    assert data["pid"] == os.getpid() == run["pid"]
    assert data["hostname"] == run["hostname"]
    assert data["beat"] == 1
    assert data["interval_sec"] == HEARTBEAT_INTERVAL == 10.0
    assert data["ended_at"] is None
    assert trainer.heartbeat.thread.is_alive()

    # 終えたときの ended の書き込みとスレッドの停止 (2 回目は書き直し無し)
    trainer.end_post()
    ended = read_heartbeat(experiment.run_dir)
    assert ended["state"] == "ended"
    assert ended["beat"] == 2
    assert ended["ended_at"] is not None
    assert ended["started_at"] == data["started_at"]
    assert not trainer.heartbeat.thread.is_alive()
    trainer.end_post()
    assert read_heartbeat(experiment.run_dir) == ended


def test_no_heartbeat_before_fit_or_on_error(tmp_path: Path) -> None:
    """
    fit の前とエラーで止まった fit では心拍を書かないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    def failing_step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        学習を止めるエラーを出す関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: tuple
            x と y
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            返さない
        """
        raise RuntimeError("学習のエラー")

    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 1\n")
    trainer = Trainer(nn.Linear(1, 1), experiment, failing_step)
    trainer.end_post()
    with pytest.raises(RuntimeError):
        trainer.fit(make_loader())
    assert not (experiment.run_dir / HEARTBEAT_FILE).exists()
    assert trainer.heartbeat is None


def test_heartbeat_rewrites_at_interval(tmp_path: Path) -> None:
    """
    間隔ごとに置き換えで書き直し, 止まれば書き直さないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    heartbeat = PostHeartbeat(tmp_path, interval=0.02)
    heartbeat.start()
    path = tmp_path / HEARTBEAT_FILE
    first = os.stat(path).st_mtime_ns

    # 書き直すたびの連番と更新の時刻の変化
    data = wait_beat(tmp_path, 3)
    assert data["state"] == "running"
    assert data["interval_sec"] == 0.02
    assert os.stat(path).st_mtime_ns > first

    # 強制終了と同じく, 終えずに書き直しだけ止まった場合の running の残り
    heartbeat.stop_event.set()
    heartbeat.thread.join()
    stopped = read_heartbeat(tmp_path)
    time.sleep(0.1)
    assert read_heartbeat(tmp_path) == stopped
    assert stopped["state"] == "running"
    assert list(tmp_path.iterdir()) == [path]


def test_refit_and_next_experiment(tmp_path: Path) -> None:
    """
    次の fit と start_experiment で前の心拍を終えることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    first = make_experiment(tmp_path / "a", "trainer:\n  epochs: 1\n")
    trainer = Trainer(nn.Linear(1, 1), first, regression_step)
    trainer.fit(make_loader())
    started = read_heartbeat(first.run_dir)["started_at"]

    # 同じ Trainer の続きの学習の間は心拍のファイルが無く, 終えると書き直し
    seen = []

    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        学習中に心拍のファイルがあるかを記録してから誤差を求める関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: tuple
            x と y
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            平均二乗誤差と平均絶対誤差
        """
        seen.append((first.run_dir / HEARTBEAT_FILE).exists())
        outputs = regression_step(model, batch, phase)

        return outputs

    old = trainer.heartbeat
    trainer.step_fn = step
    trainer.fit(make_loader())
    assert seen and not any(seen)
    assert not old.thread.is_alive()
    data = read_heartbeat(first.run_dir)
    assert data["state"] == "running"
    assert data["beat"] == 1
    assert data["started_at"] >= started

    # 次の実験を始めると, 前の実験の心拍は ended
    make_experiment(tmp_path / "b", "trainer:\n  epochs: 1\n")
    assert read_heartbeat(first.run_dir)["state"] == "ended"
    assert heartbeat_module.ACTIVE == set()


def test_other_trainer_fit_ends_heartbeat(tmp_path: Path) -> None:
    """
    同じプロセスの別の Trainer の fit で前の心拍を終えることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    first = make_experiment(tmp_path / "a", "trainer:\n  epochs: 1\n")
    second = make_experiment(tmp_path / "b", "trainer:\n  epochs: 1\n")
    Trainer(nn.Linear(1, 1), first, regression_step).fit(make_loader())
    Trainer(nn.Linear(1, 1), second, regression_step).fit(make_loader())
    assert read_heartbeat(first.run_dir)["state"] == "ended"
    assert read_heartbeat(second.run_dir)["state"] == "running"


@pytest.mark.parametrize(
    ("how", "state", "code"),
    [("exit", "ended", 0), ("error", "ended", 1), ("kill", "running", -9)]
)
def test_process_end(tmp_path: Path, how: str, state: str, code: int) -> None:
    """
    プロセスの終わり方ごとの心拍の終わりの扱いを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    how: str
        exit (普通に終わる), error (処理されないエラー), kill (強制終了)
    state: str
        終わった後の state
    code: int
        プロセスの終了コード

    Returns
    ----------
    None
    """
    result = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT, str(tmp_path), how],
        capture_output=True, timeout=60
    )
    assert result.returncode == code
    data = read_heartbeat(tmp_path)
    assert data["state"] == state
    assert data["pid"] != os.getpid()
    assert data["beat"] >= 2

    # 強制終了の後の, 書き直しが止まったままの running (読む側の止まった印)
    if state == "running":
        assert data["ended_at"] is None
        time.sleep(0.2)
        assert read_heartbeat(tmp_path) == data


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork が無い OS")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_forked_child_does_not_write(tmp_path: Path) -> None:
    """
    fork で作った子のプロセスが親の心拍を書き換えないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    heartbeat = PostHeartbeat(tmp_path, interval=60.0)
    heartbeat.start()
    before = read_heartbeat(tmp_path)

    # 子では終える, 消す, まとめて終えるの全てを呼び, atexit を通らずに終了
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            heartbeat.end()
            heartbeat.cancel()
            end_heartbeats()
        except BaseException:
            code = 1
        os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 0

    # 親の心拍は書いたときのまま (pid も親) で, 親のスレッドも動いたまま
    after = read_heartbeat(tmp_path)
    assert after == before
    assert after["pid"] == os.getpid()
    assert heartbeat.thread.is_alive()
    heartbeat.end()
    assert read_heartbeat(tmp_path)["state"] == "ended"


def test_write_error_warns_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    心拍を書けない場合に警告を 1 回だけ出し, 後の処理を止めないことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        書き込みの差し替えに使う

    Returns
    ----------
    None
    """
    def failing_write(path: Path, data: dict, sync: bool = False) -> None:
        """
        ディスクの不足と同じく書き込みに失敗する関数

        Parameters
        ----------
        path: Path
            書き込むファイル
        data: dict
            書き込む内容
        sync: bool = False
            置き換えの前にディスクに書き出すか

        Returns
        ----------
        None
        """
        raise OSError("書き込みの失敗")

    monkeypatch.setattr(heartbeat_module, "write_json", failing_write)
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 1\n")
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)

    # fit の後の最初の心拍と終わりの両方で失敗しても, エラーは出ない
    trainer.fit(make_loader())
    trainer.end_post()
    assert trainer.heartbeat.beat == 2
    assert not (experiment.run_dir / HEARTBEAT_FILE).exists()

    # 警告は train.log に 1 回だけ
    log = (experiment.run_dir / "train.log").read_text(encoding="utf-8")
    assert log.count("学習の後の処理の心拍を書き込めませんでした") == 1


def test_ended_is_synced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    終えた状態だけをディスクまで書き出すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        書き込みの差し替えに使う

    Returns
    ----------
    None
    """
    calls = []
    original = heartbeat_module.write_json

    def recording_write(path: Path, data: dict, sync: bool = False) -> None:
        """
        状態と書き出しの有無を記録してから書き込む関数

        Parameters
        ----------
        path: Path
            書き込むファイル
        data: dict
            書き込む内容
        sync: bool = False
            置き換えの前にディスクに書き出すか

        Returns
        ----------
        None
        """
        calls.append((data["state"], sync))
        original(path, data, sync)

    monkeypatch.setattr(heartbeat_module, "write_json", recording_write)
    heartbeat = PostHeartbeat(tmp_path, interval=60.0)
    heartbeat.start()
    heartbeat.end()
    assert calls == [("running", False), ("ended", True)]
