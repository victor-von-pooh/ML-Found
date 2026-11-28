import atexit
import logging
import os
import socket
import threading
from pathlib import Path
from typing import Any, Optional

from ml_found.training.status import now_iso, write_json

# 学習の後の処理の心拍を書くファイルの名前
HEARTBEAT_FILE = "heartbeat.json"

# 心拍を書き直す間隔 (秒)
HEARTBEAT_INTERVAL = 10.0

# 止めるときに, 書き直しのスレッドの終わりと書き込みの鍵を待つ秒数の上限
# (ディスクが止まっても, プロセスの終わりを止めないため)
JOIN_TIMEOUT = 5.0


class PostHeartbeat:
    def __init__(
        self, run_dir: Path, logger: Optional[logging.Logger] = None,
        interval: float = HEARTBEAT_INTERVAL
    ):
        # 実験のディレクトリ, ロガー, 書き直す間隔 (秒) を保存
        self.run_dir = Path(run_dir)
        self.logger = logger
        self.interval = interval

        # 書く内容, 書いたプロセス, 連番の入れ物を用意
        self.data: dict[str, Any] = {}
        self.pid = os.getpid()
        self.beat = 0

        # 書き直しのスレッド, 止める合図, 書き込みを 1 つずつにする鍵を用意
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        # 終えたかの印と, 書き込みの失敗を 1 回だけ警告するための印を用意
        self.ended = False
        self.warned = False

    def write(self, sync: bool = False) -> None:
        """
        心拍のファイルを書き込み, 失敗しても後の処理を止めない関数

        Parameters
        ----------
        sync: bool = False
            置き換えの前にディスクに書き出すか (終えた状態の 1 回だけ)

        Returns
        ----------
        None
        """
        # 心拍は補助の機能なので, ディスクの不足などでは後の処理を続行
        self.beat += 1
        self.data["beat"] = self.beat
        self.data["updated_at"] = now_iso()
        try:
            write_json(self.run_dir / HEARTBEAT_FILE, self.data, sync)
        except OSError as error:
            if not self.warned and self.logger is not None:
                self.logger.warning(
                    f"学習の後の処理の心拍を書き込めませんでした: {error}"
                )
            self.warned = True

    def start(self) -> None:
        """
        心拍を書き, 決まった間隔で書き直すスレッドを始める関数

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 読む側がすぐに分かるように, 最初の心拍は呼んだところで書き込み
        now = now_iso()
        self.data = {
            "state": "running",
            "pid": self.pid,
            "hostname": socket.gethostname(),
            "beat": 0,
            "interval_sec": self.interval,
            "started_at": now,
            "updated_at": now,
            "ended_at": None
        }
        with self.lock:
            self.write()

        # プロセスの終わりを待たせないデーモンのスレッドで書き直し
        self.thread = threading.Thread(
            target=self.run, name="ml_found-heartbeat", daemon=True
        )
        self.thread.start()
        ACTIVE.add(self)

    def run(self) -> None:
        """
        止める合図が来るまで, 間隔ごとに心拍を書き直す関数

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 終えた後に書かないように, 鍵の中で印を確かめてから書き込み
        while not self.stop_event.wait(self.interval):
            with self.lock:
                if self.ended:
                    return
                self.write()

    def stop(self) -> bool:
        """
        書き直しのスレッドを止める関数

        Parameters
        ----------
        None

        Returns
        ----------
        stopped: bool
            このプロセスの心拍で, まだ終えていなかったかどうか
        """
        # fork で作った子のプロセス (DataLoader の worker など) では, 親の
        # ファイルに触れずに終了
        ACTIVE.discard(self)
        if os.getpid() != self.pid:
            return False

        # 合図を送ってスレッドの終わりを待ち, 終えた印の付け替え
        self.stop_event.set()
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(JOIN_TIMEOUT)

        # 書き込みが上限を超えて終わらない場合は, 終えた状態を書かずに諦め
        if not self.lock.acquire(timeout=JOIN_TIMEOUT):
            return False
        try:
            stopped = not self.ended
            self.ended = True
        finally:
            self.lock.release()

        return stopped

    def end(self) -> None:
        """
        後の処理を終えたことを心拍のファイルに書き, 書き直しを止める関数

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 止めてから, 終えた状態を最後の 1 回として書き込み
        if not self.stop():
            return
        with self.lock:
            self.data.update(state="ended", ended_at=now_iso())
            self.write(sync=True)

    def cancel(self) -> None:
        """
        書き直しを止め, 心拍のファイルを消す関数

        同じ Trainer で fit を呼び直し, 学習を続ける場合に使う

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 学習中は progress.json が様子を表すので, 心拍のファイルの削除
        # (fork で作った子のプロセスでは, 親のファイルを消さずに終了)
        if os.getpid() != self.pid:
            ACTIVE.discard(self)
            return
        self.stop()
        try:
            (self.run_dir / HEARTBEAT_FILE).unlink(missing_ok=True)
        except OSError:
            pass


# このプロセスで書き直している心拍 (次の実験を始めたときと, プロセスの
# 終わりに終えるため)
ACTIVE: set[PostHeartbeat] = set()


def end_heartbeats() -> None:
    """
    このプロセスで書き直している心拍を全て終える関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 終える間に入れ物が変わるので, 写しの順に終了
    for heartbeat in list(ACTIVE):
        heartbeat.end()


# プロセスが普通に終わるときの, 後の処理を終えたことの書き込み
atexit.register(end_heartbeats)
