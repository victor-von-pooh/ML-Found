import json
import logging
import os
import socket
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# 実験の状態を書くファイルの形式の版 (項目を変えた場合に更新)
FORMAT_VERSION = 1

# 学習を止めてほしい場合に置くファイルの名前
STOP_FILE = "STOP"


def now_iso() -> str:
    """
    今の日時をタイムゾーン付きの ISO 8601 の文字列で取得する関数

    Parameters
    ----------
    None

    Returns
    ----------
    now: str
        秒までの日時
    """
    now = datetime.now().astimezone().isoformat(timespec="seconds")

    return now


def write_json(path: Path, data: dict) -> None:
    """
    読み手が書きかけのファイルを読まないように JSON を書き込む関数

    Parameters
    ----------
    path: Path
        書き込むファイル
    data: dict
        書き込む内容

    Returns
    ----------
    None
    """
    # 一時ファイルに書いてから置き換え, 読み手が読むのは古いか新しいかのどちらか
    # 置き換えの前にディスクに書き出し, 失敗した場合は一時ファイルの削除
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False, encoding="utf-8"
        ) as file:
            temp = Path(file.name)
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
        temp = None
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


class RunStatus:
    def __init__(
        self, run_dir: Path, logger: Optional[logging.Logger] = None,
        progress_interval: float = 1.0
    ):
        # 実験のディレクトリ, ロガー, 進み具合を書き込む最短の間隔 (秒) を保存
        self.run_dir = Path(run_dir)
        self.logger = logger
        self.progress_interval = progress_interval

        # run.json の内容, 最後に進み具合を書いた時刻, 開始時刻の入れ物を用意
        self.data: dict[str, Any] = {}
        self.last_progress = 0.0
        self.started = time.monotonic()

        # 書き込みの失敗を 1 回だけ警告するための印を用意
        self.warned = False

    def write(self, name: str, data: dict) -> None:
        """
        状態のファイルを書き込み, 失敗しても学習を止めない関数

        Parameters
        ----------
        name: str
            ファイル名
        data: dict
            書き込む内容

        Returns
        ----------
        None
        """
        # 状態の書き出しは補助の機能なので, ディスクの不足などでは学習を続行
        try:
            write_json(self.run_dir / name, data)
        except OSError as error:
            if not self.warned and self.logger is not None:
                self.logger.warning(f"実験の状態を書き込めませんでした: {error}")
            self.warned = True

    def start(
        self, planned_epochs: int, start_epoch: int, monitor: str,
        mode: str, seed: Optional[int], device: str
    ) -> None:
        """
        学習を始めたことを run.json に書く関数

        Parameters
        ----------
        planned_epochs: int
            予定のエポック数 (trainer.epochs)
        start_epoch: int
            最初のエポック, 再開した場合は 2 以上
        monitor: str
            監視する指標
        mode: str
            監視する指標の向き
        seed: Optional[int]
            乱数シード
        device: str
            学習に使うデバイス

        Returns
        ----------
        None
        """
        # 開始時刻と, 終わるまで更新していく項目の初期値の設定
        self.started = time.monotonic()
        self.last_progress = 0.0
        self.data = {
            "format_version": FORMAT_VERSION,
            "status": "running",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": now_iso(),
            "updated_at": now_iso(),
            "finished_at": None,
            "duration_sec": None,
            "planned_epochs": planned_epochs,
            "start_epoch": start_epoch,
            "completed_epochs": 0,
            "monitor": monitor,
            "mode": mode,
            "best_epoch": None,
            "best_value": None,
            "early_stopped_epoch": None,
            "final_weights": None,
            "seed": seed,
            "device": device,
            "error": None
        }
        self.write("run.json", self.data)

    def update(self, **fields: Any) -> None:
        """
        run.json の項目を更新する関数

        Parameters
        ----------
        **fields: Any
            更新する項目と値

        Returns
        ----------
        None
        """
        # 学習を始める前に呼ばれた場合は対象外
        if not self.data:
            return

        self.data.update(fields)
        self.data["updated_at"] = now_iso()
        self.write("run.json", self.data)

    def finish(self, status: str, **fields: Any) -> None:
        """
        学習を終えたことを run.json に書き, 進み具合のファイルを消す関数

        Parameters
        ----------
        status: str
            completed, stopped, failed のどれか
        **fields: Any
            一緒に更新する項目と値

        Returns
        ----------
        None
        """
        self.update(
            status=status, finished_at=now_iso(),
            duration_sec=round(time.monotonic() - self.started, 3), **fields
        )

        # 終わった学習の進み具合の削除
        try:
            (self.run_dir / "progress.json").unlink(missing_ok=True)
        except OSError:
            pass

    def progress(
        self, phase: str, epoch: int, total_epochs: int, batch: int,
        total_batches: Optional[int], force: bool = False
    ) -> None:
        """
        学習の進み具合を一定の間隔で progress.json に書く関数

        Parameters
        ----------
        phase: str
            train か valid
        epoch: int
            今のエポック
        total_epochs: int
            予定のエポック数
        batch: int
            今のエポックで終えたバッチの数
        total_batches: Optional[int]
            1 エポックのバッチの数, 分からない場合は None
        force: bool = False
            間隔に関係なく書く場合は True

        Returns
        ----------
        None
        """
        # 書き込みが学習の邪魔にならないように間隔の確保
        now = time.monotonic()
        if not force and now - self.last_progress < self.progress_interval:
            return
        self.last_progress = now

        data = {
            "phase": phase,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "batch": batch,
            "total_batches": total_batches,
            "elapsed_sec": round(now - self.started, 3),
            "updated_at": now_iso()
        }
        self.write("progress.json", data)

    def stop_requested(self) -> bool:
        """
        学習を止めてほしいという要求があるかを確かめる関数

        Parameters
        ----------
        None

        Returns
        ----------
        requested: bool
            実験のディレクトリに STOP のファイルがあるかどうか
        """
        requested = (self.run_dir / STOP_FILE).exists()

        return requested
