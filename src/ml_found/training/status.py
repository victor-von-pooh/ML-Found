import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# 実験の状態を書くファイルの形式の版, 項目を変えたら上げる
FORMAT_VERSION = 1

# 学習を止めてほしい場合に置くファイルの名前
STOP_FILE = "STOP"


def now_iso() -> str:
    """
    今の日時を ISO 8601 の文字列で取得する関数

    Parameters
    ----------
    None

    Returns
    ----------
    now: str
        秒までの日時
    """
    now = datetime.now().isoformat(timespec="seconds")

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
    # 一時ファイルに書いてから置き換えると, 読み手は古いか新しいかのどちらかを読む
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temp, path)


class RunStatus:
    def __init__(self, run_dir: Path, progress_interval: float = 1.0):
        # 実験のディレクトリと, 進み具合を書き込む最短の間隔 (秒) を保存
        self.run_dir = Path(run_dir)
        self.progress_interval = progress_interval

        # run.json の内容と, 最後に進み具合を書いた時刻を用意
        self.data: dict[str, Any] = {}
        self.last_progress = 0.0
        self.started = time.time()

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
        self.started = time.time()
        self.data = {
            "format_version": FORMAT_VERSION,
            "status": "running",
            "pid": os.getpid(),
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
        write_json(self.run_dir / "run.json", self.data)

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
        # 学習を始める前に呼ばれた場合は何もしない
        if not self.data:
            return

        self.data.update(fields)
        self.data["updated_at"] = now_iso()
        write_json(self.run_dir / "run.json", self.data)

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
            duration_sec=round(time.time() - self.started, 3), **fields
        )

        # 終わった学習の進み具合は残さない
        (self.run_dir / "progress.json").unlink(missing_ok=True)

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
        # 書き込みが学習の邪魔にならないように間隔を空ける
        now = time.time()
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
        write_json(self.run_dir / "progress.json", data)

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
