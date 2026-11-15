import sys
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional

from ml_found.training.status import (
    read_run_status, record_not_started, record_unfinished
)

if TYPE_CHECKING:
    from ml_found.experiment.setup import Experiment

# 記録の対象の実験 (最後に start_experiment で始めたもの), 置き換える前の
# sys.excepthook, 置き換えたかの印
STATE: dict[str, Any] = {
    "experiment": None, "previous": None, "installed": False
}


def record_uncaught_error(
    experiment: "Experiment", error: BaseException
) -> bool:
    """
    学習の外で処理されなかったエラーを, train.log と run.json に記録する関数

    run.json が無い (学習を始める前に止まった) か running のまま (fit を
    使わずに学習した) の場合だけ記録する. 終わった状態 (fit と Trainer の
    作成が記録したエラーと, 終えた学習) は書き換えない

    Parameters
    ----------
    experiment: Experiment
        始めた実験
    error: BaseException
        起きたエラー, キーボードの割り込みは止めたものとして記録

    Returns
    ----------
    recorded: bool
        記録したかどうか
    """
    # 終わった状態の実験は記録済みなので対象外
    data = read_run_status(experiment.run_dir)
    if data is not None and data.get("status") != "running":
        return False

    # fit と同じく, 割り込みは警告, エラーは中身を含めて記録
    if isinstance(error, KeyboardInterrupt):
        experiment.logger.warning("キーボードの割り込みで実験が止まりました")
    else:
        experiment.logger.error(
            "学習の外で処理されなかったエラーが起きました",
            exc_info=(type(error), error, error.__traceback__)
        )

    # 外から読める状態に, 止まったことを記録
    if data is None:
        record_not_started(
            experiment.run_dir, experiment.logger, experiment.cfg,
            str(experiment.device), error
        )
    else:
        record_unfinished(experiment.run_dir, data, error)
    recorded = True

    return recorded


def handle_uncaught(
    exc_type: type, exc_value: BaseException,
    exc_traceback: Optional[TracebackType]
) -> None:
    """
    処理されなかったエラーを記録してから, 元の sys.excepthook に渡す関数

    Parameters
    ----------
    exc_type: type
        エラーの型
    exc_value: BaseException
        起きたエラー
    exc_traceback: Optional[TracebackType]
        エラーの traceback

    Returns
    ----------
    None
    """
    # 記録に失敗しても, 元のエラーの表示は必ず実行
    experiment = STATE["experiment"]
    if experiment is not None:
        try:
            record_uncaught_error(experiment, exc_value)
        except Exception as record_error:
            print(
                "ML-Found: 処理されなかったエラーを記録できませんでした: "
                f"{record_error}", file=sys.stderr
            )

    # 元の sys.excepthook (無い場合は Python の既定のもの) で表示
    previous = STATE["previous"] or sys.__excepthook__
    previous(exc_type, exc_value, exc_traceback)


def watch_uncaught_errors(experiment: "Experiment") -> None:
    """
    処理されなかったエラーを実験に記録するように, sys.excepthook を置き換える
    関数

    同じプロセスで何度呼んでも置き換えるのは 1 回だけで, 記録の対象は最後に
    渡した実験. ノートブックと IPython のセルのエラーは sys.excepthook を
    通らない (カーネルが動き続ける) ので記録しない

    Parameters
    ----------
    experiment: Experiment
        start_experiment で始めた実験

    Returns
    ----------
    None
    """
    # 記録の対象を最新の実験に変更 (ロガーも最新の実験の train.log に書くため)
    STATE["experiment"] = experiment

    # 元の sys.excepthook を残して 1 回だけ置き換え (他のものが後から
    # 置き換えて元に繋いだ場合に, 互いに呼び合わないように 2 回目以降は省略)
    if STATE["installed"]:
        return
    STATE["previous"] = sys.excepthook
    sys.excepthook = handle_uncaught
    STATE["installed"] = True
