import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Optional

import pytest
from torch import nn

from ml_found.experiment import uncaught
from ml_found.experiment.uncaught import (
    STATE, handle_uncaught, watch_uncaught_errors
)
from ml_found.training.status import RunStatus
from ml_found.training.trainer import Trainer
from test_trainer import make_experiment, make_loader, regression_step


def raise_and_catch(error: BaseException) -> BaseException:
    """
    traceback を持つエラーを作る関数

    Parameters
    ----------
    error: BaseException
        投げるエラー

    Returns
    ----------
    caught: BaseException
        traceback の付いたエラー
    """
    try:
        raise error
    except BaseException as caught_error:
        caught = caught_error

    return caught


def call_hook(
    error: BaseException, monkeypatch: pytest.MonkeyPatch
) -> list:
    """
    元の sys.excepthook を差し替えて, 処理されなかったエラーとして渡す関数

    Parameters
    ----------
    error: BaseException
        渡すエラー
    monkeypatch: pytest.MonkeyPatch
        元の sys.excepthook の差し替えに使う

    Returns
    ----------
    calls: list
        元の sys.excepthook に渡ったエラー
    """
    calls: list = []

    def previous(
        exc_type: type, exc_value: BaseException, exc_traceback: Any
    ) -> None:
        """
        渡ったエラーを控える関数

        Parameters
        ----------
        exc_type: type
            エラーの型
        exc_value: BaseException
            起きたエラー
        exc_traceback: Any
            エラーの traceback

        Returns
        ----------
        None
        """
        calls.append(exc_value)

    # 元の sys.excepthook (エラーの表示) の代わりに控えるだけの関数を使用
    monkeypatch.setitem(STATE, "previous", previous)
    caught = raise_and_catch(error)
    handle_uncaught(type(caught), caught, caught.__traceback__)

    return calls


def read_files(run_dir: Path) -> tuple[Optional[dict], str]:
    """
    実験のディレクトリの run.json と train.log を読む関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ

    Returns
    ----------
    status: Optional[dict]
        run.json の内容, 無い場合は None
    log: str
        train.log の内容
    """
    path = run_dir / "run.json"
    status = None
    if path.exists():
        status = json.loads(path.read_text(encoding="utf-8"))
    log = (run_dir / "train.log").read_text(encoding="utf-8")

    return status, log


def test_error_before_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Trainer を作る前に処理されなかったエラーが, train.log と run.json に
    残ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        元の sys.excepthook の差し替えに使う

    Returns
    ----------
    None
    """
    # データの読み込みの失敗を想定したエラーは failed
    experiment = make_experiment(tmp_path / "error", "trainer:\n  epochs: 4\n")
    calls = call_hook(FileNotFoundError("data.csv がありません"), monkeypatch)
    status, log = read_files(experiment.run_dir)
    assert status["status"] == "failed"
    assert status["error"] == "FileNotFoundError: data.csv がありません"
    assert status["planned_epochs"] == 4
    assert status["completed_epochs"] == 0
    assert status["optimizers"] is None
    assert status["finished_at"] is not None
    assert "学習の外で処理されなかったエラーが起きました" in log
    assert "Traceback" in log and "raise_and_catch" in log

    # 記録した後も元の sys.excepthook でエラーを表示
    assert len(calls) == 1 and isinstance(calls[0], FileNotFoundError)

    # キーボードの割り込みは fit と同じく stopped
    experiment = make_experiment(tmp_path / "interrupt")
    call_hook(KeyboardInterrupt(), monkeypatch)
    status, log = read_files(experiment.run_dir)
    assert status["status"] == "stopped"
    assert status["error"] == "KeyboardInterrupt"
    assert "キーボードの割り込みで実験が止まりました" in log


def test_recorded_runs_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    fit と Trainer の作成が記録した実験と, 終えた実験を書き換えないことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        元の sys.excepthook の差し替えに使う

    Returns
    ----------
    None
    """
    # 終えた実験の後のエラー (テストデータの評価など) では書き換えない
    experiment = make_experiment(tmp_path / "completed")
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader(), make_loader())
    before = read_files(experiment.run_dir)
    calls = call_hook(RuntimeError("評価の失敗"), monkeypatch)
    assert read_files(experiment.run_dir) == before
    assert len(calls) == 1

    # Trainer の作成のエラーは 1 回だけ記録
    class BrokenModel(nn.Linear):
        # デバイスに移せないモデル
        def to(self, *args: Any, **kwargs: Any) -> nn.Module:
            """
            常にエラーにする関数

            Parameters
            ----------
            *args: Any
                移す先
            **kwargs: Any
                移す先

            Returns
            ----------
            model: nn.Module
                返さない
            """
            raise ValueError("このモデルは移せません")

    experiment = make_experiment(tmp_path / "init")
    with pytest.raises(ValueError):
        Trainer(BrokenModel(1, 1), experiment, regression_step)
    before = read_files(experiment.run_dir)
    assert before[0]["status"] == "failed"
    call_hook(ValueError("このモデルは移せません"), monkeypatch)
    assert read_files(experiment.run_dir) == before

    # fit が記録したエラーも書き換えない
    experiment = make_experiment(tmp_path / "fit")

    def broken_step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        常にエラーにする関数

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
        raise RuntimeError("壊れたバッチ")

    trainer = Trainer(nn.Linear(1, 1), experiment, broken_step)
    with pytest.raises(RuntimeError):
        trainer.fit(make_loader())
    before = read_files(experiment.run_dir)
    call_hook(RuntimeError("壊れたバッチ"), monkeypatch)
    assert read_files(experiment.run_dir) == before


def test_running_status_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    running のまま残った run.json (fit を使わない学習) を, 止まった状態に
    書き換えることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        元の sys.excepthook の差し替えに使う

    Returns
    ----------
    None
    """
    # 学習を始めた状態と進み具合を書いてからエラー
    experiment = make_experiment(tmp_path)
    status = RunStatus(experiment.run_dir)
    status.start(
        planned_epochs=5, start_epoch=1, monitor="valid/loss", mode="min",
        seed=0, device="cpu"
    )
    status.update(force=True, completed_epochs=2)
    status.progress("train", 3, 5, 1, 4, force=True)
    call_hook(ValueError("NaN になりました"), monkeypatch)

    # 学習の項目は残し, 終わった状態と所要時間とエラーを記録
    data, log = read_files(experiment.run_dir)
    assert data["status"] == "failed"
    assert data["completed_epochs"] == 2
    assert data["error"] == "ValueError: NaN になりました"
    assert data["duration_sec"] >= 0
    assert data["finished_at"] is not None
    assert not (experiment.run_dir / "progress.json").exists()
    assert "NaN になりました" in log


def test_hook_installed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    sys.excepthook の置き換えが 1 回だけで, 記録の対象は最後の実験になる
    ことを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        sys.excepthook と状態の差し替えに使う

    Returns
    ----------
    None
    """
    # 置き換える前の状態から, 2 回の実験の開始を想定
    def other_hook(*args: Any) -> None:
        """
        他のライブラリの sys.excepthook の代わりの関数

        Parameters
        ----------
        *args: Any
            エラーの型, エラー, traceback

        Returns
        ----------
        None
        """

    monkeypatch.setattr("sys.excepthook", other_hook)
    monkeypatch.setattr(
        uncaught, "STATE",
        {"experiment": None, "previous": None, "installed": False}
    )
    watch_uncaught_errors("first")
    watch_uncaught_errors("second")
    assert sys.excepthook is handle_uncaught
    assert uncaught.STATE["previous"] is other_hook
    assert uncaught.STATE["experiment"] == "second"


def test_uncaught_in_script(tmp_path: Path) -> None:
    """
    学習スクリプトで処理されなかったエラーが, 実際のプロセスでも記録され
    表示されることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # start_experiment の後にデータの読み込みで失敗するスクリプト
    (tmp_path / "default.yaml").write_text(
        "experiment:\n  name: test\n  device: cpu\n", encoding="utf-8"
    )
    (tmp_path / "experiment.yaml").write_text("", encoding="utf-8")
    script = tmp_path / "train.py"
    script.write_text(
        "from pathlib import Path\n"
        "from ml_found import load_project_config, start_experiment\n"
        "project_dir = Path(__file__).resolve().parent\n"
        "cfg, diff = load_project_config(\n"
        "    str(project_dir), default='default.yaml',\n"
        "    experiment='experiment.yaml'\n"
        ")\n"
        "start_experiment(cfg, diff, str(project_dir))\n"
        "raise FileNotFoundError('data.csv がありません')\n",
        encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, str(script), "trainer.epochs=2"], cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8"
    )

    # 終了コードと表示は Python の既定のまま
    assert result.returncode == 1
    assert "FileNotFoundError: data.csv がありません" in result.stderr

    # run.json, train.log, source.yaml に記録
    run_dir = next((tmp_path / "outputs" / "test").glob("*/*"))
    status, log = read_files(run_dir)
    assert status["status"] == "failed"
    assert status["planned_epochs"] == 2
    assert "data.csv がありません" in log
    source = (run_dir / "source.yaml").read_text(encoding="utf-8")
    assert "script: train.py" in source
    assert "trainer.epochs=2" in source
