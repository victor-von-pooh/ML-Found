from datetime import datetime
from pathlib import Path
import subprocess

import pytest
import torch

from ml_found.config.loader import load_config
from ml_found.experiment.env_info import get_ml_found_info
from ml_found.experiment.setup import (
    Experiment, make_run_dir, start_experiment
)


def run_experiment(
    project_dir: Path, default_text: str, experiment_text: str = ""
) -> Experiment:
    """
    テスト用の config を書き出して実験を開始する関数

    Parameters
    ----------
    project_dir: Path
        プロジェクトのディレクトリ
    default_text: str
        default の YAML の内容
    experiment_text: str = ""
        experiment の YAML の内容

    Returns
    -------
    experiment: Experiment
        開始した実験
    """
    # default と experiment を書き出して読み込む
    default_path = project_dir / "default.yaml"
    default_path.write_text(default_text, encoding="utf-8")
    experiment_path = project_dir / "experiment.yaml"
    experiment_path.write_text(experiment_text, encoding="utf-8")
    cfg, diff = load_config(str(default_path), str(experiment_path))

    # 実験を開始する
    experiment = start_experiment(cfg, diff, str(project_dir))

    return experiment


def git(args: list, repo_dir: Path) -> None:
    """
    テスト用のリポジトリで git コマンドを実行する関数

    Parameters
    ----------
    args: list
        git に渡す引数
    repo_dir: Path
        リポジトリのディレクトリ

    Returns
    -------
    None
    """
    # 環境に依存しないようにユーザー名とメールアドレスを指定して実行する
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
         *args],
        cwd=repo_dir, check=True, capture_output=True
    )


def test_make_run_dir_no_overwrite(tmp_path: Path) -> None:
    """
    同じ日時で作っても別のディレクトリになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    -------
    None
    """
    # 同じ日時で 3 回作る
    now = datetime(2026, 11, 8, 12, 34, 56)
    dirs = [make_run_dir(tmp_path, "outputs", "exp", now) for _ in range(3)]

    assert dirs[0] == tmp_path / "outputs/exp/2026-11-08/12-34-56"
    assert dirs[1].name == "12-34-56_1"
    assert dirs[2].name == "12-34-56_2"


@pytest.mark.parametrize("name", ["", "..", "a/b", "../escape"])
def test_make_run_dir_invalid_name(tmp_path: Path, name: str) -> None:
    """
    ディレクトリ名として使えない実験の名前でエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    name: str
        正しくない実験の名前

    Returns
    -------
    None
    """
    with pytest.raises(ValueError, match="experiment.name"):
        make_run_dir(tmp_path, "outputs", name, datetime.now())


def test_start_experiment(tmp_path: Path) -> None:
    """
    実験の開始で保存先とログと config が用意されることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    -------
    None
    """
    # 実験を開始する
    experiment = run_experiment(
        tmp_path, "experiment:\n  name: sample\n  device: cpu\n",
        "trainer:\n  epochs: 3\n"
    )
    log_text = (experiment.run_dir / "train.log").read_text(encoding="utf-8")

    assert experiment.run_dir.parent.parent == tmp_path / "outputs/sample"
    assert (experiment.run_dir / "config.yaml").exists()
    assert (experiment.run_dir / "diff.yaml").exists()
    assert experiment.device.type == "cpu"
    assert "trainer.epochs: 100 -> 3" in log_text
    assert "package list" in log_text


def test_git_info_before_run_dir(tmp_path: Path) -> None:
    """
    出力先を作る前に git の情報を集め, 変更がある場合は差分を保存することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    -------
    None
    """
    # outputs を無視しないリポジトリを作り, config を commit する
    default_text = "experiment:\n  device: cpu\n"
    (tmp_path / "default.yaml").write_text(default_text, encoding="utf-8")
    (tmp_path / "experiment.yaml").write_text("", encoding="utf-8")
    git(["init"], tmp_path)
    git(["add", "."], tmp_path)
    git(["commit", "-m", "init"], tmp_path)

    # 変更が無い状態で実験を開始する
    experiment = run_experiment(tmp_path, default_text)
    log_text = (experiment.run_dir / "train.log").read_text(encoding="utf-8")

    assert "uncommitted changes: 0 files" in log_text
    assert not (experiment.run_dir / "git_diff.patch").exists()

    # commit 済みのファイルを変更して実験を開始する
    experiment = run_experiment(tmp_path, default_text + "# changed\n")
    patch = (experiment.run_dir / "git_diff.patch").read_text(
        encoding="utf-8"
    )

    assert "+# changed" in patch


def test_invalid_device(tmp_path: Path) -> None:
    """
    使えないデバイスを指定すると出力先を作らずにエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    -------
    None
    """
    # この環境で使えないデバイスを選ぶ
    device = "cpuu" if torch.cuda.is_available() else "cuda"

    with pytest.raises(ValueError):
        run_experiment(tmp_path, f"experiment:\n  device: {device}\n")
    assert not (tmp_path / "outputs").exists()


def test_ml_found_info() -> None:
    """
    git 管理された ML-Found の場合に commit id が取得できることを確かめる関数

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # ML-Found のバージョンの情報を取得する
    info = get_ml_found_info()

    assert "commit id:" in info
