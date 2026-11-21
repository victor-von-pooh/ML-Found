import hashlib
import os
from pathlib import Path
import shutil
import sys

import pytest
from omegaconf import OmegaConf

from ml_found.config.loader import load_project_config
from ml_found.experiment.model_spec import MODEL_SPEC_FILE, copy_model_spec
from ml_found.experiment.setup import start_experiment

# モデルの定義の見本 (中身は読まないので形は問わない)
SPEC_TEXT = "model:\n  layers:\n  - type: Linear\n    out: 3\n"


def make_project(project: Path, spec: str = "") -> Path:
    """
    config の model.spec に場所を書いたプロジェクトを作る関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ
    spec: str = ""
        default の config に書く model の YAML (空なら model を書かない)

    Returns
    ----------
    script: Path
        学習スクリプトのパス
    """
    # default, experiment, 学習スクリプトを用意
    default_dir = project / "configs" / "default"
    default_dir.mkdir(parents=True)
    (default_dir / "train.yaml").write_text(
        "experiment:\n  name: test\n  device: cpu\n  seed: 0\n" + spec,
        encoding="utf-8"
    )
    experiment_dir = project / "configs" / "experiment"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "train.yaml").write_text("", encoding="utf-8")
    script = project / "train.py"
    script.write_text("", encoding="utf-8")

    return script


def run_start(
    project: Path, monkeypatch: pytest.MonkeyPatch, spec: str = ""
) -> Path:
    """
    プロジェクトを作って start_experiment を呼ぶ関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ
    monkeypatch: pytest.MonkeyPatch
        sys.argv の差し替えに使う
    spec: str = ""
        default の config に書く model の YAML

    Returns
    ----------
    run_dir: Path
        作った実験のディレクトリ
    """
    script = make_project(project, spec)
    monkeypatch.setattr("sys.argv", [str(script)])
    monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
    cfg, diff = load_project_config(str(project))
    experiment = start_experiment(cfg, diff, str(project))
    run_dir = experiment.run_dir

    return run_dir


def read_record(run_dir: Path) -> dict:
    """
    source.yaml の model_spec を読む関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ

    Returns
    ----------
    record: dict
        source.yaml の model_spec
    """
    source = OmegaConf.to_container(OmegaConf.load(run_dir / "source.yaml"))
    record = source["model_spec"]

    return record


def spec_config(value: object, resume: object = None) -> object:
    """
    model.spec と trainer.resume_from だけを持つ config を作る関数

    Parameters
    ----------
    value: object
        model.spec の値
    resume: object = None
        trainer.resume_from の値

    Returns
    ----------
    cfg: object
        作った config (DictConfig)
    """
    cfg = OmegaConf.create(
        {"model": {"spec": value}, "trainer": {"resume_from": resume}}
    )

    return cfg


def test_copy_spec_on_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    start_experiment が定義を写し, 場所と sha256 を source.yaml に書く
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        sys.argv の差し替えに使う

    Returns
    ----------
    None
    """
    # 定義を置いたプロジェクト (定義のファイルは読み取りだけの権限)
    project = tmp_path / "project"
    (project / "models").mkdir(parents=True)
    spec = project / "models" / "model.yaml"
    spec.write_text(SPEC_TEXT, encoding="utf-8")
    spec.chmod(0o444)
    run_dir = run_start(
        project, monkeypatch, "model:\n  spec: models/model.yaml\n"
    )

    # 中身はそのまま, 記録は相対パスと中身の sha256
    copied = run_dir / MODEL_SPEC_FILE
    assert copied.read_bytes() == spec.read_bytes()
    assert read_record(run_dir) == {
        "path": "models/model.yaml", "status": "copied",
        "sha256": hashlib.sha256(SPEC_TEXT.encode()).hexdigest()
    }
    log = (run_dir / "train.log").read_text(encoding="utf-8")
    assert "model_spec.yaml に写しました" in log

    # 元の権限は写さないので, 学習のスクリプトが同じ名前で写し直せる
    assert os.access(copied, os.W_OK)
    shutil.copy(spec, copied)
    assert copied.read_text(encoding="utf-8") == SPEC_TEXT


@pytest.mark.parametrize(
    "spec", ["", "model:\n  hidden: 3\n", "model:\n  spec: null\n",
             "model:\n  spec: '  '\n", "model:\n  spec: 3\n"]
)
def test_no_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: str
) -> None:
    """
    model.spec が無い, 空, 文字でない場合は何も写さないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        sys.argv の差し替えに使う
    spec: str
        default の config に書く model の YAML

    Returns
    ----------
    None
    """
    run_dir = run_start(tmp_path / "project", monkeypatch, spec)
    assert not (run_dir / MODEL_SPEC_FILE).exists()
    assert read_record(run_dir) is None


def test_missing_spec_keeps_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    定義のファイルが無い場合は写さずに理由を残すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        sys.argv の差し替えに使う

    Returns
    ----------
    None
    """
    run_dir = run_start(
        tmp_path / "project", monkeypatch,
        "model:\n  spec: models/model.yaml\n"
    )
    assert not (run_dir / MODEL_SPEC_FILE).exists()
    assert read_record(run_dir) == {
        "path": "models/model.yaml", "status": "missing", "sha256": None
    }
    log = (run_dir / "train.log").read_text(encoding="utf-8")
    assert "写しませんでした: ファイルがありません" in log


def test_skip_outside_and_links(tmp_path: Path) -> None:
    """
    プロジェクトの外, リンク, 普通でないファイルを写さないことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # プロジェクトの外のファイル, 中へのリンク, 外へのリンクのディレクトリ
    project = tmp_path / "project"
    run_dir = tmp_path / "run"
    (project / "models").mkdir(parents=True)
    run_dir.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text(SPEC_TEXT, encoding="utf-8")
    (project / "models" / "real.yaml").write_text(SPEC_TEXT, encoding="utf-8")
    (project / "models" / "link.yaml").symlink_to(
        project / "models" / "real.yaml"
    )
    (project / "linked").symlink_to(tmp_path, target_is_directory=True)
    (project / "models" / "folder").mkdir()

    # 場所ごとの記録 (外の場所は絶対パス, リンクは辿らずに書く)
    cases = {
        "../outside.yaml": ("outside", outside.resolve().as_posix()),
        str(outside): ("outside", outside.resolve().as_posix()),
        "models/link.yaml": ("link", "models/link.yaml"),
        "linked/outside.yaml": ("link", "linked/outside.yaml"),
        "models/folder": ("not_file", "models/folder"),
        "models/../models/missing.yaml": ("missing", "models/missing.yaml")
    }
    for value, (status, path) in cases.items():
        record = copy_model_spec(spec_config(value), run_dir, project)
        assert record == {"path": path, "status": status, "sha256": None}
    assert not (run_dir / MODEL_SPEC_FILE).exists()

    # プロジェクトの中の絶対パスは写し, 相対パスで記録
    record = copy_model_spec(
        spec_config(str(project / "models" / "real.yaml")), run_dir, project
    )
    assert record["status"] == "copied"
    assert record["path"] == "models/real.yaml"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO を作れない")
def test_skip_fifo(tmp_path: Path) -> None:
    """
    FIFO の定義の場所で止まらずに写さないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    project = tmp_path / "project"
    project.mkdir()
    os.mkfifo(project / "model.yaml")
    record = copy_model_spec(spec_config("model.yaml"), tmp_path, project)
    assert record == {
        "path": "model.yaml", "status": "not_file", "sha256": None
    }


def test_resumed_does_not_copy(tmp_path: Path) -> None:
    """
    再開する学習では定義を写し直さないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    project = tmp_path / "project"
    run_dir = tmp_path / "run"
    project.mkdir()
    run_dir.mkdir()
    (project / "model.yaml").write_text(SPEC_TEXT, encoding="utf-8")
    record = copy_model_spec(
        spec_config("model.yaml", "outputs/last.pt"), run_dir, project
    )
    assert record == {
        "path": "model.yaml", "status": "resumed", "sha256": None
    }
    assert not (run_dir / MODEL_SPEC_FILE).exists()
