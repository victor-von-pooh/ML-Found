import gc
from pathlib import Path
import sys

import pytest
from omegaconf import OmegaConf

from ml_found.config.loader import load_config, load_project_config
from ml_found.config.source import SOURCES, save_source
from ml_found.experiment.setup import start_experiment


def make_project(project: Path) -> Path:
    """
    default と experiment の config と学習スクリプトを持つプロジェクトを作る関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    script: Path
        学習スクリプトのパス
    """
    # default, experiment, 学習スクリプトを用意
    default_dir = project / "configs" / "default"
    default_dir.mkdir(parents=True)
    (default_dir / "train.yaml").write_text(
        "experiment:\n  name: test\n  device: cpu\n  seed: 0\n",
        encoding="utf-8"
    )
    experiment_dir = project / "configs" / "experiment"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "train.yaml").write_text("", encoding="utf-8")
    (experiment_dir / "short.yaml").write_text(
        "trainer:\n  epochs: 5\n", encoding="utf-8"
    )
    script = project / "train.py"
    script.write_text("", encoding="utf-8")

    return script


def read_source(run_dir: Path) -> dict:
    """
    実験のディレクトリの source.yaml を読む関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ

    Returns
    ----------
    source: dict
        source.yaml の内容
    """
    source = OmegaConf.to_container(OmegaConf.load(run_dir / "source.yaml"))

    return source


def test_source_from_project_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    load_project_config で読んだ config の起動のしかたが source.yaml に
    残ることを確かめる関数

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
    # 実行した場所がプロジェクトの外でも, パスはプロジェクトからの相対パス
    project = tmp_path / "project"
    script = make_project(project)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [str(script), "--experiment", "configs/experiment/short.yaml",
         "trainer.epochs=3", "experiment.seed=7"]
    )
    monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
    cfg, diff = load_project_config(str(project))
    experiment = start_experiment(cfg, diff, str(project))

    source = read_source(experiment.run_dir)
    assert source == {
        "format_version": 1,
        "script": "train.py",
        "default": "configs/default/train.yaml",
        "experiment": "configs/experiment/short.yaml",
        "overrides": ["trainer.epochs=3", "experiment.seed=7"],
        "model_spec": None
    }

    # 上書きの値の ${...} も, 補間にせず書いた文字のまま残す
    cfg, diff = load_project_config(
        str(project), ["experiment.name=${experiment.device}"]
    )
    experiment = start_experiment(cfg, diff, str(project))
    source = read_source(experiment.run_dir)
    assert source["experiment"] == "configs/experiment/train.yaml"
    assert source["overrides"] == ["experiment.name=${experiment.device}"]


def test_source_unknown_and_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    読み込み関数を通らない config とプロジェクトの外のパス, ノートブックの
    中の source.yaml を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        sys.argv と sys.modules の差し替えに使う

    Returns
    ----------
    None
    """
    # プロジェクトの外の experiment は絶対パス, ファイルの無い argv[0]
    # (python -c など) のスクリプトは null
    project = tmp_path / "project"
    make_project(project)
    outside = tmp_path / "outside.yaml"
    outside.write_text("", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["-c"])
    cfg, diff = load_config(
        str(project / "configs" / "default" / "train.yaml"), str(outside)
    )
    experiment = start_experiment(cfg, diff, str(project))
    source = read_source(experiment.run_dir)
    assert source["script"] is None
    assert source["default"] == "configs/default/train.yaml"
    assert source["experiment"] == outside.resolve().as_posix()
    assert source["overrides"] == []

    # 写した config は, どのファイルから作ったか分からないので null
    # ノートブックの中ではスクリプトも null
    monkeypatch.setitem(sys.modules, "ipykernel", object())
    copied = OmegaConf.merge(cfg)
    experiment = start_experiment(copied, diff, str(project))
    assert read_source(experiment.run_dir) == {
        "format_version": 1, "script": None, "default": None,
        "experiment": None, "overrides": None, "model_spec": None
    }


def test_source_keeps_existing(tmp_path: Path) -> None:
    """
    既にある source.yaml の情報を, 情報の少ない書き込みで消さないことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 分からない項目 (null) だけを今回の値で埋める
    (tmp_path / "source.yaml").write_text(
        "script: train.py\ndefault: configs/default/train.yaml\n"
        "experiment: null\noverrides:\n- trainer.epochs=3\n",
        encoding="utf-8"
    )
    cfg = OmegaConf.create({"trainer": {"epochs": 3}})
    save_source(cfg, str(tmp_path), tmp_path)
    assert read_source(tmp_path) == {
        "format_version": 1, "script": "train.py",
        "default": "configs/default/train.yaml",
        "experiment": None, "overrides": ["trainer.epochs=3"],
        "model_spec": None
    }


def test_source_registry_released(tmp_path: Path) -> None:
    """
    config が消えると, 覚えた起動のしかたも消えることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 読み込んだ config の id で覚え, config が消えたら削除
    make_project(tmp_path)
    cfg, _ = load_config(
        str(tmp_path / "configs" / "default" / "train.yaml"),
        str(tmp_path / "configs" / "experiment" / "train.yaml")
    )
    key = id(cfg)
    assert key in SOURCES
    del cfg
    gc.collect()
    assert key not in SOURCES
