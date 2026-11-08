from datetime import datetime
from pathlib import Path

from ml_found.config.loader import load_config
from ml_found.experiment.setup import make_run_dir, start_experiment


def test_make_run_dir_no_overwrite(tmp_path: Path):
    """
    同じ日時で作っても別のディレクトリになることを確かめる関数
    """
    # 同じ日時で 3 回作る
    now = datetime(2026, 11, 8, 12, 34, 56)
    dirs = [make_run_dir(tmp_path, "outputs", "exp", now) for _ in range(3)]

    assert dirs[0] == tmp_path / "outputs/exp/2026-11-08/12-34-56"
    assert dirs[1].name == "12-34-56_1"
    assert dirs[2].name == "12-34-56_2"


def test_start_experiment(tmp_path: Path):
    """
    実験の開始で保存先とログと config が用意されることを確かめる関数
    """
    # default と experiment を用意して読み込む
    default_path = tmp_path / "default.yaml"
    default_path.write_text(
        "experiment:\n  name: sample\n  device: cpu\n", encoding="utf-8"
    )
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text("trainer:\n  epochs: 3\n", encoding="utf-8")
    cfg, diff = load_config(str(default_path), str(experiment_path))

    # 実験を開始する
    experiment = start_experiment(cfg, diff, str(tmp_path))
    log_text = (experiment.run_dir / "train.log").read_text(encoding="utf-8")

    assert experiment.run_dir.parent.parent == tmp_path / "outputs/sample"
    assert (experiment.run_dir / "config.yaml").exists()
    assert (experiment.run_dir / "diff.yaml").exists()
    assert experiment.device.type == "cpu"
    assert "trainer.epochs: 100 -> 3" in log_text
    assert "package list" in log_text
