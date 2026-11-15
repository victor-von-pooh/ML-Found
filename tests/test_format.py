import csv
import json
from pathlib import Path
import re
import sys

import pytest
from omegaconf import OmegaConf
from torch import nn

from ml_found.config.loader import load_project_config
from ml_found.config.source import SOURCE_FORMAT_VERSION
from ml_found.experiment.setup import start_experiment
from ml_found.training.status import FORMAT_VERSION
from ml_found.training.trainer import Trainer
from test_trainer import make_loader, regression_step

# 形の約束を書いた文書
FORMAT_DOC = Path(__file__).resolve().parents[1] / "docs" / "FORMAT.md"


def documented_keys(section: str) -> set[str]:
    """
    docs/FORMAT.md の節の表に書いた項目の名前を集める関数

    Parameters
    ----------
    section: str
        節の見出し (`### run.json` など)

    Returns
    ----------
    keys: set[str]
        表の 1 列目に `` で囲んで書いた名前
    """
    # 見出しから次の見出しまでの表の行の 1 列目を読む
    text = FORMAT_DOC.read_text(encoding="utf-8")
    body = text.split(section + "\n", 1)[1].split("\n#", 1)[0]
    keys = set()
    for line in body.splitlines():
        if not line.startswith("| `"):
            continue
        first = line.split("|")[1]
        keys.update(re.findall(r"`([^`]+)`", first))

    return keys


def run_project(project: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    文書のとおりのプロジェクトを作り, 2 エポックの学習を流す関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ
    monkeypatch: pytest.MonkeyPatch
        sys.argv の差し替えに使う

    Returns
    ----------
    run_dir: Path
        学習した実験のディレクトリ
    """
    # 必須の default と既定の experiment, 学習のスクリプトを用意
    (project / "configs" / "default").mkdir(parents=True)
    (project / "configs" / "experiment").mkdir(parents=True)
    (project / "configs" / "default" / "train.yaml").write_text(
        "experiment:\n  name: contract\n  device: cpu\n  seed: 0\n",
        encoding="utf-8"
    )
    (project / "configs" / "experiment" / "train.yaml").write_text(
        "", encoding="utf-8"
    )
    script = project / "train.py"
    script.write_text("", encoding="utf-8")

    # 文書の起動のしかた (スクリプトと「項目=値」) で学習
    monkeypatch.setattr(
        "sys.argv",
        [str(script), "trainer.epochs=2", "trainer.progress_bar=false"]
    )
    monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
    cfg, diff = load_project_config(str(project))
    experiment = start_experiment(cfg, diff, str(project))
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader(), make_loader())

    return experiment.run_dir


def test_run_directory_matches_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    実験のディレクトリが docs/FORMAT.md の形と合うことを確かめる関数

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
    project = tmp_path / "project"
    run_dir = run_project(project, monkeypatch)

    # <output_dir>/<name>/<日付>/<時刻>/ の場所
    relative = run_dir.relative_to(project).as_posix()
    assert re.fullmatch(
        r"outputs/contract/\d{4}-\d{2}-\d{2}/\d{2}-\d{2}-\d{2}(_\d+)?",
        relative
    )

    # 学習を終えた実験に必ずあるファイルと, 学習中だけのファイル
    for name in (
        "config.yaml", "diff.yaml", "source.yaml", "train.log",
        "run.json", "history.csv", "model_summary.csv", "final_model.pt"
    ):
        assert (run_dir / name).is_file(), name
    assert not (run_dir / "progress.json").exists()

    # source.yaml の項目と版
    source = OmegaConf.to_container(OmegaConf.load(run_dir / "source.yaml"))
    assert set(source) == documented_keys("### source.yaml")
    assert source["format_version"] == SOURCE_FORMAT_VERSION == 1
    assert source["script"] == "train.py"
    assert source["default"] == "configs/default/train.yaml"

    # run.json の項目と版
    status = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert set(status) == documented_keys("### run.json")
    assert status["format_version"] == FORMAT_VERSION == 1
    assert status["status"] == "completed"

    # history.csv は epoch の列と, 接頭辞の付いた指標の列
    with open(run_dir / "history.csv", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["epoch"] for row in rows] == ["1", "2"]
    for column in rows[0]:
        assert column in ("epoch", "lr") or re.match(
            r"(train|valid|train_eval)/", column
        ), column

    # diff.yaml は . でつないだ名前ごとの, default と変えた後の値
    diff = OmegaConf.to_container(OmegaConf.load(run_dir / "diff.yaml"))
    assert diff["trainer.epochs"] == {"default": 100, "experiment": 2}
