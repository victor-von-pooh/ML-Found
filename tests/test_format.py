import csv
import json
from pathlib import Path
import re
import sys
from typing import Callable

import pytest
from omegaconf import OmegaConf
from torch import nn

from ml_found.config.loader import load_project_config
from ml_found.config.source import SOURCE_FORMAT_VERSION
from ml_found.experiment.results import (
    SAMPLES_FORMAT_VERSION, load_evaluations, load_samples, save_evaluation,
    save_sample
)
from ml_found.experiment.setup import start_experiment
from ml_found.experiment.task import TASK_FORMAT_VERSION, load_task
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
    # 見出しから次の見出しまでの, 表の行の 1 列目の読み込み
    text = FORMAT_DOC.read_text(encoding="utf-8")
    body = text.split(section + "\n", 1)[1].split("\n#", 1)[0]
    keys = set()
    for line in body.splitlines():
        if not line.startswith("| `"):
            continue
        first = line.split("|")[1]
        keys.update(re.findall(r"`([^`]+)`", first))

    return keys


def run_project(
    project: Path, monkeypatch: pytest.MonkeyPatch,
    step: Callable = regression_step,
    extra: str = ""
) -> Path:
    """
    文書のとおりのプロジェクトを作り, 2 エポックの学習を流す関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ
    monkeypatch: pytest.MonkeyPatch
        sys.argv の差し替えに使う
    step: Callable = regression_step
        1 バッチの計算
    extra: str = ""
        default に加える YAML の内容

    Returns
    ----------
    run_dir: Path
        学習した実験のディレクトリ
    """
    # 必須の default と既定の experiment, 学習のスクリプトを用意
    (project / "configs" / "default").mkdir(parents=True)
    (project / "configs" / "experiment").mkdir(parents=True)
    (project / "configs" / "default" / "train.yaml").write_text(
        "experiment:\n  name: contract\n  device: cpu\n  seed: 0\n"
        "trainer:\n  epochs: 3\n" + extra,
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
    trainer = Trainer(nn.Linear(1, 1), experiment, step)
    try:
        trainer.fit(make_loader(), make_loader())
    except RuntimeError:
        pass

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
    assert source["model_spec"] is None

    # run.json の項目と版
    status = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert set(status) == documented_keys("### run.json")
    assert status["format_version"] == FORMAT_VERSION == 1
    assert status["status"] == "completed"

    # history.csv は epoch, lr と, 接頭辞の付いた指標の列
    with open(run_dir / "history.csv", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["epoch"] for row in rows] == ["1", "2"]
    for column in rows[0]:
        assert column in ("epoch", "lr") or re.match(
            r"(train|valid|train_eval)/", column
        ), column

    # model_summary.csv の列
    with open(run_dir / "model_summary.csv", encoding="utf-8") as file:
        header = next(csv.reader(file))
    assert header == [
        "name", "kind", "module", "shape", "dtype", "numel",
        "requires_grad", "same_as"
    ]

    # diff.yaml は . でつないだ名前ごとの, default と変えた後の値
    diff = OmegaConf.to_container(OmegaConf.load(run_dir / "diff.yaml"))
    assert diff["trainer.epochs"] == {"default": 3, "experiment": 2}

    # 書かなかった任意のファイルは無く, 読む側では分からないものの扱い
    for name in ("task.yaml", "evaluations", "samples"):
        assert not (run_dir / name).exists(), name
    assert load_task(run_dir) is None
    assert load_evaluations(run_dir) == []
    assert load_samples(run_dir) == []


def test_history_with_valid_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    検証の間隔を空けた history.csv が, 検証しないエポックの valid/ を空に
    する docs/FORMAT.md の形と合うことを確かめる関数

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
    # 2 エポックで間隔 2 (エポック 2 だけ検証)
    run_dir = run_project(
        tmp_path / "project", monkeypatch,
        extra="  valid_every_n_epochs: 2\n"
    )

    # エポック 1 の valid/ は空, エポック 2 は値あり. 列の名前は同じ決まり
    with open(run_dir / "history.csv", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["epoch"] for row in rows] == ["1", "2"]
    assert rows[0]["valid/loss"] == ""
    assert float(rows[1]["valid/loss"]) >= 0.0
    for column in rows[0]:
        assert column in ("epoch", "lr") or re.match(
            r"(train|valid|train_eval)/", column
        ), column


def test_failed_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    学習の途中のエラーで止まった実験のファイルが, docs/FORMAT.md の
    「いつあるか」と合うことを確かめる関数

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
    # 2 エポック目の学習でエラーを出す 1 バッチの計算
    calls = {"train": 0}

    def failing_step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        5 回目の学習のバッチでエラーを出す 1 バッチの計算

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
            平均二乗誤差と平均絶対誤差
        """
        # 1 エポックは 4 バッチなので, 2 エポック目の最初のバッチでエラー
        if phase == "train":
            calls["train"] += 1
            if calls["train"] > 4:
                raise RuntimeError("テストのエラー")
        outputs = regression_step(model, batch, phase)

        return outputs

    project = tmp_path / "project"
    run_dir = run_project(
        project, monkeypatch, failing_step,
        "  select_best: false\n"
    )

    # 状態は failed, ベストモデルを選ばない設定のベストは null
    status = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["select_best"] is False
    assert status["best_epoch"] is None
    assert status["completed_epochs"] == 1

    # 始めた後のファイルはあり, fit を終えたときに書くファイルは無し
    for name in (
        "config.yaml", "diff.yaml", "source.yaml", "train.log",
        "model_summary.csv"
    ):
        assert (run_dir / name).is_file(), name
    for name in ("history.csv", "final_model.pt", "progress.json"):
        assert not (run_dir / name).exists(), name


def test_optional_files_match_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    任意のファイル (task.yaml, evaluations, samples) が docs/FORMAT.md の
    形と合うことを確かめる関数

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
    # ML-EDA の版 2 の説明を置き, config の task からの読み込み
    dataset = tmp_path / "project" / "outputs" / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "dataset.yaml").write_text(
        "format_version: 2\nname: pairs\nrole: predict\n"
        "target:\n  kind: mask\n  entries: [mask]\n"
        "  origin:\n    type: process\n    name: thresholding\n"
        "    settings: {level: 0.5}\n",
        encoding="utf-8"
    )
    run_dir = run_project(
        tmp_path / "project", monkeypatch,
        extra="task:\n  dataset: outputs/dataset\n  method: scratch\n"
    )

    # task.yaml の項目と版. 全ての項目を書き, 欄も文書と同じ
    task = load_task(run_dir)
    keys = documented_keys("### task.yaml")
    assert set(task) <= keys
    assert set(task) == keys - {"type", "name", "settings"}
    assert set(task["target"]["origin"]) == {"type", "name", "settings"}
    assert set(task["base"]) == {"experiment", "weights"}
    assert set(task["dataset"]) == {"path", "name", "format_version"}
    assert task["format_version"] == TASK_FORMAT_VERSION == 1
    assert task["dataset"]["path"] == "outputs/dataset"

    # evaluations/<名前>.json の項目
    save_evaluation(run_dir, "per_group", {"group": ["a"], "iou": [0.5]},
                    "test")
    evaluation = json.loads(
        (run_dir / "evaluations" / "per_group.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(evaluation) == documented_keys("### evaluations")

    # samples/index.json の項目と置き場所
    save_sample(run_dir, "note", "見本の文", epoch=5)
    index = json.loads(
        (run_dir / "samples" / "index.json").read_text(encoding="utf-8")
    )
    assert list(index) == ["format_version", "samples"]
    assert index["format_version"] == SAMPLES_FORMAT_VERSION == 2
    assert set(index["samples"][0]) == documented_keys("### samples")
    assert (run_dir / "samples" / "note" / "epoch_0005.txt").is_file()
