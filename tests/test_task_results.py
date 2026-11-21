import json
from pathlib import Path
import re
import threading
import wave

from matplotlib import image as mpimg
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from ml_found import (
    describe_origin, load_evaluations, load_samples, load_task,
    original_coordinates, save_evaluation, save_sample, task_from_dataset,
    write_task
)
from ml_found.experiment.results import spread_better
from ml_found.experiment.setup import Experiment
from ml_found.experiment.task import TASK_FILE
from ml_found.training.trainer import Trainer
from test_trainer import make_experiment, make_loader, regression_step


def write_dataset(folder: Path, version: object = 2) -> Path:
    """
    ML-EDA の書き出しの形の dataset.yaml を置く関数

    Parameters
    ----------
    folder: Path
        データセットのディレクトリ
    version: object = 2
        format_version に書く値

    Returns
    ----------
    folder: Path
        データセットのディレクトリ
    """
    # 正解が今ある処理の出力の, 撮影と点群の対の説明
    folder.mkdir(parents=True, exist_ok=True)
    info = {
        "format_version": version,
        "name": "clouds",
        "role": "predict",
        "target": {
            "kind": "point_cloud",
            "entries": ["cloud"],
            "origin": {
                "type": "process",
                "name": "reconstruct",
                "settings": {"voxel": 0.01, "note": "${not_interpolated}"}
            }
        }
    }
    OmegaConf.save(OmegaConf.create(info), folder / "dataset.yaml")

    return folder


def fit_briefly(experiment: Experiment) -> None:
    """
    1 エポックだけ学習する関数

    Parameters
    ----------
    experiment: Experiment
        開始した実験

    Returns
    ----------
    None
    """
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader(), make_loader())


def test_task_from_dataset(tmp_path: Path) -> None:
    """
    dataset.yaml の版 2 から受け持ちと正解の出どころを写すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # ディレクトリでも dataset.yaml でも同じ中身. 補間の文字はそのまま
    folder = write_dataset(tmp_path / "data")
    task = task_from_dataset(folder)
    assert task == task_from_dataset(folder / "dataset.yaml")
    assert task["role"] == "predict"
    assert task["target"]["kind"] == "point_cloud"
    assert task["target"]["origin"]["type"] == "process"
    assert task["target"]["origin"]["settings"]["note"] == (
        "${not_interpolated}"
    )
    assert task["dataset"]["name"] == "clouds"
    assert task["dataset"]["format_version"] == 2

    # 相対パスは渡したディレクトリからの辿り
    relative = task_from_dataset("data", project_dir=str(tmp_path))
    assert relative == task


def test_task_from_dataset_version_3(tmp_path: Path) -> None:
    """
    中身の手順を持つ dataset.yaml の版 3 からも同じ形で写し, 学習すると
    task.yaml に版 3 を書くことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 版 2 と版の他は同じ中身
    folder = write_dataset(tmp_path / "data", 3)
    task = task_from_dataset(folder)
    other = task_from_dataset(write_dataset(tmp_path / "other", 2))
    assert task["dataset"]["format_version"] == 3
    assert task["role"] == other["role"]
    assert task["target"] == other["target"]

    # config の task から書いた task.yaml
    experiment = make_experiment(tmp_path, "task:\n  dataset: data\n")
    written = load_task(experiment.run_dir)
    assert written["target"]["kind"] == "point_cloud"
    assert written["target"]["origin"]["type"] == "process"
    assert written["dataset"] == {
        "path": "data", "name": "clouds", "format_version": 3
    }
    fit_briefly(experiment)
    assert load_task(experiment.run_dir) == written


@pytest.mark.parametrize(
    ("version", "message"),
    [(1, "版 1"), (4, "読めません"), (None, "読めません"),
     ("2", "読めません"), (2.0, "読めません"), (True, "読めません")]
)
def test_task_from_dataset_versions(
    tmp_path: Path, version: object, message: str
) -> None:
    """
    dataset.yaml の版が整数の 2 か 3 でなければはっきり止まることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    version: object
        format_version に書く値
    message: str
        誤りのメッセージに含まれる文字

    Returns
    ----------
    None
    """
    folder = write_dataset(tmp_path / "data", version)
    with pytest.raises(ValueError, match=message):
        task_from_dataset(folder)


def test_task_from_dataset_missing(tmp_path: Path) -> None:
    """
    dataset.yaml が無い場合に止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    with pytest.raises(FileNotFoundError, match="dataset.yaml"):
        task_from_dataset(tmp_path)


def test_config_task_writes_task_yaml(tmp_path: Path) -> None:
    """
    config の task から task.yaml を書き, 書いた項目が説明より優先される
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # データセットのパスと, 出力の意味, 正解の種類の上書き
    write_dataset(tmp_path / "data")
    experiment = make_experiment(
        tmp_path,
        "task:\n"
        "  dataset: data\n"
        "  output: shape\n"
        "  what: 撮影ごとの物の表面の点\n"
        "  target:\n"
        "    kind: dense_cloud\n"
        "  method: scratch\n"
    )
    task = load_task(experiment.run_dir)
    assert task["role"] == "predict"
    assert task["output"] == "shape"
    assert task["what"] == "撮影ごとの物の表面の点"
    assert task["target"]["kind"] == "dense_cloud"
    assert task["target"]["origin"]["name"] == "reconstruct"
    assert task["method"] == "scratch"
    assert task["base"] == {"experiment": None, "weights": None}
    assert task["dataset"]["path"] == "data"

    # train.log に指標が何との一致かを記録
    log = (experiment.run_dir / "train.log").read_text(encoding="utf-8")
    assert "超えられない" in log

    # 学習しても task.yaml はそのまま
    fit_briefly(experiment)
    assert load_task(experiment.run_dir) == task


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("task:\n  role: predict\n  wat: x\n", "知らない項目"),
        ("task:\n  role: classify\n", "task.role"),
        ("task:\n  method: scratch\n  base:\n    weights: bert\n",
         "scratch"),
        ("task:\n  target:\n    origin:\n      type: guess\n",
         "origin.type"),
        ("task:\n  dataset: missing\n", "dataset.yaml"),
    ]
)
def test_config_task_errors_before_run_dir(
    tmp_path: Path, text: str, message: str
) -> None:
    """
    config の task の誤りは出力先を作る前に見つかることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    text: str
        default に加える YAML
    message: str
        誤りのメッセージに含まれる文字

    Returns
    ----------
    None
    """
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        make_experiment(tmp_path, text)
    assert not (tmp_path / "outputs").exists()


def test_config_task_not_mapping_is_ignored(tmp_path: Path) -> None:
    """
    辞書でない task (プロジェクトが別の意味で使う項目) では task.yaml を
    書かないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path, "task: classification\n")
    assert not (experiment.run_dir / TASK_FILE).exists()


def test_write_task_finetune(tmp_path: Path) -> None:
    """
    write_task が元にした実験をプロジェクトからの相対パスで書き,
    後から書いた方で置き換えることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 元にする実験と, それを微調整する実験
    first = make_experiment(tmp_path)
    second = make_experiment(tmp_path)
    task = write_task(
        second,
        {"role": "predict", "method": "finetune",
         "base": {"experiment": str(first.run_dir), "weights": "bert-base"}},
        dataset=write_dataset(tmp_path / "data")
    )
    expected = first.run_dir.relative_to(tmp_path).as_posix()
    assert task["base"] == {"experiment": expected, "weights": "bert-base"}
    assert task["target"]["origin"]["type"] == "process"
    assert load_task(second.run_dir) == task

    # task_from_dataset の結果に足して渡した場合も同じ結果
    filled = task_from_dataset(tmp_path / "data")
    filled["what"] = "物の表面の点"
    again = write_task(second, filled)
    assert again["what"] == "物の表面の点"
    assert again["base"]["experiment"] is None
    assert load_task(second.run_dir)["what"] == "物の表面の点"


def test_write_task_warns_without_base(tmp_path: Path) -> None:
    """
    微調整なのに元が書かれていない場合に警告することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    with pytest.warns(UserWarning, match="元にした"):
        task = write_task(experiment, {"method": "finetune"})
    assert task["target"]["origin"] == {
        "type": "unknown", "name": None, "settings": None
    }


def test_load_task_newer_version(tmp_path: Path) -> None:
    """
    新しい版の task.yaml は警告して読める項目を返すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 処理の出力を正解とした新しい版. 出どころは分からないものに置き換え
    (tmp_path / TASK_FILE).write_text(
        "format_version: 2\nrole: predict\nnew_field: 1\n"
        "target:\n  kind: mask\n  origin:\n    type: process\n"
        "    name: reconstruct\n",
        encoding="utf-8"
    )
    with pytest.warns(UserWarning, match="分からないもの"):
        task = load_task(tmp_path)
    assert task["role"] == "predict"
    assert task["newer_version"] is True
    assert task["target"]["kind"] == "mask"
    assert task["target"]["origin"]["type"] == "unknown"
    note = describe_origin(task["target"]["origin"])
    assert "超えられない" not in note
    newer = describe_origin(
        {"type": "process", "name": "reconstruct"}, format_version=2
    )
    assert "超えられない" not in newer and "新しい版" in newer

    # 壊れたファイルと辞書でないファイルは分からないもの
    (tmp_path / TASK_FILE).write_text("- 1\n", encoding="utf-8")
    assert load_task(tmp_path) is None


def test_describe_origin() -> None:
    """
    正解の出どころの説明が種類ごとに変わることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    note = describe_origin({"type": "process", "name": "reconstruct"})
    assert "reconstruct" in note and "超えられない" in note
    assert "付けた人" in describe_origin({"type": "annotation"})
    assert "分からない" in describe_origin(None)
    assert "分からない" in describe_origin({"type": "future_type"})
    assert "環境の報酬" in describe_origin({"type": "environment"})


def test_save_evaluation_forms(tmp_path: Path) -> None:
    """
    いろいろな形の表を同じ形の評価に書くことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)

    # 列の辞書 (テンソルの値も可), nan は null
    path = save_evaluation(
        experiment, "per_group",
        {"group": ["a", "b"], "iou": torch.tensor([0.5, float("nan")])},
        "test", description="グループごとの IoU", guard=True, epoch=3
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["columns"] == ["group", "iou"]
    assert data["rows"] == [
        {"group": "a", "iou": 0.5}, {"group": "b", "iou": None}
    ]
    assert data["guard"] is True and data["epoch"] == 3

    # 行の辞書の並び (列の欠けは null) と, 1 行の辞書
    save_evaluation(
        experiment.run_dir, "baseline_diff",
        [{"model": 0.3, "baseline": 0.4}, {"model": np.float32(0.2)}],
        "test", baseline="今の処理"
    )
    save_evaluation(experiment, "summary", {"iou": 0.7}, "valid")

    # 横軸の列を持つ曲線
    save_evaluation(
        experiment, "pr_curve",
        {"threshold": [0.1, 0.5], "precision": [0.6, 0.9]}, "test",
        kind="curve", x="threshold"
    )
    evaluations = load_evaluations(experiment.run_dir)
    names = [item["name"] for item in evaluations]
    assert names == ["baseline_diff", "per_group", "pr_curve", "summary"]
    assert evaluations[0]["rows"][1] == {
        "model": pytest.approx(0.2), "baseline": None
    }
    assert evaluations[3]["rows"] == [{"iou": 0.7}]


def test_save_evaluation_dataframe(tmp_path: Path) -> None:
    """
    pandas の DataFrame を書けることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    pandas = pytest.importorskip("pandas")
    experiment = make_experiment(tmp_path)
    table = pandas.DataFrame({"group": ["a", "b"], "count": [3, 4]})
    path = save_evaluation(experiment, "counts", table, "test")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["rows"] == [
        {"group": "a", "count": 3}, {"group": "b", "count": 4}
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "../x"}, "名前"),
        ({"name": "評価"}, "名前"),
        ({"name": ".hidden"}, "名前"),
        ({"epoch": -1}, "epoch"),
        ({"split": ""}, "split"),
        ({"kind": "curve"}, "横軸"),
        ({"kind": "chart"}, "kind"),
        ({"x": "iou"}, "curve"),
        ({"table": {"iou": [1, 2], "n": [1]}}, "長さ"),
        ({"table": {"iou": [[1, 2]]}}, "値"),
        ({"table": 3}, "表は"),
    ]
)
def test_save_evaluation_errors(
    tmp_path: Path, kwargs: dict, message: str
) -> None:
    """
    評価の誤りを書く前に見つけることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    kwargs: dict
        既定から変える引数
    message: str
        誤りのメッセージに含まれる文字

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    arguments = {"name": "ok", "table": {"iou": [0.5]}, "split": "test"}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        save_evaluation(experiment, **arguments)
    assert load_evaluations(experiment.run_dir) == []


def test_save_evaluation_better(tmp_path: Path) -> None:
    """
    列ごとの指標の向きを書き, 同じ指標の列に広げて読めることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)

    # 指標に書いた向きは _baseline と _diff にも広げる
    path = save_evaluation(
        experiment, "guard",
        {
            "group": ["a"], "accuracy": [0.9], "accuracy_baseline": [0.8],
            "accuracy_diff": [0.1], "loss": [0.2], "loss_diff": [-0.1]
        },
        "test", baseline="今の処理", guard=True,
        better={"accuracy": "higher", "loss_diff": "lower"}
    )
    expected = {
        "accuracy": "higher", "accuracy_baseline": "higher",
        "accuracy_diff": "higher", "loss": "lower", "loss_diff": "lower"
    }
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["better"] == expected
    assert load_evaluations(experiment.run_dir)[0]["better"] == expected

    # 書かなかった評価は空の辞書
    path = save_evaluation(experiment, "plain", {"iou": 0.5}, "test")
    assert json.loads(path.read_text(encoding="utf-8"))["better"] == {}

    # 曲線と空の辞書. 広げた向きは列の順
    path = save_evaluation(
        experiment, "curve",
        {"t": [0.1, 0.5], "recall_diff": [0.0, 0.1], "recall": [0.6, 0.9]},
        "test", kind="curve", x="t", better={"recall_diff": "higher"}
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert list(data["better"]) == ["recall_diff", "recall"]
    path = save_evaluation(experiment, "empty", {"iou": 0.5}, "test",
                           better={})
    assert json.loads(path.read_text(encoding="utf-8"))["better"] == {}


def test_better_does_not_depend_on_order(tmp_path: Path) -> None:
    """
    指標の向きの広げ方が書いた順で変わらないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # acc と acc_baseline_diff の向きは, どちらも acc_baseline に届く
    experiment = make_experiment(tmp_path)
    columns = ["acc", "acc_baseline", "acc_diff", "acc_baseline_diff"]
    table = {column: [0.1] for column in columns}
    for better in (
        {"acc": "higher", "acc_baseline_diff": "lower"},
        {"acc_baseline_diff": "lower", "acc": "higher"}
    ):
        with pytest.raises(ValueError, match="acc_baseline で食い違います") as error:
            save_evaluation(experiment, "acc", table, "test", better=better)
        assert "書いた列 " in str(error.value)
        assert "acc: higher" in str(error.value)
        assert "広げた列 無し" in str(error.value)

    # 読む側は, 2 つの向きが届く書いていない列を埋めない
    folder = experiment.run_dir / "evaluations"
    folder.mkdir(exist_ok=True)
    for index, better in enumerate((
        {"acc": "higher", "acc_baseline_diff": "lower"},
        {"acc_baseline_diff": "lower", "acc": "higher"}
    )):
        (folder / f"e{index}.json").write_text(json.dumps({
            "name": f"e{index}", "columns": columns, "rows": [],
            "better": better
        }), encoding="utf-8")
    results = [item["better"] for item in load_evaluations(experiment.run_dir)]
    assert results[0] == results[1] == {
        "acc": "higher", "acc_diff": "higher", "acc_baseline_diff": "lower"
    }


def test_better_spreads_until_stable(tmp_path: Path) -> None:
    """
    指標の向きを変化が無くなるまで広げ, 書いた向きが読み直しで変わらない
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # acc_baseline_diff から acc_baseline, そこから acc と acc_diff へ
    experiment = make_experiment(tmp_path)
    columns = ["acc", "acc_baseline", "acc_diff", "acc_baseline_diff"]
    path = save_evaluation(
        experiment, "acc", {column: [0.1] for column in columns}, "test",
        better={"acc_baseline_diff": "lower"}
    )
    written = json.loads(path.read_text(encoding="utf-8"))["better"]
    assert written == {column: "lower" for column in columns}
    assert spread_better(written, columns, strict=True) == written
    assert load_evaluations(experiment.run_dir)[0]["better"] == written


def test_better_mapping_and_message(tmp_path: Path) -> None:
    """
    DictConfig と NumPy の文字の向きを受け, 表に無い列の誤りに表の列を
    添えることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    better = OmegaConf.create({"iou": "higher"})
    path = save_evaluation(
        experiment, "cfg", {"iou": [0.5], "iou_diff": [0.1]}, "test",
        better=better
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["better"] == {"iou": "higher", "iou_diff": "higher"}
    path = save_evaluation(
        experiment, "numpy", {"mae": [0.5]}, "test",
        better={np.str_("mae"): np.str_("lower")}
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["better"] == {"mae": "lower"}
    assert type(next(iter(data["better"]))) is str

    # 多い列は初めの 10 個と残りの数を添える
    table = {f"m{index}": [0.0] for index in range(12)}
    with pytest.raises(ValueError, match=r"表の列: m0, .*m9 ほか 2 列"):
        save_evaluation(experiment, "many", table, "test",
                        better={"x": "lower"})


def test_load_evaluations_better_old_and_unknown(tmp_path: Path) -> None:
    """
    better の無い前のファイルと, 知らない値や表に無い列の欄を読めることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    folder = tmp_path / "evaluations"
    folder.mkdir()
    (folder / "old.json").write_text(json.dumps({
        "name": "old", "split": "test", "kind": "table", "x": None,
        "columns": ["iou"], "rows": [{"iou": 0.5}]
    }), encoding="utf-8")
    (folder / "odd.json").write_text(json.dumps({
        "name": "odd", "columns": ["iou", "iou_diff", "mae"],
        "rows": [], "better": {
            "iou_diff": "higher", "mae": "best", "missing": "lower"
        }
    }), encoding="utf-8")
    (folder / "mixed.json").write_text(json.dumps({
        "name": "mixed", "columns": ["a", "a_baseline", "a_diff"],
        "rows": [], "better": {"a_baseline": "lower", "a": "higher"}
    }), encoding="utf-8")
    (folder / "broken.json").write_text(json.dumps({
        "name": "broken", "columns": ["iou"], "rows": [], "better": ["iou"]
    }), encoding="utf-8")
    evaluations = {
        item["name"]: item["better"] for item in load_evaluations(tmp_path)
    }
    assert evaluations == {
        "broken": {}, "odd": {"iou": "higher", "iou_diff": "higher"},
        "old": {}, "mixed": {"a": "higher", "a_baseline": "lower"}
    }


@pytest.mark.parametrize(
    ("better", "message"),
    [
        (["iou"], "辞書"),
        ({"miou": "higher"}, "表にありません"),
        ({1: "higher"}, "表にありません"),
        ({"iou": "max"}, "lower か higher"),
        ({"iou": "higher", "iou_diff": "lower"}, "同じ向き"),
    ]
)
def test_save_evaluation_better_errors(
    tmp_path: Path, better: object, message: str
) -> None:
    """
    指標の向きの誤りを書く前に見つけることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    better: object
        渡す向き
    message: str
        誤りのメッセージに含まれる文字

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    with pytest.raises(ValueError, match=message):
        save_evaluation(
            experiment, "ok", {"iou": [0.5], "iou_diff": [0.1]}, "test",
            better=better
        )
    assert load_evaluations(experiment.run_dir) == []


def test_save_samples(tmp_path: Path) -> None:
    """
    画像, 文, 音, 点群, ファイルの見本を書き, 一覧に加えることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)

    # (チャンネル, 高さ, 幅) の小数の画像は PNG
    image = save_sample(
        experiment, "recon", torch.rand(3, 8, 6), epoch=1, kind="image",
        split="valid", caption="復元"
    )
    assert image.relative_to(experiment.run_dir).as_posix() == (
        "samples/recon/epoch_0001.png"
    )
    assert mpimg.imread(image).shape[:2] == (8, 6)

    # 白黒の整数の画像, 文, 音, 点群
    save_sample(
        experiment, "gray", np.full((4, 5), 300), epoch=1, kind="image"
    )
    text = save_sample(experiment, "caption", "犬が走る", epoch=2)
    assert text.read_text(encoding="utf-8") == "犬が走る"
    audio = save_sample(
        experiment, "voice", np.zeros((2, 100)), kind="audio",
        sample_rate=8000
    )
    with wave.open(str(audio), "rb") as file:
        assert file.getnchannels() == 2
        assert file.getnframes() == 100
        assert file.getframerate() == 8000
    cloud = save_sample(
        experiment, "cloud", torch.zeros(5, 3), epoch=2, kind="point_cloud"
    )
    lines = cloud.read_text(encoding="ascii").splitlines()
    assert lines[0] == "ply" and "element vertex 5" in lines
    assert len(lines) == lines.index("end_header") + 6

    # 拡張子から種類を決めたファイルの写し
    source = tmp_path / "clip.gif"
    source.write_bytes(b"GIF89a")
    copied = save_sample(experiment, "clip", source)
    assert copied.read_bytes() == b"GIF89a"

    # 一覧は書いた順で, 学習の後は final
    samples = load_samples(experiment.run_dir)
    assert [(item["epoch"], item["name"], item["kind"]) for item in samples] \
        == [
            (1, "recon", "image"), (1, "gray", "image"),
            (2, "caption", "text"), (None, "voice", "audio"),
            (2, "cloud", "point_cloud"), (None, "clip", "image")
        ]
    assert samples[0]["split"] == "valid"
    assert samples[0]["caption"] == "復元"
    assert samples[3]["file"] == "voice/final.wav"
    assert not list((experiment.run_dir / "samples").rglob("*.tmp"))


def test_save_sample_replaces(tmp_path: Path) -> None:
    """
    同じエポックと名前の見本を置き換えることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 文から画像に変えると, 前のファイルを消して一覧の項目を置き換え
    experiment = make_experiment(tmp_path)
    old = save_sample(experiment, "out", "前の見本", epoch=3)
    new = save_sample(
        experiment, "out", np.zeros((2, 2)), epoch=3, kind="image"
    )
    save_sample(experiment, "out", "別のエポック", epoch=4)
    assert not old.exists() and new.is_file()
    samples = load_samples(experiment.run_dir)
    assert [(item["epoch"], item["kind"]) for item in samples] == [
        (3, "image"), (4, "text")
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"value": np.zeros(3)}, "kind"),
        ({"value": np.zeros((4, 2)), "kind": "point_cloud"}, "点群"),
        ({"value": np.zeros((2, 3, 4, 5)), "kind": "image"}, "画像"),
        ({"value": np.zeros(10), "kind": "audio"}, "sample_rate"),
        ({"value": np.zeros((100, 2)), "kind": "audio", "sample_rate": 8},
         ".T"),
        ({"value": np.full((2, 3), np.nan), "kind": "point_cloud"}, "nan"),
        ({"value": np.zeros((2, 2)), "kind": "file"}, "pathlib"),
        ({"value": "x", "name": ".hidden"}, "名前"),
        ({"value": "x", "kind": "video"}, "kind"),
        ({"value": "x", "epoch": -1}, "epoch"),
        ({"value": "x", "name": "a/b"}, "名前"),
        ({"value": Path("missing.png")}, "ありません"),
    ]
)
def test_save_sample_errors(
    tmp_path: Path, kwargs: dict, message: str
) -> None:
    """
    見本の誤りを一覧を変えずに見つけることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    kwargs: dict
        既定から変える引数
    message: str
        誤りのメッセージに含まれる文字

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    arguments = {"name": "sample"}
    arguments.update(kwargs)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        save_sample(experiment, **arguments)
    assert load_samples(experiment.run_dir) == []
    samples_dir = experiment.run_dir / "samples"
    assert not samples_dir.exists() or not list(samples_dir.rglob("*.*"))


def test_old_run_without_optional_files(tmp_path: Path) -> None:
    """
    任意のファイルを持たない実験 (task を書かなかった実験など) を
    読めて, 再開にも使えることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # task を書かない実験
    first = make_experiment(tmp_path)
    fit_briefly(first)
    for name in (TASK_FILE, "evaluations", "samples"):
        assert not (first.run_dir / name).exists()
    assert load_task(first.run_dir) is None
    assert load_evaluations(first.run_dir) == []
    assert load_samples(first.run_dir) == []

    # その実験からの再開と, 微調整の元としての書き込み
    second = make_experiment(
        tmp_path / "resume",
        f"trainer:\n  epochs: 2\n  resume_from: {first.run_dir / 'last.pt'}\n"
    )
    fit_briefly(second)
    task = write_task(
        second, {"method": "finetune", "base": {
            "experiment": str(first.run_dir)
        }}
    )
    assert task["base"]["experiment"] == first.run_dir.as_posix()


def test_save_sample_failure_keeps_previous(tmp_path: Path) -> None:
    """
    置き換えの書き込みに失敗した場合に前の見本と一覧が残ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    old = save_sample(experiment, "out", "前の見本", epoch=1)
    with pytest.raises(ValueError, match="画像"):
        save_sample(
            experiment, "out", np.zeros((2, 2, 2, 2)), epoch=1, kind="image"
        )
    assert old.read_text(encoding="utf-8") == "前の見本"
    assert [item["file"] for item in load_samples(experiment.run_dir)] == [
        "out/epoch_0001.txt"
    ]


def test_task_dataset_unknown_origin_field_and_null(tmp_path: Path) -> None:
    """
    dataset.yaml の知らない欄を写さず, null と書いた項目が埋めた値を
    消さないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # ML-EDA が版を上げずに出どころへ足した欄
    folder = write_dataset(tmp_path / "data")
    info = OmegaConf.load(folder / "dataset.yaml")
    info.target.origin.version = "1.2"
    OmegaConf.save(info, folder / "dataset.yaml")

    # 全ての項目を null で並べた config
    experiment = make_experiment(
        tmp_path,
        "task:\n  dataset: data\n  role: null\n  what: null\n"
        "  target:\n    kind: null\n    origin: null\n"
    )
    task = load_task(experiment.run_dir)
    assert task["role"] == "predict"
    assert task["target"]["kind"] == "point_cloud"
    assert set(task["target"]["origin"]) == {"type", "name", "settings"}
    assert task["target"]["origin"]["type"] == "process"


def test_config_task_without_task_keys(tmp_path: Path) -> None:
    """
    task.yaml の項目を持たない辞書の task は警告して使わないことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    with pytest.warns(UserWarning, match="task.yaml の項目が無い"):
        experiment = make_experiment(tmp_path, "task:\n  classes: 10\n")
    assert not (experiment.run_dir / TASK_FILE).exists()


def test_save_evaluation_empty_table(tmp_path: Path) -> None:
    """
    空の表は行の無い評価になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    save_evaluation(experiment, "empty", {}, "test")
    evaluation = load_evaluations(experiment.run_dir)[0]
    assert evaluation["columns"] == [] and evaluation["rows"] == []


@pytest.mark.parametrize("version", ["'2'", "2.0", "true", "0", "null"])
def test_load_task_invalid_version(tmp_path: Path, version: str) -> None:
    """
    整数でない版の task.yaml を版 1 として読まずに止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    version: str
        format_version に書く YAML の値

    Returns
    ----------
    None
    """
    (tmp_path / TASK_FILE).write_text(
        f"format_version: {version}\nrole: predict\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="format_version"):
        load_task(tmp_path)


def test_partial_origin_warns(tmp_path: Path) -> None:
    """
    種類を書かない正解の出どころが unknown になることを警告する関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    with pytest.warns(UserWarning, match="type が無い"):
        task = write_task(
            experiment, {"target": {"origin": {"name": "reconstruct"}}}
        )
    assert task["target"]["origin"]["type"] == "unknown"
    assert task["target"]["origin"]["name"] == "reconstruct"


def test_load_samples_legacy_flat_files(tmp_path: Path) -> None:
    """
    samples/ の直下に置かれた番号付きの画像 (ML-Found の形より前の
    プロジェクト) も見本として読み, 新しい見本とぶつからないことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # GAN と拡散のタスクが書いていた形
    experiment = make_experiment(tmp_path)
    root = experiment.run_dir / "samples"
    root.mkdir()
    for name in ("epoch_010.png", "epoch_002.png", "fake_003.png", "a.txt"):
        (root / name).write_bytes(b"x")
    samples = load_samples(experiment.run_dir)
    assert [(item["name"], item["epoch"], item["file"]) for item in samples] \
        == [
            ("fake", 3, "fake_003.png"), ("samples", 2, "epoch_002.png"),
            ("samples", 10, "epoch_010.png")
        ]
    assert all(item["legacy"] for item in samples)

    # 新しい見本は名前のディレクトリに置き, 直下のファイルとのぶつかりは誤り
    save_sample(experiment, "recon", np.zeros((2, 2)), epoch=1, kind="image")
    samples = load_samples(experiment.run_dir)
    assert samples[0]["file"] == "recon/epoch_0001.png"
    assert len(samples) == 4
    with pytest.raises(ValueError, match="ぶつかります"):
        save_sample(experiment, "epoch_010.png", "x")
    with pytest.raises(ValueError, match="index.json"):
        save_sample(experiment, "index.json", "x")


def test_sample_layout_matches_dashboard_series() -> None:
    """
    新しい見本の場所が ML-Dashboard の画像の推移の見分け方 (番号の前の
    部分とディレクトリでまとめる) で名前ごとの並びになることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # ML-Dashboard の NUMBERED と同じ正規表現でのまとまり
    numbered = re.compile(r"^(.*?)(\d+)\.(png|jpe?g|gif|webp)$", re.I)
    groups: dict[str, list[int]] = {}
    paths = [
        "samples/recon/epoch_0001.png", "samples/recon/epoch_0012.png",
        "samples/real/epoch_0001.png", "samples/epoch_010.png",
        "samples/epoch_020.png", "samples/recon/final.png"
    ]
    for path in paths:
        folder, file = path.rsplit("/", 1)
        match = numbered.match(file)
        if match is None:
            continue
        prefix = match.group(1).rstrip("_.-")
        name = f"{folder}/{prefix}" if prefix else folder
        groups.setdefault(name, []).append(int(match.group(2)))
    assert groups == {
        "samples/recon/epoch": [1, 12], "samples/real/epoch": [1],
        "samples/epoch": [10, 20]
    }


def test_samples_threads_keep_all_entries(tmp_path: Path) -> None:
    """
    複数のスレッドから書いた見本が一覧から落ちないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    threads = [
        threading.Thread(
            target=save_sample, args=(experiment, f"s{index}", "x"),
            kwargs={"epoch": 1}
        )
        for index in range(40)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(load_samples(experiment.run_dir)) == 40


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"value": np.full((2, 2), np.nan), "kind": "image"}, "nan"),
        ({"value": np.zeros(10, dtype=np.int16), "kind": "audio",
          "sample_rate": 8}, "小数"),
        ({"value": np.full(10, np.nan), "kind": "audio",
          "sample_rate": 8}, "nan"),
        ({"value": "x", "caption": 3}, "caption"),
        ({"value": "x", "split": ["test"]}, "split"),
    ]
)
def test_save_sample_value_errors(
    tmp_path: Path, kwargs: dict, message: str
) -> None:
    """
    nan, 整数の音, 文字でない添え書きを, 前の見本と一覧を変えずに
    誤りにすることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    kwargs: dict
        既定から変える引数
    message: str
        誤りのメッセージに含まれる文字

    Returns
    ----------
    None
    """
    # 前の見本を書いてから, 同じエポックと名前で誤りの見本
    experiment = make_experiment(tmp_path)
    old = save_sample(experiment, "out", "前の見本", epoch=1)
    arguments = {"name": "out", "epoch": 1}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        save_sample(experiment, **arguments)
    assert old.read_text(encoding="utf-8") == "前の見本"
    assert [item["file"] for item in load_samples(experiment.run_dir)] == [
        "out/epoch_0001.txt"
    ]
    assert not list(experiment.run_dir.rglob("*.tmp"))


def test_evaluation_text_arguments(tmp_path: Path) -> None:
    """
    評価の説明と比べる相手が文字でなければ書く前に誤りにすることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    with pytest.raises(ValueError, match="description"):
        save_evaluation(experiment, "a", {"x": 1}, "test", description=1)
    with pytest.raises(ValueError, match="baseline"):
        save_evaluation(experiment, "a", {"x": 1}, "test", baseline=2.0)
    assert load_evaluations(experiment.run_dir) == []

    # 複数の要素を持つ値は, 0 次元でない 1 要素のテンソルなら数として可
    save_evaluation(
        experiment, "b", {"x": [torch.tensor([0.5])]}, "test"
    )
    assert load_evaluations(experiment.run_dir)[0]["rows"] == [{"x": 0.5}]


def test_new_files_are_readable_by_others(tmp_path: Path) -> None:
    """
    新しいファイルが config.yaml と同じ権限 (umask に従う) で作られる
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    write_task(experiment, {"role": "predict"})
    paths = [
        experiment.run_dir / TASK_FILE,
        save_evaluation(experiment, "a", {"x": 1}, "test"),
        save_sample(experiment, "s", "x", epoch=1),
        experiment.run_dir / "samples" / "index.json"
    ]
    expected = (experiment.run_dir / "config.yaml").stat().st_mode & 0o777
    for path in paths:
        assert path.stat().st_mode & 0o777 == expected, path


def test_dataset_yaml_dollar_text(tmp_path: Path) -> None:
    """
    dataset.yaml の ${ を含む文字を補間の書き方として解かずに写すことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # OmegaConf では読めない書き方 (閉じない ${) を含む設定
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "dataset.yaml").write_text(
        "format_version: 2\nrole: predict\ntarget:\n  kind: mask\n"
        "  origin:\n    type: process\n    name: tool\n"
        "    settings: {cmd: 'run ${HOME and ${x'}\n",
        encoding="utf-8"
    )
    task = task_from_dataset(folder)
    assert task["target"]["origin"]["settings"]["cmd"] == "run ${HOME and ${x"
    experiment = make_experiment(tmp_path / "run")
    write_task(experiment, task)
    loaded = load_task(experiment.run_dir)
    assert loaded["target"]["origin"]["settings"]["cmd"] == (
        "run ${HOME and ${x"
    )


def test_sample_names_with_dots_do_not_collide(tmp_path: Path) -> None:
    """
    拡張子に見える名前 (a.txt) と拡張子の無い名前 (a) の見本が,
    名前ごとのディレクトリで別のファイルになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    first = save_sample(experiment, "a", "文", epoch=1)
    second = save_sample(experiment, "a.txt", "別の文", epoch=1)
    assert first != second
    assert first.read_text(encoding="utf-8") == "文"
    assert second.read_text(encoding="utf-8") == "別の文"
    assert len(load_samples(experiment.run_dir)) == 2


def test_policy_task_values(tmp_path: Path) -> None:
    """
    方策の task.yaml (出力の形 action, 正解の出どころ environment) を書け,
    出どころの説明が環境の報酬の文になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    write_task(experiment, {
        "role": "effect",
        "output": "action",
        "what": "観測から押す向きを選ぶ方策",
        "target": {
            "kind": "reward",
            "origin": {"type": "environment", "name": "CartPole-v1"}
        },
        "method": "scratch"
    })
    task = load_task(experiment.run_dir)
    assert task["format_version"] == 1
    assert task["output"] == "action"
    assert task["target"] == {
        "kind": "reward",
        "origin": {
            "type": "environment", "name": "CartPole-v1", "settings": None
        }
    }
    assert describe_origin(task["target"]["origin"]) == (
        "環境の報酬で測る. 実機での結果は別に確かめる"
    )
    log = (experiment.run_dir / "train.log").read_text(encoding="utf-8")
    assert "環境の報酬で測る" in log


def test_save_sample_mask(tmp_path: Path) -> None:
    """
    領域の番号の見本を決まった色で PNG に書き, 番号の表を一覧に残すことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    mask = torch.tensor([[0, 1], [2, 2]])
    path = save_sample(
        experiment, "seg", mask, epoch=3, kind="mask",
        classes=["背景", "物", "縁", "影"]
    )
    assert path.relative_to(experiment.run_dir).as_posix() == (
        "samples/seg/epoch_0003.png"
    )

    # 色は PASCAL VOC と同じ表 (0 は黒, 1 は暗い赤, 2 は暗い緑)
    pixels = np.rint(mpimg.imread(path)[..., :3] * 255).astype(int)
    assert pixels[0, 0].tolist() == [0, 0, 0]
    assert pixels[0, 1].tolist() == [128, 0, 0]
    assert pixels[1, 0].tolist() == [0, 128, 0]

    # 名前を渡した番号と現れた番号の全てを番号の順に
    entry = load_samples(experiment.run_dir)[0]
    assert entry["kind"] == "mask"
    assert entry["classes"] == [
        {"id": 0, "name": "背景", "color": "#000000"},
        {"id": 1, "name": "物", "color": "#800000"},
        {"id": 2, "name": "縁", "color": "#008000"},
        {"id": 3, "name": "影", "color": "#808000"}
    ]
    assert entry["convention"] is None

    # 辞書の名前と名前の無い番号, 同じ番号は実験が違っても同じ色
    save_sample(
        experiment, "seg2", np.array([[0, 5]], dtype=np.uint8), kind="mask",
        classes={0: "背景"}
    )
    second = load_samples(experiment.run_dir)[1]
    assert second["classes"] == [
        {"id": 0, "name": "背景", "color": "#000000"},
        {"id": 5, "name": None, "color": "#800080"}
    ]

    # 真偽値と (1, 高さ, 幅) の形も可
    save_sample(experiment, "fg", torch.ones(1, 2, 3, dtype=torch.bool),
                kind="mask")
    assert [row["id"] for row in load_samples(experiment.run_dir)[2][
        "classes"
    ]] == [1]


@pytest.mark.parametrize(
    "value, kwargs, message",
    [
        (np.zeros((2, 2)), {}, "整数か真偽値"),
        (np.array([[0, 256]]), {}, "0 から 255"),
        (np.array([[-1, 0]]), {}, "0 から 255"),
        (np.zeros((2, 2, 3), dtype=int), {}, "高さ, 幅"),
        (np.zeros((0, 3), dtype=int), {}, "空"),
        (np.zeros((2, 2), dtype=int), {"classes": "abc"}, "classes"),
        (np.zeros((2, 2), dtype=int), {"classes": {300: "x"}}, "classes"),
        (np.zeros((2, 2), dtype=int), {"classes": [1]}, "名前"),
        (Path("seg.png"), {}, "番号の配列")
    ]
)
def test_save_sample_mask_errors(
    tmp_path: Path, value: object, kwargs: dict, message: str
) -> None:
    """
    領域の見本の値と番号の表の誤りを, 書く前に拒むことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    value: object
        見本の値
    kwargs: dict
        save_sample に足す引数
    message: str
        誤りのメッセージに含まれる文字

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    with pytest.raises(ValueError, match=message):
        save_sample(experiment, "seg", value, kind="mask", **kwargs)
    assert load_samples(experiment.run_dir) == []


def test_save_sample_convention(tmp_path: Path) -> None:
    """
    点群の座標の決まりを一覧の convention に残し, 番号の表は mask だけに
    渡せることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    convention = {"units": "m", "shift": [0.1, 0.0, 0.5], "scale": 2.0}
    save_sample(
        experiment, "cloud", np.zeros((3, 3)), kind="point_cloud",
        convention=convention
    )
    entry = load_samples(experiment.run_dir)[0]
    assert entry["convention"] == convention
    assert entry["classes"] is None
    assert entry["newer_version"] is False

    # JSON にできない値, 辞書でない値, mask でない種類の classes は誤り
    for kwargs in (
        {"convention": {"shift": np.zeros(3)}},
        {"convention": {"scale": float("nan")}},
        {"convention": ["m"]},
        {"classes": ["背景"]}
    ):
        with pytest.raises(ValueError):
            save_sample(
                experiment, "cloud2", np.zeros((3, 3)), kind="point_cloud",
                **kwargs
            )
    assert len(load_samples(experiment.run_dir)) == 1


def test_load_samples_fills_new_fields(tmp_path: Path) -> None:
    """
    classes と convention の無い一覧の項目にも null を補うことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # classes と convention の無い形の一覧
    experiment = make_experiment(tmp_path)
    root = experiment.run_dir / "samples"
    (root / "note").mkdir(parents=True)
    (root / "note" / "final.txt").write_text("x", encoding="utf-8")
    old = {
        "epoch": None, "name": "note", "kind": "text",
        "file": "note/final.txt", "split": None, "caption": None,
        "created_at": "2026-11-19T00:00:00+09:00"
    }
    (root / "index.json").write_text(
        json.dumps({"samples": [old]}), encoding="utf-8"
    )
    entry = load_samples(experiment.run_dir)[0]
    assert entry["classes"] is None and entry["convention"] is None
    assert entry["name"] == "note"


def write_index(root: Path, data: dict) -> None:
    """
    点群の見本のファイルと, 渡した形の一覧を置く関数

    Parameters
    ----------
    root: Path
        samples/ のディレクトリ
    data: dict
        index.json に書く辞書

    Returns
    ----------
    None
    """
    (root / "cloud").mkdir(parents=True)
    (root / "cloud" / "final.ply").write_text("ply\n", encoding="utf-8")
    (root / "index.json").write_text(json.dumps(data), encoding="utf-8")


def cloud_entry(convention: dict) -> dict:
    """
    点群の見本の一覧の項目を返す関数

    Parameters
    ----------
    convention: dict
        値の決まり

    Returns
    ----------
    entry: dict
        学習の後の点群の見本の項目
    """
    entry = {
        "epoch": None, "name": "cloud", "kind": "point_cloud",
        "file": "cloud/final.ply", "split": None, "caption": None,
        "classes": None, "convention": convention,
        "created_at": "2026-11-19T00:00:00+09:00"
    }

    return entry


def test_sample_shift_matches_eda(tmp_path: Path) -> None:
    """
    今の版の shift が ML-EDA と同じ向き (正規化した座標 = 元の座標 +
    shift) で, original_coordinates が元の座標に戻すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # ML-EDA の正規化 (中心を原点に) の後に scale で割った点群
    original = np.array([[1.0, 2.0, 3.0], [1.5, 2.0, 2.0]])
    shift = [-1.0, -2.0, -3.0]
    scale = 2.0
    values = (original + np.array(shift)) / scale
    experiment = make_experiment(tmp_path)
    convention = {"units": "m", "shift": shift, "scale": scale}
    save_sample(experiment, "cloud", values, kind="point_cloud",
                convention=convention)
    index = json.loads(
        (experiment.run_dir / "samples" / "index.json").read_text(
            encoding="utf-8"
        )
    )
    assert index["format_version"] == 2
    assert index["samples"][0]["convention"]["shift"] == shift
    entry = load_samples(experiment.run_dir)[0]
    np.testing.assert_allclose(
        original_coordinates(values, entry["convention"]), original
    )

    # 値の決まりが無ければ値のまま. 軸ごとの scale も使える
    np.testing.assert_allclose(original_coordinates(values, None), values)
    np.testing.assert_allclose(
        original_coordinates(values, {"shift": None, "scale": None}), values
    )
    np.testing.assert_allclose(
        original_coordinates(values, {"scale": [1.0, 2.0, 1.0]}),
        values * np.array([1.0, 2.0, 1.0])
    )

    # 点群でない形と, 形の合わない shift は誤り
    for points, given in (
        (np.zeros(3), None),
        (np.zeros((2, 2)), None),
        (values, {"shift": [1.0, 2.0]}),
        (values, {"scale": "x"}),
        (values, ["m"])
    ):
        with pytest.raises(ValueError):
            original_coordinates(points, given)


def test_old_sample_shift_still_reads(tmp_path: Path) -> None:
    """
    版の無い (版 1 の) 一覧の shift (元の座標 = 値 × scale + shift) を,
    今の意味に写して読み, 書き足しても元の座標が変わらないことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 版 1 の向きで書いた見本
    original = np.array([[1.0, 2.0, 3.0], [1.5, 2.0, 2.0]])
    old_shift = [1.0, 2.0, 3.0]
    scale = 2.0
    values = (original - np.array(old_shift)) / scale
    experiment = make_experiment(tmp_path)
    root = experiment.run_dir / "samples"
    old = {"units": "m", "shift": old_shift, "scale": scale}
    write_index(root, {"samples": [cloud_entry(old)]})
    entry = load_samples(experiment.run_dir)[0]
    assert entry["convention"]["shift"] == [-1.0, -2.0, -3.0]
    assert entry["convention"]["units"] == "m"
    assert entry["newer_version"] is False
    np.testing.assert_allclose(
        original_coordinates(values, entry["convention"]), original
    )

    # 書き足すと一覧は今の版になり, 前の項目も今の意味で残る
    save_sample(experiment, "note", "見本の文")
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    assert index["format_version"] == 2
    assert index["samples"][0]["convention"]["shift"] == [-1.0, -2.0, -3.0]
    entry = load_samples(experiment.run_dir)[0]
    np.testing.assert_allclose(
        original_coordinates(values, entry["convention"]), original
    )

    # 数でない shift と, 入れ子の数
    nested = {"shift": [[1, -2], "x", None, True, 0.5]}
    other = make_experiment(tmp_path / "other")
    write_index(other.run_dir / "samples", {"samples": [cloud_entry(nested)]})
    entry = load_samples(other.run_dir)[0]
    assert entry["convention"]["shift"] == [[-1, 2], "x", None, True, -0.5]


def test_newer_samples_index(tmp_path: Path) -> None:
    """
    新しい版の一覧は値の決まりを読まずに知らせ, 書き足さないことと,
    版が整数でない一覧は読めないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    root = experiment.run_dir / "samples"
    entry = cloud_entry({"shift": [1.0, 0.0, 0.0]})
    write_index(root, {"format_version": 3, "samples": [entry]})
    loaded = load_samples(experiment.run_dir)[0]
    assert loaded["convention"] is None
    assert loaded["newer_version"] is True
    assert loaded["name"] == "cloud"
    with pytest.raises(ValueError, match="新しい"):
        save_sample(experiment, "note", "見本の文")

    # 文字や真偽値の版は誤り (版 1 として読み違えないため)
    for version in ("2", True, 0):
        (root / "index.json").write_text(
            json.dumps({"format_version": version, "samples": [entry]}),
            encoding="utf-8"
        )
        assert load_samples(experiment.run_dir) == []
        with pytest.raises(ValueError, match="format_version"):
            save_sample(experiment, "note", "見本の文")
