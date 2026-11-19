import json
from pathlib import Path
import wave

from matplotlib import image as mpimg
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from ml_found import (
    describe_origin, load_evaluations, load_samples, load_task,
    save_evaluation, save_sample, task_from_dataset, write_task
)
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

    # 相対パスは渡したディレクトリから辿る
    relative = task_from_dataset("data", project_dir=str(tmp_path))
    assert relative == task


@pytest.mark.parametrize(
    ("version", "message"),
    [(1, "版 1"), (3, "読めません"), (None, "読めません"),
     ("2", "読めません")]
)
def test_task_from_dataset_versions(
    tmp_path: Path, version: object, message: str
) -> None:
    """
    dataset.yaml の版が 2 でなければはっきり止まることを確かめる関数

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
        ("task:\n  rol: predict\n", "知らない項目"),
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

    # task_from_dataset の結果に足して渡しても同じ
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
    (tmp_path / TASK_FILE).write_text(
        "format_version: 2\nrole: predict\nnew_field: 1\n", encoding="utf-8"
    )
    with pytest.warns(UserWarning, match="新しい版"):
        task = load_task(tmp_path)
    assert task["role"] == "predict"

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

    # 曲線は横軸の列が要る
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
        "samples/epoch_0001/recon.png"
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

    # ファイルは拡張子から種類を決めて写す
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
    assert samples[3]["file"] == "final/voice.wav"
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
    任意のファイルを持たない実験 (足す前の ML-Found が作った実験) を
    読めて, 再開にも使えることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # task を書かない今までの実験
    first = make_experiment(tmp_path)
    fit_briefly(first)
    for name in (TASK_FILE, "evaluations", "samples"):
        assert not (first.run_dir / name).exists()
    assert load_task(first.run_dir) is None
    assert load_evaluations(first.run_dir) == []
    assert load_samples(first.run_dir) == []

    # その実験から再開し, 微調整の元としても書ける
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
