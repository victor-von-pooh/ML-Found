import hashlib
import os
from pathlib import Path
import shutil
import sys

import pytest
from omegaconf import OmegaConf

from ml_found.config.loader import load_project_config
from ml_found.experiment import model_spec, setup
from ml_found.experiment.model_spec import (
    MODEL_SPEC_FILE, copy_model_spec, describe_model_spec, read_model_spec
)
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


def copy_spec(value: object, run_dir: Path, project: Path) -> object:
    """
    model.spec に場所を書いた config で定義を読んで写す関数

    Parameters
    ----------
    value: object
        model.spec の値
    run_dir: Path
        実験のディレクトリ
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    record: object
        source.yaml の model_spec の記録 (dict か None)
    """
    spec = read_model_spec(spec_config(value), project)
    record = copy_model_spec(spec, run_dir)

    return record


def describe_spec(value: object, project: Path) -> object:
    """
    model.spec に場所を書いた config で定義を読み, train.log に書く文を
    作る関数 (写さない場所に使う)

    Parameters
    ----------
    value: object
        model.spec の値
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    message: object
        train.log に書く文 (str か None)
    """
    spec = read_model_spec(spec_config(value), project)
    message = describe_model_spec(spec, spec.record)

    return message


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

    # 文字でない値だけは, 書き方の誤りとして train.log に書く
    log = (run_dir / "train.log").read_text(encoding="utf-8")
    assert ("場所の文字でない" in log) == ("spec: 3" in spec)


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
        record = copy_spec(value, run_dir, project)
        assert record == {"path": path, "status": status, "sha256": None}
    assert not (run_dir / MODEL_SPEC_FILE).exists()

    # プロジェクトの中の絶対パスは写し, 相対パスで記録
    record = copy_spec(
        str(project / "models" / "real.yaml"), run_dir, project
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
    record = copy_spec("model.yaml", tmp_path, project)
    assert record == {
        "path": "model.yaml", "status": "not_file", "sha256": None
    }


def test_resumed_copies(tmp_path: Path) -> None:
    """
    再開する学習でも, 学習を始めるときの定義を写すことを確かめる関数

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
    spec = read_model_spec(
        spec_config("model.yaml", "outputs/last.pt"), project
    )
    record = copy_model_spec(spec, run_dir)
    assert record["status"] == "copied"
    assert (run_dir / MODEL_SPEC_FILE).read_text(encoding="utf-8") == (
        SPEC_TEXT
    )


def test_project_reached_through_link(tmp_path: Path) -> None:
    """
    リンクを通したプロジェクトの場所から書いた絶対パスも, プロジェクトの
    中として写すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 本物のプロジェクトと, それを指すリンク (macOS の /tmp と同じ形)
    project = tmp_path / "project"
    run_dir = tmp_path / "run"
    (project / "models").mkdir(parents=True)
    run_dir.mkdir()
    (project / "models" / "m.yaml").write_text(SPEC_TEXT, encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(project, target_is_directory=True)

    # リンクから書いた絶対パスと, リンクで渡したプロジェクト
    value = str(alias / "models" / "m.yaml")
    for root in (project, alias):
        record = copy_spec(value, run_dir, root)
        assert record["status"] == "copied"
        assert record["path"] == "models/m.yaml"


def test_not_directory_and_unreadable(tmp_path: Path) -> None:
    """
    途中がファイルの場所を無いものとし, 読めない途中のディレクトリでも
    止まらずに理由を残すことを確かめる関数

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
    (project / "models").write_text("", encoding="utf-8")
    record = copy_spec("models/m.yaml", run_dir, project)
    assert record["status"] == "missing"

    # 入れないディレクトリの中 (root では確かめられないので飛ばす)
    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        pytest.skip("権限で読めない場所を作れない")
    locked = project / "locked"
    (locked / "inner").mkdir(parents=True)
    (locked / "inner" / "m.yaml").write_text(SPEC_TEXT, encoding="utf-8")
    locked.chmod(0)
    try:
        record = copy_spec("locked/inner/m.yaml", run_dir, project)
    finally:
        locked.chmod(0o755)
    assert record["status"] == "unreadable"
    assert not (run_dir / MODEL_SPEC_FILE).exists()


def test_dotdot_after_link(tmp_path: Path) -> None:
    """
    リンクの後の .. で上がる場所を, 正規化した別のファイルとして写さない
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 中と外に同じ名前の定義, 外の下のディレクトリを指すリンク
    project = tmp_path / "project"
    run_dir = tmp_path / "run"
    outside = tmp_path / "outside"
    (outside / "sub").mkdir(parents=True)
    project.mkdir()
    run_dir.mkdir()
    (outside / "m.yaml").write_text("outside\n", encoding="utf-8")
    (project / "m.yaml").write_text(SPEC_TEXT, encoding="utf-8")
    (project / "extlink").symlink_to(
        outside / "sub", target_is_directory=True
    )
    record = copy_spec("extlink/../m.yaml", run_dir, project)
    assert record == {
        "path": "extlink/../m.yaml", "status": "link", "sha256": None
    }
    absolute = str(project / "extlink" / ".." / "m.yaml")
    assert copy_spec(absolute, run_dir, project) == {
        "path": absolute, "status": "link", "sha256": None
    }
    assert copy_spec("extlink/..", run_dir, project)["status"] == "link"
    assert not (run_dir / MODEL_SPEC_FILE).exists()

    # .. の直前が無い場所は開けないので missing, 末尾の区切りは not_file.
    # train.log の説明は書いた文字を添えて理由を書く
    record = copy_spec("nosuch/../m.yaml", run_dir, project)
    assert record == {"path": "m.yaml", "status": "missing", "sha256": None}
    message = describe_spec("nosuch/../m.yaml", project)
    assert "'nosuch/../m.yaml' の .. の直前が無いか" in message
    assert "ファイルがありません" not in message
    for value in ("m.yaml/", "m.yaml/."):
        assert copy_spec(value, run_dir, project)["status"] == "not_file"
        message = describe_spec(value, project)
        assert f"{value!r} は末尾が区切りの" in message
        assert "普通のファイルではありません" not in message

    # ファイルが本当に無い場合と開けない場所は, status の説明のまま
    assert describe_spec("nosuch.yaml", project).endswith(
        "ファイルがありません"
    )
    assert describe_spec("m.yaml/x.yaml", project).endswith(
        "ファイルがありません"
    )

    # リンクと関係の無い .. はリンクで止めても正規化して記録
    (project / "sub").mkdir()
    (project / "lnk").symlink_to(project / "m.yaml")
    record = copy_spec("sub/../lnk", run_dir, project)
    assert record == {"path": "lnk", "status": "link", "sha256": None}

    # リンクでないディレクトリの後の .. はそのまま中の場所
    (project / "models").mkdir()
    record = copy_spec("models/../m.yaml", run_dir, project)
    assert record["status"] == "copied"
    record = copy_spec("./models/./../m.yaml", run_dir, project)
    assert record == {
        "path": "m.yaml", "status": "copied",
        "sha256": hashlib.sha256(SPEC_TEXT.encode()).hexdigest()
    }


def test_invalid_places_and_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    読めない書き方の場所と写しの書き込みの失敗で, 止まらずに理由を残す
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
    # 存在しないユーザーの ~ とヌル文字
    project = tmp_path / "project"
    run_dir = tmp_path / "run"
    project.mkdir()
    run_dir.mkdir()
    for value in ("~no_such_user_for_ml_found/m.yaml", "models/a\0b.yaml"):
        record = copy_spec(value, run_dir, project)
        assert record["status"] == "invalid", value
        assert record["sha256"] is None

    # 写す先に同じ名前のディレクトリがあると書けない
    (project / "m.yaml").write_text(SPEC_TEXT, encoding="utf-8")
    (run_dir / MODEL_SPEC_FILE).mkdir()
    record = copy_spec("m.yaml", run_dir, project)
    assert record == {
        "path": "m.yaml", "status": "write_failed", "sha256": None
    }
    assert not list(run_dir.glob(f".{MODEL_SPEC_FILE}.*.tmp"))

    # start_experiment も止まらずに train.log と source.yaml に残す
    run = run_start(
        tmp_path / "started", monkeypatch,
        "model:\n  spec: ~no_such_user_for_ml_found/m.yaml\n"
    )
    assert read_record(run)["status"] == "invalid"
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "'~no_such_user_for_ml_found/m.yaml'" in log


def test_too_large(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    大きさの上限までは写し, 超えると写さないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        上限の差し替えに使う

    Returns
    ----------
    None
    """
    project = tmp_path / "project"
    run_dir = tmp_path / "run"
    project.mkdir()
    run_dir.mkdir()
    monkeypatch.setattr(model_spec, "MAX_SPEC_BYTES", 8)
    monkeypatch.setattr(model_spec, "CHUNK_BYTES", 3)
    (project / "ok.yaml").write_bytes(b"12345678")
    (project / "big.yaml").write_bytes(b"123456789")
    record = copy_spec("ok.yaml", run_dir, project)
    assert record["status"] == "copied"
    assert record["sha256"] == hashlib.sha256(b"12345678").hexdigest()
    assert (run_dir / MODEL_SPEC_FILE).read_bytes() == b"12345678"
    record = copy_spec("big.yaml", run_dir, project)
    assert record == {
        "path": "big.yaml", "status": "too_large", "sha256": None
    }


def test_reads_before_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    出力先を作る前に定義を読み, 写しと sha256 が学習を始めるときの中身
    であることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        出力先を作る関数の差し替えに使う

    Returns
    ----------
    None
    """
    # 出力先を作るときに定義を書き換える
    project = tmp_path / "project"
    (project / "models").mkdir(parents=True)
    spec = project / "models" / "model.yaml"
    spec.write_text(SPEC_TEXT, encoding="utf-8")
    original = setup.make_run_dir

    def rewriting_make_run_dir(*args: object) -> Path:
        """
        定義を書き換えてから出力先を作る関数

        Parameters
        ----------
        *args: object
            make_run_dir の引数

        Returns
        ----------
        run_dir: Path
            作った出力先
        """
        spec.write_text("changed: true\n", encoding="utf-8")
        run_dir = original(*args)

        return run_dir

    monkeypatch.setattr(setup, "make_run_dir", rewriting_make_run_dir)
    run_dir = run_start(
        project, monkeypatch, "model:\n  spec: models/model.yaml\n"
    )
    assert (run_dir / MODEL_SPEC_FILE).read_text(encoding="utf-8") == (
        SPEC_TEXT
    )
    assert read_record(run_dir)["sha256"] == hashlib.sha256(
        SPEC_TEXT.encode()
    ).hexdigest()


@pytest.mark.parametrize("value", ["${nope}", "${oc.env:ML_FOUND_NO_SUCH}"])
def test_unresolved_reference(tmp_path: Path, value: str) -> None:
    """
    解決できない参照の model.spec を, 写さずに知らせることを確かめる関数

    load_config は解決できない参照で止まるので, 読み込み関数を通らない
    config で確かめる

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    value: str
        model.spec に書く参照

    Returns
    ----------
    None
    """
    spec = read_model_spec(spec_config(value), tmp_path)
    record = copy_model_spec(spec, tmp_path)
    assert record is None
    message = describe_model_spec(spec, record)
    assert message is not None and "参照を解決できない" in message
    assert "\n" not in message
