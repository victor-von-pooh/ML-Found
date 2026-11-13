from pathlib import Path

import pytest

from ml_found.config.loader import (
    ABSENT, format_diff, load_config, load_project_config, save_config
)


def write_yaml(path: Path, text: str) -> str:
    """
    テスト用の YAML ファイルを書き出す関数

    Parameters
    ----------
    path: Path
        書き出すファイルのパス
    text: str
        YAML の内容

    Returns
    ----------
    path_str: str
        書き出したファイルのパス
    """
    # ファイルに書き出し
    path.write_text(text, encoding="utf-8")
    path_str = str(path)

    return path_str


@pytest.fixture
def default_path(tmp_path: Path) -> str:
    """
    プロジェクト固有の項目を含む default を用意する関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    path: str
        default の config のパス
    """
    # ML-Found の項目とプロジェクト固有の項目の混在
    text = (
        "experiment:\n"
        "  name: vae\n"
        "trainer:\n"
        "  epochs: 50\n"
        "scheduler:\n"
        "  name: StepLR\n"
        "  params:\n"
        "    step_size: 10\n"
        "    gamma: 0.5\n"
        "model:\n"
        "  hidden_dim: 64\n"
        "  encoder:\n"
        "    layers: 2\n"
        "    act: relu\n"
        "  pretrained: null\n"
    )
    path = write_yaml(tmp_path / "default.yaml", text)

    return path


def test_empty_experiment(default_path: str, tmp_path: Path) -> None:
    """
    experiment が空の場合に default がそのまま使われることを確かめる関数

    Parameters
    ----------
    default_path: str
        default の config のパス
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 空の experiment を用意して読み込み
    experiment_path = write_yaml(tmp_path / "experiment.yaml", "")
    cfg, diff = load_config(default_path, experiment_path)

    assert cfg.experiment.name == "vae"
    assert cfg.trainer.epochs == 50
    assert cfg.model.hidden_dim == 64
    assert cfg.optimizer.name == "Adam"
    assert diff == {}


def test_missing_file(default_path: str, tmp_path: Path) -> None:
    """
    default や experiment のファイルが無い場合にエラーになることを確かめる関数

    Parameters
    ----------
    default_path: str
        default の config のパス
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # experiment が無い場合
    with pytest.raises(FileNotFoundError, match="none.yaml"):
        load_config(default_path, str(tmp_path / "none.yaml"))

    # default が無い場合
    experiment_path = write_yaml(tmp_path / "experiment.yaml", "")
    with pytest.raises(FileNotFoundError, match="none.yaml"):
        load_config(str(tmp_path / "none.yaml"), experiment_path)


def test_experiment_override(default_path: str, tmp_path: Path) -> None:
    """
    experiment の差分が重なり, 差分として取得できることを確かめる関数

    Parameters
    ----------
    default_path: str
        default の config のパス
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 値を変える項目, default と同じ値の項目, null の項目, params への追加の混在
    text = (
        "trainer:\n"
        "  epochs: 300\n"
        "model:\n"
        "  hidden_dim: 64\n"
        "  pretrained: weights.pt\n"
        "optimizer:\n"
        "  params:\n"
        "    weight_decay: 0.01\n"
    )
    experiment_path = write_yaml(tmp_path / "experiment.yaml", text)
    cfg, diff = load_config(default_path, experiment_path)

    assert cfg.trainer.epochs == 300
    assert cfg.optimizer.params.lr == 1e-3
    assert cfg.optimizer.params.weight_decay == 0.01
    assert diff == {
        "trainer.epochs": {"default": 50, "experiment": 300},
        "model.pretrained": {"default": None, "experiment": "weights.pt"},
        "optimizer.params.weight_decay": {
            "default": ABSENT, "experiment": 0.01
        }
    }
    assert "trainer.epochs: 50 -> 300" in format_diff(diff)


def test_same_class_keeps_params(default_path: str, tmp_path: Path) -> None:
    """
    同じクラスのまま引数を変える場合は他の引数が残ることを確かめる関数

    Parameters
    ----------
    default_path: str
        default の config のパス
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # StepLR のまま gamma だけを変更
    text = "scheduler:\n  params:\n    gamma: 0.1\n"
    experiment_path = write_yaml(tmp_path / "experiment.yaml", text)
    cfg, _ = load_config(default_path, experiment_path)

    assert dict(cfg.scheduler.params) == {"step_size": 10, "gamma": 0.1}


def test_switch_class_resets_params(
    default_path: str, tmp_path: Path
) -> None:
    """
    クラスを変える場合は default の引数が捨てられることを確かめる関数

    Parameters
    ----------
    default_path: str
        default の config のパス
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # StepLR から CosineAnnealingLR に変更
    text = (
        "scheduler:\n"
        "  name: CosineAnnealingLR\n"
        "  params:\n"
        "    T_max: 10\n"
    )
    experiment_path = write_yaml(tmp_path / "experiment.yaml", text)
    cfg, diff = load_config(default_path, experiment_path)

    assert dict(cfg.scheduler.params) == {"T_max": 10}
    assert diff["scheduler.params.step_size"] == {
        "default": 10, "experiment": ABSENT
    }


@pytest.mark.parametrize(
    "text",
    [
        "trainer:\n  epoch: 10\n",
        "model:\n  hiden_dim: 32\n",
        "loss:\n  beta: 0.1\n",
        "model:\n  pretrained:\n    path: a.pt\n",
        "model:\n  hidden_dim:\n    a: 1\n",
        "model:\n  encoder: 5\n",
        "model:\n  encoder: null\n"
    ]
)
def test_experiment_invalid_structure(
    default_path: str, tmp_path: Path, text: str
) -> None:
    """
    experiment で項目を追加したり構造を変えたりするとエラーになることを確かめる関数

    Parameters
    ----------
    default_path: str
        default の config のパス
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    text: str
        typo や構造の変更を含む experiment の内容

    Returns
    ----------
    None
    """
    # 正しくない experiment を用意
    experiment_path = write_yaml(tmp_path / "experiment.yaml", text)

    with pytest.raises(ValueError, match="experiment"):
        load_config(default_path, experiment_path)


def test_default_invalid(tmp_path: Path) -> None:
    """
    ML-Found の項目の typo や型違いを default に書くとエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 空の experiment を用意
    experiment_path = write_yaml(tmp_path / "experiment.yaml", "")

    # typo の場合
    path = write_yaml(tmp_path / "default.yaml", "trainer:\n  epoch: 10\n")
    with pytest.raises(ValueError, match="default"):
        load_config(path, experiment_path)

    # 型違いの場合
    path = write_yaml(tmp_path / "default.yaml", "trainer:\n  epochs: abc\n")
    with pytest.raises(ValueError, match="default"):
        load_config(path, experiment_path)


def test_unresolvable_interpolation(tmp_path: Path) -> None:
    """
    解決できない ${...} がある場合に分かりやすいエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 存在しない項目を参照する default を用意
    path = write_yaml(
        tmp_path / "default.yaml", "model:\n  dim: ${data.dim}\n"
    )
    experiment_path = write_yaml(tmp_path / "experiment.yaml", "")

    with pytest.raises(ValueError, match="解決できません"):
        load_config(path, experiment_path)


def test_save_config(default_path: str, tmp_path: Path) -> None:
    """
    保存した config を読み込み直すと同じ config になることを確かめる関数

    Parameters
    ----------
    default_path: str
        default の config のパス
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # experiment を重ねて読み込み
    experiment_path = write_yaml(
        tmp_path / "experiment.yaml", "trainer:\n  epochs: 300\n"
    )
    cfg, diff = load_config(default_path, experiment_path)

    # 保存して, 保存した config を default として再読み込み
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_config(cfg, diff, str(run_dir))
    empty_path = write_yaml(tmp_path / "empty.yaml", "")
    reloaded, _ = load_config(str(run_dir / "config.yaml"), empty_path)

    assert (run_dir / "diff.yaml").exists()
    assert reloaded == cfg


def test_overrides(default_path: str, tmp_path: Path) -> None:
    """
    「項目=値」の上書きが experiment の後に重なり, 差分に残ることを確かめる関数

    Parameters
    ----------
    default_path: str
        default の config のパス
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = write_yaml(
        tmp_path / "experiment.yaml", "trainer:\n  epochs: 20\n"
    )
    cfg, diff = load_config(
        default_path, experiment,
        ["trainer.epochs=10", "model.encoder.act=gelu", "scheduler.name=null"]
    )
    assert cfg.trainer.epochs == 10
    assert cfg.model.encoder.act == "gelu"
    assert cfg.scheduler.name is None
    assert diff["trainer.epochs"]["experiment"] == 10

    # default に無い項目と, = の無いものはエラー
    with pytest.raises(ValueError, match="model.dropout"):
        load_config(default_path, experiment, ["model.dropout=0.1"])
    with pytest.raises(ValueError, match="項目=値"):
        load_config(default_path, experiment, ["trainer.epochs"])

    # クラスを変える上書きは, experiment と同じく default の引数を捨てる
    cfg, _ = load_config(
        default_path, experiment,
        ["scheduler.name=ExponentialLR", "scheduler.params.gamma=0.9"]
    )
    assert dict(cfg.scheduler.params) == {"gamma": 0.9}


def test_load_project_config(
    default_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    コマンドライン引数から, プロジェクトの config を作ることを確かめる関数

    Parameters
    ----------
    default_path: str
        default の config のパス (tmp_path/default.yaml)
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        実行した場所を変える道具

    Returns
    ----------
    None
    """
    # プロジェクトの構成を用意
    project = tmp_path / "project"
    (project / "configs/default").mkdir(parents=True)
    (project / "configs/experiment/examples").mkdir(parents=True)
    Path(default_path).rename(project / "configs/default/train.yaml")
    write_yaml(project / "configs/experiment/train.yaml", "")
    write_yaml(
        project / "configs/experiment/examples/short.yaml",
        "trainer:\n  epochs: 5\n"
    )

    # 既定の experiment
    cfg, diff = load_project_config(str(project), [])
    assert cfg.trainer.epochs == 50
    assert diff == {}

    # 相対パスはプロジェクトのディレクトリから, 上書きは最後に重ねる
    monkeypatch.chdir(tmp_path)
    cfg, diff = load_project_config(
        str(project),
        ["--experiment", "configs/experiment/examples/short.yaml",
         "model.hidden_dim=128"]
    )
    assert cfg.trainer.epochs == 5
    assert cfg.model.hidden_dim == 128
    assert set(diff) == {"trainer.epochs", "model.hidden_dim"}

    # プロジェクトに無い相対パスは, 実行した場所から辿る
    write_yaml(tmp_path / "here.yaml", "trainer:\n  epochs: 7\n")
    cfg, _ = load_project_config(str(project), ["--experiment", "here.yaml"])
    assert cfg.trainer.epochs == 7
