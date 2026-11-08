from pathlib import Path

import pytest

from ml_found.config.loader import format_diff, load_config, save_config


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
    -------
    path_str: str
        書き出したファイルのパス
    """
    # ファイルに書き出す
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
    -------
    path: str
        default の config のパス
    """
    # ML-Found の項目とプロジェクト固有の項目を混ぜる
    text = (
        "experiment:\n"
        "  name: vae\n"
        "trainer:\n"
        "  epochs: 50\n"
        "model:\n"
        "  hidden_dim: 64\n"
    )
    path = write_yaml(tmp_path / "default.yaml", text)

    return path


def test_default_only(default_path: str):
    """
    experiment が無い場合に default がそのまま使われることを確かめる関数
    """
    # experiment を指定せずに読み込む
    cfg, diff = load_config(default_path)

    assert cfg.experiment.name == "vae"
    assert cfg.trainer.epochs == 50
    assert cfg.model.hidden_dim == 64
    assert cfg.optimizer.name == "Adam"
    assert diff == {}


def test_missing_experiment(default_path: str, tmp_path: Path):
    """
    experiment のファイルが無い場合に default のまま使われることを確かめる関数
    """
    # 存在しないパスを指定して読み込む
    cfg, diff = load_config(default_path, str(tmp_path / "none.yaml"))

    assert cfg.trainer.epochs == 50
    assert diff == {}


def test_empty_experiment(default_path: str, tmp_path: Path):
    """
    experiment が空の場合に default のまま使われることを確かめる関数
    """
    # 空の experiment を用意して読み込む
    experiment_path = write_yaml(tmp_path / "experiment.yaml", "")
    cfg, diff = load_config(default_path, experiment_path)

    assert cfg.trainer.epochs == 50
    assert diff == {}


def test_experiment_override(default_path: str, tmp_path: Path):
    """
    experiment の差分が重なり, 差分として取得できることを確かめる関数
    """
    # 値を変える項目, default と同じ値の項目, params への追加を混ぜる
    text = (
        "trainer:\n"
        "  epochs: 300\n"
        "model:\n"
        "  hidden_dim: 64\n"
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
        "optimizer.params.weight_decay": {
            "default": None, "experiment": 0.01
        }
    }
    assert "trainer.epochs: 50 -> 300" in format_diff(diff)


def test_experiment_switch_scheduler(default_path: str, tmp_path: Path):
    """
    experiment で Scheduler を使い始められることを確かめる関数
    """
    # default では使っていない Scheduler とその引数を指定する
    text = (
        "scheduler:\n"
        "  name: StepLR\n"
        "  params:\n"
        "    step_size: 10\n"
    )
    experiment_path = write_yaml(tmp_path / "experiment.yaml", text)
    cfg, _ = load_config(default_path, experiment_path)

    assert cfg.scheduler.name == "StepLR"
    assert cfg.scheduler.params.step_size == 10


@pytest.mark.parametrize(
    "text",
    [
        "trainer:\n  epoch: 10\n",
        "model:\n  hiden_dim: 32\n",
        "loss:\n  beta: 0.1\n"
    ]
)
def test_experiment_unknown_key(
    default_path: str, tmp_path: Path, text: str
):
    """
    default に無いキーを experiment に書くとエラーになることを確かめる関数
    """
    # typo を含む experiment を用意する
    experiment_path = write_yaml(tmp_path / "experiment.yaml", text)

    with pytest.raises(ValueError, match="experiment"):
        load_config(default_path, experiment_path)


def test_default_unknown_key(tmp_path: Path):
    """
    ML-Found の項目の typo を default に書くとエラーになることを確かめる関数
    """
    # ML-Found の項目に typo を含む default を用意する
    path = write_yaml(tmp_path / "default.yaml", "trainer:\n  epoch: 10\n")

    with pytest.raises(ValueError, match="default"):
        load_config(path)


def test_default_wrong_type(tmp_path: Path):
    """
    型が違う値を default に書くとエラーになることを確かめる関数
    """
    # epochs に文字列を書いた default を用意する
    path = write_yaml(tmp_path / "default.yaml", "trainer:\n  epochs: abc\n")

    with pytest.raises(ValueError, match="default"):
        load_config(path)


def test_save_config(default_path: str, tmp_path: Path):
    """
    保存した config を読み込み直すと同じ config になることを確かめる関数
    """
    # experiment を重ねて読み込む
    experiment_path = write_yaml(
        tmp_path / "experiment.yaml", "trainer:\n  epochs: 300\n"
    )
    cfg, diff = load_config(default_path, experiment_path)

    # 保存して, 保存した config を default として読み込み直す
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_config(cfg, diff, str(run_dir))
    reloaded, _ = load_config(str(run_dir / "config.yaml"))

    assert (run_dir / "diff.yaml").exists()
    assert reloaded == cfg
