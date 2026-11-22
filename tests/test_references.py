from pathlib import Path
from typing import Any, Callable, Iterator

from omegaconf import DictConfig, OmegaConf, open_dict
import pytest

from ml_found.config.loader import load_config
from ml_found.config.references import check_references
from ml_found.experiment.setup import start_experiment


def register(name: str, resolver: Callable[..., Any]) -> None:
    """
    OmegaConf の版に合わせて resolver を登録する関数

    Parameters
    ----------
    name: str
        resolver の名前
    resolver: Callable[..., Any]
        値を作る関数

    Returns
    ----------
    None
    """
    # 新しい版の登録の関数が無ければ古い版の関数で登録
    method = getattr(OmegaConf, "register_resolver", None)
    if method is None:
        method = OmegaConf.register_new_resolver
    method(name, resolver)


def fail(value: Any) -> str:
    """
    いつも誤りを出す resolver の関数

    Parameters
    ----------
    value: Any
        参照に書いた引数

    Returns
    ----------
    result: str
        返さない
    """
    raise RuntimeError(f"作れません: {value}")


def lookup(value: Any) -> str:
    """
    いつも KeyError を出す resolver の関数

    Parameters
    ----------
    value: Any
        参照に書いた引数

    Returns
    ----------
    result: str
        返さない
    """
    raise KeyError(f"Environment variable '{value}' not found")


@pytest.fixture
def resolvers() -> Iterator[None]:
    """
    テスト用の resolver を登録し, 終えたら外す関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 登録して使わせ, 他のテストに残さないように最後に解除
    register("test_double", lambda value: int(value) * 2)
    register("test_fail", fail)
    register("test_lookup", lookup)
    yield
    OmegaConf.clear_resolver("test_double")
    OmegaConf.clear_resolver("test_fail")
    OmegaConf.clear_resolver("test_lookup")


def project_config(tmp_path: Path, extra: dict) -> DictConfig:
    """
    読み込み関数で作った config に, 読み込みの後で項目を足す関数

    Parameters
    ----------
    tmp_path: Path
        プロジェクトのディレクトリ
    extra: dict
        足す項目 (読み込み関数を通らない値)

    Returns
    ----------
    cfg: DictConfig
        学習に使う config
    """
    # 正しい config を読み込んでから, 確かめを通らない値を書き込み
    default_path = tmp_path / "default.yaml"
    default_path.write_text("experiment:\n  device: cpu\n", encoding="utf-8")
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text("", encoding="utf-8")
    cfg, _ = load_config(str(default_path), str(experiment_path))
    with open_dict(cfg):
        for key, value in extra.items():
            cfg[key] = value

    return cfg


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"model": {"dim": "${data.dim}"}},
         r"model.dim: '\$\{data.dim\}' \(参照する項目がありません\)"),
        ({"paths": ["ok", "${nope}"]},
         r"paths\[1\]: '\$\{nope\}' \(参照する項目がありません\)"),
        ({"token": "${oc.env:ML_FOUND_NO_SUCH}"},
         r"token: .*\(環境変数がありません"),
        ({"size": "${no_such_resolver:1}"},
         r"size: .*\(登録されていない resolver です"),
        ({"size": "${test_fail:1}"},
         r"size: .*\(resolver が誤り \(RuntimeError\) を出しました\)"),
        ({"size": "${test_lookup:${oc.env:ML_FOUND_NO_SUCH,HOME}}"},
         r"size: .*\(resolver が誤り \(KeyError\) を出しました\)"),
        ({"a": "${b}", "b": "${a}"},
         r"a: '\$\{b\}' \(参照が循環しています\)\n\tb: '\$\{a\}'"),
        ({"a": "???", "b": "${a}"},
         r"b: '\$\{a\}' \(参照する項目の値が決まっていません")
    ]
)
def test_unresolved_stops_before_run_dir(
    tmp_path: Path, resolvers: None, extra: dict, message: str
) -> None:
    """
    解決できない参照のある config で, 出力先を作る前に項目の場所と参照を
    添えたエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    resolvers: None
        テスト用の resolver を登録する fixture
    extra: dict
        読み込みの後で足す項目
    message: str
        エラーの文に含まれるべき形

    Returns
    ----------
    None
    """
    cfg = project_config(tmp_path, extra)
    with pytest.raises(ValueError, match=message):
        start_experiment(cfg, {}, str(tmp_path))
    assert not (tmp_path / "outputs").exists()


def test_resolvable_references(
    tmp_path: Path, resolvers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    解決できる参照 (項目, 既定値付きの環境変数, 登録した resolver, 相対の
    参照) は止めずに解決した値を保存することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    resolvers: None
        テスト用の resolver を登録する fixture
    monkeypatch: pytest.MonkeyPatch
        環境変数を設定するのに使う

    Returns
    ----------
    None
    """
    monkeypatch.setenv("ML_FOUND_TEST_DIR", "data")
    cfg = project_config(tmp_path, {
        "data": {
            "dir": "${oc.env:ML_FOUND_TEST_DIR}",
            "cache": "${oc.env:ML_FOUND_NO_SUCH,cache}",
            "dim": "${test_double:4}",
            "same": "${.dim}"
        },
        "model": {"dim": "${data.dim}"},
        "missing": "???"
    })
    experiment = start_experiment(cfg, {}, str(tmp_path))
    saved = OmegaConf.load(experiment.run_dir / "config.yaml")
    assert saved.data.dir == "data"
    assert saved.data.cache == "cache"
    assert saved.model.dim == 8
    assert saved.data.same == 8


def test_secret_not_in_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    環境変数の値がエラーの文に出ないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        環境変数を設定するのに使う

    Returns
    ----------
    None
    """
    # 環境変数の値で項目を選ぶ参照 (OmegaConf の文は値を含む)
    monkeypatch.setenv("ML_FOUND_TEST_SECRET", "s3cr3t-value")
    cfg = project_config(
        tmp_path, {"key": "${${oc.env:ML_FOUND_TEST_SECRET}}"}
    )
    with pytest.raises(ValueError) as info:
        start_experiment(cfg, {}, str(tmp_path))
    assert "参照する項目がありません" in str(info.value)
    assert "s3cr3t-value" not in str(info.value)
    assert "s3cr3t-value" not in repr(info.value.__cause__)


def test_load_config_names_item(tmp_path: Path) -> None:
    """
    読み込み関数でも, 解決できない参照を項目の場所を添えて知らせることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    default_path = tmp_path / "default.yaml"
    default_path.write_text(
        "experiment:\n  device: cpu\nmodel:\n  dim: 4\n", encoding="utf-8"
    )
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text("model:\n  dim: ${nope}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"^config の参照.*\n\tmodel.dim"):
        load_config(str(default_path), str(experiment_path))


def test_check_references_label() -> None:
    """
    誤りの文が渡した名前で config を指すことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    cfg = OmegaConf.create({"a": "${b}"})
    with pytest.raises(ValueError, match="^default の config の参照"):
        check_references(cfg, "default の config")
    check_references(OmegaConf.create({"a": 1, "b": "${a}"}))
