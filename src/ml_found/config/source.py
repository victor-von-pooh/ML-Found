from pathlib import Path
import sys
from typing import Any, Optional, Sequence
import weakref

from omegaconf import DictConfig, OmegaConf

# 実験の起動のしかたを書くファイルの名前
SOURCE_FILE = "source.yaml"

# 読み込んだ config ごとの, default と experiment の絶対パスと上書き
# (config の id をキーにし, config が消えたら一緒に削除.
# DictConfig のハッシュは中身から求めるので, 中身を変えても引けるように id を使用)
SOURCES: dict[int, dict] = {}


def remember_source(
    cfg: DictConfig, default_path: str, experiment_path: str,
    overrides: Optional[Sequence[str]]
) -> None:
    """
    config を作るのに使ったファイルと上書きを, config に結び付けて覚える関数

    Parameters
    ----------
    cfg: DictConfig
        作った config
    default_path: str
        default の config のパス
    experiment_path: str
        experiment の config のパス
    overrides: Optional[Sequence[str]]
        「項目=値」の上書き

    Returns
    ----------
    None
    """
    # 読み込んだ時点の作業ディレクトリから辿った絶対パスで保持
    key = id(cfg)
    SOURCES[key] = {
        "default": Path(default_path).expanduser().resolve(),
        "experiment": Path(experiment_path).expanduser().resolve(),
        "overrides": [str(item) for item in overrides or []]
    }

    # config が消えた後に同じ id の別の config と取り違えないように削除
    weakref.finalize(cfg, SOURCES.pop, key, None)


def relative_to_project(path: Path, project_dir: Path) -> str:
    """
    パスを, プロジェクトのディレクトリの中ならそこからの相対パスにする関数

    Parameters
    ----------
    path: Path
        絶対パス
    project_dir: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    relative: str
        プロジェクトの中なら相対パス, 外なら絶対パス (区切りは / )
    """
    # プロジェクトの外のパスは絶対パスのまま使用
    try:
        relative = path.relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        relative = path.as_posix()

    return relative


def script_path(project_dir: Path) -> Optional[str]:
    """
    実行したスクリプトのパスを取得する関数

    Parameters
    ----------
    project_dir: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    script: Optional[str]
        スクリプトのパス (プロジェクトの中なら相対パス). ノートブックの中と,
        python -c のようにファイルが無い場合は None
    """
    # ノートブックの sys.argv[0] はカーネルのものなので対象外
    # 引数の残り (sys.argv[1:]) は秘密の値を含み得るので読まない
    if "ipykernel" in sys.modules or not sys.argv:
        return None
    path = Path(sys.argv[0])
    if not path.is_file():
        return None
    script = relative_to_project(path.resolve(), project_dir)

    return script


def build_source(cfg: DictConfig, project_dir: Path) -> dict:
    """
    実験の起動のしかたをまとめる関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config
    project_dir: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    source: dict
        script, default, experiment, overrides. load_config などを使わずに
        作った config では, script 以外は None
    """
    # 読み込み関数が覚えた情報を, プロジェクトからの相対パスに変換
    remembered = SOURCES.get(id(cfg))
    source: dict[str, Any] = {
        "script": script_path(project_dir),
        "default": None,
        "experiment": None,
        "overrides": None
    }
    if remembered is not None:
        source["default"] = relative_to_project(
            remembered["default"], project_dir
        )
        source["experiment"] = relative_to_project(
            remembered["experiment"], project_dir
        )
        source["overrides"] = list(remembered["overrides"])

    return source


def save_source(cfg: DictConfig, run_dir: str, project_dir: Path) -> None:
    """
    実験の起動のしかたを source.yaml に保存する関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config
    run_dir: str
        保存先のディレクトリ
    project_dir: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    None
    """
    # 既にある場合は, 分からない項目 (None) だけを今回の値で埋めて情報を保持
    path = Path(run_dir) / SOURCE_FILE
    source = build_source(cfg, project_dir)
    if path.exists():
        existing = OmegaConf.to_container(OmegaConf.load(path))
        if isinstance(existing, dict):
            for key, value in existing.items():
                if value is not None or key not in source:
                    source[key] = value

    # 上書きの値の ${...} を補間として扱わないように, 辞書のまま保存
    OmegaConf.save(OmegaConf.create(source), path)
