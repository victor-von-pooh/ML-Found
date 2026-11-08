from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ml_found.config.schema import Config


def load_yaml(path: str) -> DictConfig:
    """
    YAML ファイルを読み込む関数

    Parameters
    ----------
    path: str
        YAML ファイルのパス

    Returns
    -------
    cfg: DictConfig
        読み込んだ config, 空のファイルの場合は空の config
    """
    # ファイルを読み込む
    cfg = OmegaConf.load(path)

    # 空のファイルは None になるので空の config にする
    if cfg is None:
        cfg = OmegaConf.create({})

    return cfg


def unlock_params(cfg: DictConfig) -> DictConfig:
    """
    torch に渡す引数の項目だけはキーを追加できるようにする関数

    Parameters
    ----------
    cfg: DictConfig
        存在しないキーを禁止した config

    Returns
    -------
    cfg: DictConfig
        optimizer と scheduler の params だけキーを追加できるようにした config
    """
    # Optimizer や Scheduler を切り替えると必要な引数も変わるので自由に追加させる
    OmegaConf.set_struct(cfg.optimizer.params, False)
    OmegaConf.set_struct(cfg.scheduler.params, False)

    return cfg


def build_default(default_path: str) -> DictConfig:
    """
    ML-Found の既定値にプロジェクトの default を重ねる関数

    Parameters
    ----------
    default_path: str
        プロジェクトの default の config のパス

    Returns
    -------
    default_cfg: DictConfig
        ML-Found の既定値に default を重ねた config
    """
    # ML-Found の既定値を作る
    schema = OmegaConf.structured(Config)

    # 最上位には model や data などプロジェクト固有の項目を追加できるようにする
    OmegaConf.set_struct(schema, False)

    # ML-Found が使う項目は typo を見つけられるように存在しないキーを禁止する
    for key in schema.keys():
        OmegaConf.set_struct(schema[key], True)
    schema = unlock_params(schema)

    # プロジェクトの default を重ねる
    try:
        default_cfg = OmegaConf.merge(schema, load_yaml(default_path))
    except OmegaConfBaseException as e:
        raise ValueError(f"default の config が正しくありません: {e}") from e

    return default_cfg


def merge_experiment(
    default_cfg: DictConfig, experiment_path: Optional[str] = None
) -> DictConfig:
    """
    default に experiment の差分を重ねる関数

    Parameters
    ----------
    default_cfg: DictConfig
        ML-Found の既定値に default を重ねた config
    experiment_path: Optional[str] = None
        experiment の config のパス, null やファイルが無い場合は default のまま

    Returns
    -------
    cfg: DictConfig
        学習に使う config
    """
    # experiment が無い場合は default をそのまま使う
    if experiment_path is None or not Path(experiment_path).exists():
        cfg = default_cfg.copy()
        return cfg

    # default に無いキーを experiment で書けないようにする
    base = default_cfg.copy()
    OmegaConf.set_struct(base, True)
    base = unlock_params(base)

    # experiment の差分を重ねる
    try:
        cfg = OmegaConf.merge(base, load_yaml(experiment_path))
    except OmegaConfBaseException as e:
        raise ValueError(
            f"experiment の config が正しくありません: {e}"
        ) from e

    return cfg


def flatten(data: Any, prefix: str = "") -> dict:
    """
    入れ子の辞書を . でつないだキーの辞書に平らにする関数

    Parameters
    ----------
    data: Any
        平らにする辞書, 辞書以外の場合はそのまま値として扱う
    prefix: str = ""
        親のキー

    Returns
    -------
    flat: dict
        . でつないだキーと値の辞書
    """
    # 辞書以外の場合はそこで止める
    if not isinstance(data, dict):
        flat = {prefix: data}
        return flat

    # 子の辞書を順に平らにしてまとめる
    flat = {}
    for key, value in data.items():
        child_key = f"{prefix}.{key}" if prefix else str(key)
        flat.update(flatten(value, child_key))

    return flat


def get_diff(default_cfg: DictConfig, cfg: DictConfig) -> dict:
    """
    default から変わった項目を取得する関数

    Parameters
    ----------
    default_cfg: DictConfig
        ML-Found の既定値に default を重ねた config
    cfg: DictConfig
        学習に使う config

    Returns
    -------
    diff: dict
        変わった項目ごとの default の値と experiment の値
    """
    # 比べやすいように両方を平らにする
    default_flat = flatten(OmegaConf.to_container(default_cfg, resolve=True))
    cfg_flat = flatten(OmegaConf.to_container(cfg, resolve=True))

    # 値が変わった項目と experiment で追加された項目を集める
    diff = {}
    for key, value in cfg_flat.items():
        if key not in default_flat or default_flat[key] != value:
            diff[key] = {
                "default": default_flat.get(key),
                "experiment": value
            }

    return diff


def format_diff(diff: dict) -> str:
    """
    差分をログに書き出せる文字列にする関数

    Parameters
    ----------
    diff: dict
        変わった項目ごとの default の値と experiment の値

    Returns
    -------
    diff_str: str
        1 項目 1 行の差分
    """
    # 差分が無い場合はその旨を返す
    if len(diff) == 0:
        diff_str = "\tdefault との差分はありません\n"
        return diff_str

    # 1 項目ずつ default の値と experiment の値を並べる
    diff_str = "".join(
        f"\t{key}: {values['default']} -> {values['experiment']}\n"
        for key, values in diff.items()
    )

    return diff_str


def load_config(
    default_path: str, experiment_path: Optional[str] = None
) -> tuple[DictConfig, dict]:
    """
    default と experiment を読み込んで学習に使う config を作る関数

    Parameters
    ----------
    default_path: str
        プロジェクトの default の config のパス
    experiment_path: Optional[str] = None
        experiment の config のパス, null やファイルが無い場合は default のまま

    Returns
    -------
    cfg: DictConfig
        学習に使う config
    diff: dict
        default から変わった項目
    """
    # ML-Found の既定値にプロジェクトの default を重ねる
    default_cfg = build_default(default_path)

    # experiment の差分を重ねる
    cfg = merge_experiment(default_cfg, experiment_path)

    # default から変わった項目を取得する
    diff = get_diff(default_cfg, cfg)

    return cfg, diff


def save_config(cfg: DictConfig, diff: dict, run_dir: str) -> None:
    """
    学習に使った config と差分を保存する関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config
    diff: dict
        default から変わった項目
    run_dir: str
        保存先のディレクトリ

    Returns
    -------
    None
    """
    # 後から同じ条件で学習できるように config 全体を保存する
    OmegaConf.save(cfg, Path(run_dir) / "config.yaml", resolve=True)

    # どこを変えた実験なのかが分かるように差分を保存する
    OmegaConf.save(OmegaConf.create(diff), Path(run_dir) / "diff.yaml")
