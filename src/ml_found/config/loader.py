import argparse
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ml_found.config.schema import Config

# 差分で, 片方にしか存在しない項目の値として使う印
ABSENT = "<未定義>"

# torch のクラスに渡す引数の項目 (クラスを変えると必要な引数も変化)
PARAMS_SECTIONS = ("optimizer", "scheduler")


def load_yaml(path: str) -> DictConfig:
    """
    YAML ファイルを読み込む関数

    Parameters
    ----------
    path: str
        YAML ファイルのパス

    Returns
    ----------
    cfg: DictConfig
        読み込んだ config, 空のファイルの場合は空の config
    """
    # ファイルが無い場合は分かりやすいエラー
    if not Path(path).exists():
        raise FileNotFoundError(f"config ファイルがありません: {path}")

    # ファイルの読み込み
    cfg = OmegaConf.load(path)

    # 空のファイルは None になるので空の config に変換
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
    ----------
    cfg: DictConfig
        optimizer と scheduler の params だけキーを追加できるようにした config
    """
    # Optimizer や Scheduler を切り替えると必要な引数も変わるので自由に追加可能に設定
    for section in PARAMS_SECTIONS:
        OmegaConf.set_struct(cfg[section].params, False)

    return cfg


def build_default(default_path: str) -> DictConfig:
    """
    ML-Found の既定値にプロジェクトの default を重ねる関数

    Parameters
    ----------
    default_path: str
        プロジェクトの default の config のパス

    Returns
    ----------
    default_cfg: DictConfig
        ML-Found の既定値に default を重ねた config
    """
    # ML-Found の既定値の作成
    schema = OmegaConf.structured(Config)

    # 最上位には model や data などプロジェクト固有の項目を追加できるように設定
    OmegaConf.set_struct(schema, False)

    # ML-Found が使う項目は typo を見つけられるように存在しないキーを禁止
    for key in schema.keys():
        OmegaConf.set_struct(schema[key], True)
    schema = unlock_params(schema)

    # プロジェクトの default のマージ
    try:
        default_cfg = OmegaConf.merge(schema, load_yaml(default_path))
    except OmegaConfBaseException as e:
        raise ValueError(f"default の config が正しくありません: {e}") from e

    return default_cfg


def check_structure(
    experiment: Any, default: Any, prefix: str = ""
) -> None:
    """
    experiment が default に無い項目を追加していないかを確かめる関数

    Parameters
    ----------
    experiment: Any
        experiment の値, 辞書の場合は中を順に調べる
    default: Any
        同じ位置の default の値
    prefix: str = ""
        親のキー

    Returns
    ----------
    None
    """
    # torch に渡す引数の項目は自由に書けるので確認不要
    if prefix in [f"{section}.params" for section in PARAMS_SECTIONS]:
        return

    # 辞書と辞書以外を入れ替えると項目の追加や削除になるのでエラー
    if isinstance(experiment, dict) != isinstance(default, dict):
        raise ValueError(
            f"experiment で {prefix} の構造は変えられません: "
            f"default は {default}, experiment は {experiment}"
        )

    # 辞書の場合は各キーが default にあるかを調べ, 中も順に確認
    if isinstance(experiment, dict):
        for key, value in experiment.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            if key not in default:
                raise ValueError(
                    f"experiment に default に無い項目があります: {child_key}"
                )
            check_structure(value, default[key], child_key)


def reset_params(base: DictConfig, experiment: DictConfig) -> DictConfig:
    """
    experiment でクラスを変える場合に default の引数を捨てる関数

    Parameters
    ----------
    base: DictConfig
        experiment を重ねる前の config
    experiment: DictConfig
        experiment の config

    Returns
    ----------
    base: DictConfig
        クラスを変える項目だけ params を空にした config
    """
    # クラスを変えると default の引数はそのクラスで使えないことがあるので空に初期化
    for section in PARAMS_SECTIONS:
        name = OmegaConf.select(experiment, f"{section}.name", default=None)
        if name is not None and name != base[section].name:
            base[section].params = {}

    return base


def parse_overrides(overrides: Sequence[str]) -> DictConfig:
    """
    「項目=値」の形の上書き (trainer.epochs=10 など) を config にする関数

    Parameters
    ----------
    overrides: Sequence[str]
        「項目=値」の文字列, 値は YAML として読む

    Returns
    ----------
    cfg: DictConfig
        上書きする項目と値
    """
    # = の無いものは, 何を書き間違えたか分かるようにエラー
    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"上書きは「項目=値」の形で書いてください: {item} "
                "(例: trainer.epochs=10. ~ や + の書き方には未対応)"
            )

    # 値は YAML として読み, 読めない場合は分かりやすいエラー
    try:
        cfg = OmegaConf.from_dotlist(list(overrides))
    except OmegaConfBaseException as e:
        raise ValueError(f"上書きを読めません: {e}") from e

    return cfg


def merge_experiment(
    default_cfg: DictConfig, experiment_path: str,
    overrides: Optional[Sequence[str]] = None
) -> DictConfig:
    """
    default に experiment の差分を重ねる関数

    Parameters
    ----------
    default_cfg: DictConfig
        ML-Found の既定値に default を重ねた config
    experiment_path: str
        experiment の config のパス
    overrides: Optional[Sequence[str]] = None
        experiment の後に重ねる「項目=値」の上書き

    Returns
    ----------
    cfg: DictConfig
        学習に使う config
    """
    # experiment と上書きを重ね, default に無い項目を追加していないかの確認
    # 上書きでクラス名を変える場合は, experiment に書いた引数も捨てる
    experiment = load_yaml(experiment_path)
    if overrides:
        override = parse_overrides(overrides)
        for section in PARAMS_SECTIONS:
            name = OmegaConf.select(override, f"{section}.name", default=None)
            old = OmegaConf.select(experiment, f"{section}.name", default=None)
            if name is not None and name != old and OmegaConf.select(
                experiment, f"{section}.params", default=None
            ) is not None:
                experiment[section].params = {}
        experiment = OmegaConf.merge(experiment, override)
    check_structure(
        OmegaConf.to_container(experiment),
        OmegaConf.to_container(default_cfg)
    )

    # default に無いキーを experiment で書けないように設定
    base = default_cfg.copy()
    OmegaConf.set_struct(base, True)

    # クラスを変える項目は default の引数を捨て, params だけキーを追加できるように設定
    base = reset_params(base, experiment)
    base = unlock_params(base)

    # experiment の差分のマージ
    try:
        cfg = OmegaConf.merge(base, experiment)
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
    ----------
    flat: dict
        . でつないだキーと値の辞書
    """
    # 辞書以外の場合はそこで停止
    if not isinstance(data, dict):
        flat = {prefix: data}
        return flat

    # 子の辞書を順に平らにして集約
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
    ----------
    diff: dict
        変わった項目ごとの default の値と experiment の値
    """
    # 比べやすいように両方の平坦化
    try:
        default_flat = flatten(
            OmegaConf.to_container(default_cfg, resolve=True)
        )
        cfg_flat = flatten(OmegaConf.to_container(cfg, resolve=True))
    except OmegaConfBaseException as e:
        raise ValueError(f"config の ${{...}} を解決できません: {e}") from e

    # 追加, 削除, 変更された項目を両方のキーから収集
    keys = list(default_flat) + [k for k in cfg_flat if k not in default_flat]
    diff = {}
    for key in keys:
        default_value = default_flat.get(key, ABSENT)
        value = cfg_flat.get(key, ABSENT)
        if default_value != value:
            diff[key] = {"default": default_value, "experiment": value}

    return diff


def format_diff(diff: dict) -> str:
    """
    差分をログに書き出せる文字列にする関数

    Parameters
    ----------
    diff: dict
        変わった項目ごとの default の値と experiment の値

    Returns
    ----------
    diff_str: str
        1 項目 1 行の差分
    """
    # 差分が無い場合はその旨を返却
    if len(diff) == 0:
        diff_str = "\tdefault との差分はありません\n"
        return diff_str

    # 1 項目ずつ default の値と experiment の値の列挙
    diff_str = "".join(
        f"\t{key}: {values['default']} -> {values['experiment']}\n"
        for key, values in diff.items()
    )

    return diff_str


def load_config(
    default_path: str, experiment_path: str,
    overrides: Optional[Sequence[str]] = None
) -> tuple[DictConfig, dict]:
    """
    default と experiment を読み込んで学習に使う config を作る関数

    Parameters
    ----------
    default_path: str
        プロジェクトの default の config のパス
    experiment_path: str
        experiment の config のパス, 差分が無い場合も空のファイルを置く
    overrides: Optional[Sequence[str]] = None
        experiment の後に重ねる「項目=値」の上書き (trainer.epochs=10 など).
        experiment と同じく default にある項目だけを書け, 差分にも残る

    Returns
    ----------
    cfg: DictConfig
        学習に使う config
    diff: dict
        default から変わった項目
    """
    # ML-Found の既定値にプロジェクトの default のマージ
    default_cfg = build_default(default_path)

    # experiment の差分と上書きのマージ
    cfg = merge_experiment(default_cfg, experiment_path, overrides)

    # default から変わった項目を取得
    diff = get_diff(default_cfg, cfg)

    return cfg, diff


def resolve_path(path: str, project_dir: Path) -> Path:
    """
    config のパスを, プロジェクトのディレクトリか実行した場所から解決する関数

    Parameters
    ----------
    path: str
        コマンドラインで指定したパス
    project_dir: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    resolved: Path
        絶対パス, 相対パスはプロジェクトのディレクトリから辿り, 無ければ
        実行した場所から辿る
    """
    # 絶対パスの場合はそのまま使用
    given = Path(path).expanduser()
    if given.is_absolute():
        resolved = given
        return resolved

    # プロジェクトのディレクトリから辿り, 無い場合は実行した場所から探索
    resolved = project_dir / given
    if not resolved.exists() and given.exists():
        resolved = given.resolve()

    return resolved


def add_config_arguments(
    parser: argparse.ArgumentParser,
    experiment: str = "configs/experiment/train.yaml"
) -> argparse.ArgumentParser:
    """
    experiment の config のパスと「項目=値」の上書きを受け取る引数を加える関数

    独自の引数を持つ学習スクリプトは, 自分の parser にこの関数で加え,
    読み込んだ引数を load_config_from_args に渡す

    Parameters
    ----------
    parser: argparse.ArgumentParser
        引数を加える parser
    experiment: str = "configs/experiment/train.yaml"
        --experiment を指定しない場合の experiment の config のパス

    Returns
    ----------
    parser: argparse.ArgumentParser
        --experiment と上書き (overrides) を加えた parser
    """
    parser.add_argument(
        "--experiment", default=experiment,
        help=(
            "experiment の config のパス, 相対パスはプロジェクトのディレクトリ"
            "から辿り, 無い場合は実行した場所から辿る (既定: %(default)s)"
        )
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="experiment の後に重ねる「項目=値」(例: trainer.epochs=10)"
    )

    return parser


def load_config_from_args(
    project_dir: str, args: argparse.Namespace,
    default: str = "configs/default/train.yaml"
) -> tuple[DictConfig, dict]:
    """
    add_config_arguments の引数を読んだ結果から, 学習に使う config を作る関数

    Parameters
    ----------
    project_dir: str
        プロジェクトのディレクトリ (相対パスの基準)
    args: argparse.Namespace
        experiment と overrides を持つ, 読み込んだ引数
    default: str = "configs/default/train.yaml"
        default の config のパス, 相対パスはプロジェクトのディレクトリから

    Returns
    ----------
    cfg: DictConfig
        学習に使う config
    diff: dict
        default から変わった項目 (上書きも含む)
    """
    # default はプロジェクトのもの, experiment は実行した場所からも探索
    root = Path(project_dir).resolve()
    default_path = Path(default).expanduser()
    if not default_path.is_absolute():
        default_path = root / default_path
    cfg, diff = load_config(
        str(default_path), str(resolve_path(args.experiment, root)),
        args.overrides
    )

    return cfg, diff


def load_project_config(
    project_dir: str, argv: Optional[Sequence[str]] = None,
    default: str = "configs/default/train.yaml",
    experiment: str = "configs/experiment/train.yaml"
) -> tuple[DictConfig, dict]:
    """
    コマンドライン引数から, プロジェクトの学習に使う config を作る関数

    `python train.py --experiment configs/experiment/examples/sgd.yaml
    trainer.epochs=10` のように, experiment の config のパスと「項目=値」の
    上書きを受け取る. 順番は自由. ノートブックの中では sys.argv を読まず,
    引数が無いものとして扱う

    Parameters
    ----------
    project_dir: str
        プロジェクトのディレクトリ (相対パスの基準)
    argv: Optional[Sequence[str]] = None
        コマンドライン引数, None の場合は sys.argv[1:]
    default: str = "configs/default/train.yaml"
        default の config のパス
    experiment: str = "configs/experiment/train.yaml"
        --experiment を指定しない場合の experiment の config のパス

    Returns
    ----------
    cfg: DictConfig
        学習に使う config
    diff: dict
        default から変わった項目 (上書きも含む)
    """
    # ノートブックの sys.argv はカーネルの引数なので読まない
    if argv is None and "ipykernel" in sys.modules:
        argv = []

    # experiment の config のパスと上書きを, 順番に関係なく受け取り
    parser = add_config_arguments(
        argparse.ArgumentParser(
            description="ML-Found のプロジェクトの学習", allow_abbrev=False
        ),
        experiment
    )
    args = parser.parse_intermixed_args(argv)
    cfg, diff = load_config_from_args(project_dir, args, default)

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
    ----------
    None
    """
    # 後から同じ条件で学習できるように config 全体を保存
    OmegaConf.save(cfg, Path(run_dir) / "config.yaml", resolve=True)

    # どこを変えた実験なのかが分かるように差分を保存
    OmegaConf.save(OmegaConf.create(diff), Path(run_dir) / "diff.yaml")
