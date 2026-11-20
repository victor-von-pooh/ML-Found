import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union
import warnings

import yaml

from ml_found.config.source import relative_to_project
from ml_found.experiment.results import new_temp

if TYPE_CHECKING:
    from ml_found.experiment.setup import Experiment

# 何の実験かを書くファイルの名前
TASK_FILE = "task.yaml"

# task.yaml の形式の版 (項目の意味を変えるか消す場合に上げる. docs/FORMAT.md)
TASK_FORMAT_VERSION = 1

# 読める ML-EDA の dataset.yaml の版 (正解の出どころを持つ版. 版 3 は
# 版 2 に項目の中身の手順を足したもので, 受け持ちと正解は同じ形)
DATASET_FORMAT_VERSIONS = (2, 3)

# 受け持ち, 出力の形, 作り方の値 (ML-Quest の問題の設定の値を写したもの.
# action は強化学習の方策の出力の形として足した値)
# 値を足すのは版を変えない変更 (読む側は知らない値を分からないものとして読む)
ROLES = ("predict", "generate", "rank", "group", "effect")
OUTPUTS = (
    "class", "value", "distribution", "region", "shape", "text", "image",
    "table", "ranking", "cluster", "effect", "action"
)
METHODS = ("rule", "pretrained", "head", "partial", "finetune", "scratch")

# 元にした重みか実験があるはずの作り方
BASED_METHODS = ("head", "partial", "finetune")

# 正解の出どころの種類 (ML-EDA の dataset.yaml の版 2 の値を写したもの.
# environment は環境の報酬で測る方策のために足した値)
ORIGIN_TYPES = (
    "process", "annotation", "measurement", "synthetic", "environment",
    "unknown"
)

# 書ける項目 (知らない項目は typo として誤り)
TASK_KEYS = ("role", "output", "what", "target", "method", "base", "dataset")
TARGET_KEYS = ("kind", "origin")
ORIGIN_KEYS = ("type", "name", "settings")
BASE_KEYS = ("experiment", "weights")
DATASET_KEYS = ("path", "name", "format_version")

# 正解の出どころごとの, 指標が何との一致かの説明
ORIGIN_NOTES = {
    "process": (
        "正解は処理 {name} の出力. 指標はこの処理との一致を測り, "
        "モデルはこの処理を超えられない"
    ),
    "annotation": (
        "正解は人が付けたもの ({name}). 指標は付けた人との一致を測り, "
        "付けた人を超えたかは分からない"
    ),
    "measurement": "正解は測った値 ({name}). 指標は測りの誤差を含む",
    "synthetic": (
        "正解は作った正確な値 ({name}). 実際のデータでの性能は別に確かめる"
    ),
    "environment": "環境の報酬で測る. 実機での結果は別に確かめる",
    "unknown": "正解の出どころは分からない. 指標が何との一致かを確かめる"
}

# 新しい版の task.yaml の正解の出どころの説明
NEWER_NOTE = (
    "新しい版の ML-Found が書いた task.yaml なので, 正解の出どころは"
    "分からないものとして扱う. ML-Found を更新すると読める"
)


def check_keys(data: Any, allowed: tuple, where: str) -> dict:
    """
    辞書が決まった項目だけを持つことを確かめる関数

    Parameters
    ----------
    data: Any
        確かめる値, None は空の辞書として扱う
    allowed: tuple
        書ける項目の名前
    where: str
        誤りのメッセージに出す場所の名前

    Returns
    ----------
    checked: dict
        確かめた辞書
    """
    # None は書かなかったものとして空の辞書
    if data is None:
        checked: dict = {}
        return checked
    if not isinstance(data, dict):
        raise ValueError(f"{where} は辞書にしてください: {data!r}")
    unknown = [key for key in data if key not in allowed]
    if unknown:
        raise ValueError(
            f"{where} に知らない項目があります: {unknown} "
            f"(書ける項目: {', '.join(allowed)})"
        )
    checked = dict(data)

    return checked


def check_choice(
    value: Any, choices: tuple, where: str, allow_none: bool = True
) -> Optional[str]:
    """
    値が決まった選択肢のどれかであることを確かめる関数

    Parameters
    ----------
    value: Any
        確かめる値
    choices: tuple
        選択肢
    where: str
        誤りのメッセージに出す場所の名前
    allow_none: bool = True
        None (分からない) を許すか

    Returns
    ----------
    checked: Optional[str]
        確かめた値
    """
    if value is None and allow_none:
        return None
    if value not in choices:
        raise ValueError(
            f"{where} は {', '.join(choices)} のどれかにしてください: {value!r}"
        )
    checked = str(value)

    return checked


def check_text(value: Any, where: str) -> Optional[str]:
    """
    値が文字か None であることを確かめる関数

    Parameters
    ----------
    value: Any
        確かめる値
    where: str
        誤りのメッセージに出す場所の名前

    Returns
    ----------
    checked: Optional[str]
        確かめた値
    """
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{where} は空でない文字にしてください: {value!r}")
    checked = value

    return checked


def resolve_from(path: Union[str, os.PathLike], base_dir: Path) -> Path:
    """
    相対パスを決まったディレクトリから辿った絶対パスにする関数

    Parameters
    ----------
    path: Union[str, os.PathLike]
        パス
    base_dir: Path
        相対パスを辿るディレクトリ

    Returns
    ----------
    resolved: Path
        絶対パス
    """
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    resolved = resolved.resolve()

    return resolved


def read_dataset_yaml(path: Path) -> dict:
    """
    ML-EDA が書き出したデータセットの dataset.yaml を読む関数

    Parameters
    ----------
    path: Path
        データセットのディレクトリか dataset.yaml

    Returns
    ----------
    info: dict
        dataset.yaml の中身
    """
    # ディレクトリなら中の dataset.yaml
    file = path / "dataset.yaml" if path.is_dir() else path
    if not file.is_file():
        raise FileNotFoundError(f"dataset.yaml がありません: {file}")

    # 設定の値の ${...} を補間として扱わないように, YAML のまま読み込み
    try:
        info = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(
            f"dataset.yaml を読めません: {file}: {error}"
        ) from error
    if not isinstance(info, dict):
        raise ValueError(f"dataset.yaml が辞書ではありません: {file}")

    return info


def check_dataset_version(info: dict, file: Path) -> None:
    """
    dataset.yaml の版が読める版 (2 か 3) であることを確かめる関数

    Parameters
    ----------
    info: dict
        dataset.yaml の中身
    file: Path
        誤りのメッセージに出すファイル

    Returns
    ----------
    None
    """
    # 整数の版だけ (task.yaml の dataset.format_version に写すため). 版 1
    # は列の正解のデータセットで, 正解の出どころが無いので停止. 知らない版
    # も読み違えないように停止
    version = info.get("format_version")
    if type(version) is int and version in DATASET_FORMAT_VERSIONS:
        return
    if type(version) is int and version == 1:
        raise ValueError(
            f"{file} は版 1 (列の正解) の dataset.yaml なので, 受け持ちと"
            "正解の出どころを持ちません. task の role と target を直接"
            "書いてください"
        )
    raise ValueError(
        f"{file} の版 ({version!r}) は読めません. 読めるのは ML-EDA の"
        "dataset.yaml の版 "
        f"{' と '.join(str(v) for v in DATASET_FORMAT_VERSIONS)} だけです"
        " (新しい版なら ML-Found を更新してください)"
    )


def task_from_dataset(
    path: Union[str, os.PathLike], project_dir: Optional[str] = None
) -> dict:
    """
    ML-EDA の dataset.yaml (版 2 か 3) から task.yaml の中身を埋める関数

    Parameters
    ----------
    path: Union[str, os.PathLike]
        データセットのディレクトリか dataset.yaml
    project_dir: Optional[str] = None
        相対パスを辿るディレクトリ, None の場合は実行した場所

    Returns
    ----------
    task: dict
        role, target (kind, origin), dataset (path, name, format_version).
        write_task にそのまま渡せ, 他の項目を足してもよい
    """
    # ml_eda に頼らない YAML の直接の読み込みと, 版の確認
    base_dir = Path.cwd() if project_dir is None else Path(project_dir)
    resolved = resolve_from(path, base_dir)
    info = read_dataset_yaml(resolved)
    file = resolved / "dataset.yaml" if resolved.is_dir() else resolved
    check_dataset_version(info, file)

    # 受け持ち, 正解の種類, 正解の出どころの写し (出どころが無ければ unknown)
    # ML-EDA が版を上げずに足した欄は無視
    target = info.get("target") or {}
    if not isinstance(target, dict):
        raise ValueError(f"{file} の target が辞書ではありません")
    origin = target.get("origin") or {"type": "unknown"}
    if not isinstance(origin, dict):
        raise ValueError(f"{file} の target.origin が辞書ではありません")
    origin = {
        key: value for key, value in origin.items() if key in ORIGIN_KEYS
    }
    task = {
        "role": info.get("role"),
        "target": {
            "kind": target.get("kind"),
            "origin": copy.deepcopy(origin)
        },
        "dataset": {
            "path": file.parent.as_posix(),
            "name": info.get("name"),
            "format_version": info["format_version"]
        }
    }

    return task


def build_origin(origin: Any) -> dict:
    """
    正解の出どころを確かめて, 全ての欄を持つ形にする関数

    Parameters
    ----------
    origin: Any
        正解の出どころ (type, name, settings), None は分からないもの

    Returns
    ----------
    built: dict
        type, name, settings
    """
    # 種類を書かずに名前か設定だけを書くと unknown になるので警告
    data = check_keys(origin, ORIGIN_KEYS, "task.target.origin")
    if data and data.get("type") is None:
        warnings.warn(
            "task.target.origin に type が無いので, 正解の出どころは unknown "
            "(分からない) として書きます. process, annotation, measurement, "
            "synthetic, environment のどれかなら type を書いてください",
            stacklevel=4
        )
    built = {
        "type": check_choice(
            data.get("type") or "unknown", ORIGIN_TYPES,
            "task.target.origin.type", allow_none=False
        ),
        "name": check_text(data.get("name"), "task.target.origin.name"),
        "settings": copy.deepcopy(data.get("settings"))
    }

    return built


def build_base(base: Any, method: Optional[str], project_dir: Path) -> dict:
    """
    元にした実験と重みを確かめて, 全ての欄を持つ形にする関数

    Parameters
    ----------
    base: Any
        元にしたもの (experiment, weights), None は無いもの
    method: Optional[str]
        作り方
    project_dir: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    built: dict
        experiment (プロジェクトの中なら相対パス), weights
    """
    data = check_keys(base, BASE_KEYS, "task.base")
    experiment = check_text(data.get("experiment"), "task.base.experiment")
    weights = check_text(data.get("weights"), "task.base.weights")

    # 元にした実験はプロジェクトから辿り, 無ければ警告 (別のホストの実験など)
    if experiment is not None:
        resolved = resolve_from(experiment, project_dir)
        if not resolved.is_dir():
            warnings.warn(
                f"task.base.experiment のディレクトリがありません: {resolved}",
                stacklevel=3
            )
        experiment = relative_to_project(resolved, project_dir)

    # 一から学習するのに元がある, 元から作るのに元が無い場合の確認
    if method == "scratch" and (experiment or weights):
        raise ValueError(
            "task.method が scratch (一から学習) なのに task.base が"
            "書かれています"
        )
    if method in BASED_METHODS and not (experiment or weights):
        warnings.warn(
            f"task.method が {method} ですが, 元にした実験か重み "
            "(task.base.experiment か weights) が書かれていません",
            stacklevel=3
        )
    built = {"experiment": experiment, "weights": weights}

    return built


def build_dataset(dataset: Any, project_dir: Path) -> dict:
    """
    学習に使ったデータセットの情報を, 全ての欄を持つ形にする関数

    Parameters
    ----------
    dataset: Any
        task_from_dataset が返した辞書 (path, name, format_version) か None
    project_dir: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    built: dict
        path (プロジェクトの中なら相対パス), name, format_version
    """
    data = check_keys(dataset, DATASET_KEYS, "task.dataset")
    path = check_text(data.get("path"), "task.dataset.path")
    if path is not None:
        resolved = resolve_from(path, project_dir)
        path = relative_to_project(resolved, project_dir)
    version = data.get("format_version")
    if version is not None and (
        not isinstance(version, int) or isinstance(version, bool)
    ):
        raise ValueError(
            f"task.dataset.format_version は整数にしてください: {version!r}"
        )
    built = {
        "path": path,
        "name": check_text(data.get("name"), "task.dataset.name"),
        "format_version": version
    }

    return built


def merge_dataset(task: dict, project_dir: Path) -> dict:
    """
    task の dataset がパスなら dataset.yaml から埋め, 書いた項目を重ねる関数

    Parameters
    ----------
    task: dict
        書いた task
    project_dir: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    merged: dict
        dataset.yaml の値に書いた項目を重ねた task
    """
    # パスでなければそのまま
    dataset = task.get("dataset")
    if not isinstance(dataset, (str, os.PathLike)):
        merged = task
        return merged

    # 書いた項目を優先 (正解の出どころは欄ごとではなく丸ごと置き換え)
    # null は書かなかったものとして扱い, 埋めた値を保持
    merged = task_from_dataset(dataset, str(project_dir))
    for key, value in task.items():
        if key == "dataset" or value is None:
            continue
        if key == "target" and isinstance(value, dict):
            target = dict(merged["target"])
            for name, item in value.items():
                if item is not None:
                    target[name] = item
            merged["target"] = target
        else:
            merged[key] = value

    return merged


def build_task(task: Any, project_dir: Path) -> dict:
    """
    書いた task を確かめて, task.yaml の中身 (全ての項目を持つ形) にする関数

    Parameters
    ----------
    task: Any
        受け持ちなどの辞書. dataset にパスを書くと dataset.yaml から埋める
    project_dir: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    built: dict
        task.yaml の中身
    """
    # 知らない項目の確認と, データセットの説明からの補い
    data = check_keys(task, TASK_KEYS, "task")
    data = merge_dataset(data, project_dir)
    target = check_keys(data.get("target"), TARGET_KEYS, "task.target")
    method = check_choice(data.get("method"), METHODS, "task.method")

    # 全ての項目を書き, 分からない値は None
    built = {
        "format_version": TASK_FORMAT_VERSION,
        "role": check_choice(data.get("role"), ROLES, "task.role"),
        "output": check_choice(data.get("output"), OUTPUTS, "task.output"),
        "what": check_text(data.get("what"), "task.what"),
        "target": {
            "kind": check_text(target.get("kind"), "task.target.kind"),
            "origin": build_origin(target.get("origin"))
        },
        "method": method,
        "base": build_base(data.get("base"), method, project_dir),
        "dataset": build_dataset(data.get("dataset"), project_dir)
    }

    # YAML に書けない値 (配列など) を書く前に確認
    try:
        yaml.safe_dump(built)
    except Exception as error:
        raise ValueError(
            f"task に YAML に書けない値があります: {error}"
        ) from error

    return built


def save_task(run_dir: Path, task: dict) -> None:
    """
    task.yaml を書く関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ
    task: dict
        build_task で作った中身

    Returns
    ----------
    None
    """
    # 読み手が書きかけを読まないように, 一時ファイルから置き換え
    # 失敗した場合は一時ファイルの削除
    path = Path(run_dir) / TASK_FILE
    temp = new_temp(path)
    try:
        text = yaml.safe_dump(task, allow_unicode=True, sort_keys=False)
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def describe_origin(
    origin: Any, format_version: Optional[int] = None
) -> str:
    """
    正解の出どころから, 指標が何との一致かの説明を作る関数

    Parameters
    ----------
    origin: Any
        task.yaml の target.origin (type, name, settings)
    format_version: Optional[int] = None
        その task.yaml の版. 知っている版より新しい場合は言い切らない

    Returns
    ----------
    note: str
        人が読む説明. 出どころが分からない場合もその旨を返す
    """
    # 新しい版は欄の意味が変わったかもしれないので, 言い切らない知らせ
    if isinstance(format_version, int) and (
        format_version > TASK_FORMAT_VERSION
    ):
        note = NEWER_NOTE
        return note

    # 知らない種類と壊れた値は分からないものとして説明
    data = origin if isinstance(origin, dict) else {}
    kind = data.get("type")
    if kind not in ORIGIN_NOTES:
        kind = "unknown"
    name = data.get("name") or "名前の分からないもの"
    note = ORIGIN_NOTES[kind].format(name=name)

    return note


def write_task(
    experiment: "Experiment", task: Optional[dict] = None,
    dataset: Optional[Union[str, os.PathLike]] = None
) -> dict:
    """
    何の実験かを実験のディレクトリの task.yaml に書く関数

    Parameters
    ----------
    experiment: Experiment
        start_experiment が返した実験
    task: Optional[dict] = None
        受け持ち, 出力の意味, 正解, 作り方などの辞書 (docs/FORMAT.md の
        「task.yaml」の項目)
    dataset: Optional[Union[str, os.PathLike]] = None
        ML-EDA の dataset.yaml (版 2) かそのディレクトリ. 渡すと受け持ち,
        正解の種類と出どころを埋める (task に書いた項目が優先)

    Returns
    ----------
    built: dict
        書いた task.yaml の中身
    """
    # データセットのパスは task の dataset として扱う
    data = dict(task or {})
    if dataset is not None:
        data["dataset"] = dataset
    built = build_task(data, experiment.project_dir)

    # 書いて, 指標が何との一致かを train.log に記録
    save_task(experiment.run_dir, built)
    experiment.logger.info(
        f"task.yaml を書きました: {describe_origin(built['target']['origin'])}"
    )

    return built


def load_task(run_dir: Union[str, os.PathLike]) -> Optional[dict]:
    """
    実験のディレクトリの task.yaml を読む関数

    Parameters
    ----------
    run_dir: Union[str, os.PathLike]
        実験のディレクトリ

    Returns
    ----------
    task: Optional[dict]
        task.yaml の中身, 無いか読めない場合は None
        (task.yaml を書かなかった実験など).
        新しい版では target.origin を unknown に置き換え, newer_version を
        True にする
    """
    # 無い場合と壊れた場合は分からないものとして None
    path = Path(run_dir) / TASK_FILE
    if not path.is_file():
        return None
    try:
        task = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return None
    if not isinstance(task, dict):
        return None

    # 版は 1 以上の整数 (文字や小数を版 1 として読み違えないため)
    version = task.get("format_version")
    if (
        not isinstance(version, int) or isinstance(version, bool)
        or version < 1
    ):
        raise ValueError(
            f"{path} の format_version が 1 以上の整数ではありません: "
            f"{version!r}"
        )

    # 新しい版は意味が変わったかもしれないので, 正解の出どころ (「処理を
    # 超えられない」などの言い切りの元) を分からないものに置き換え
    if version > TASK_FORMAT_VERSION:
        warnings.warn(
            f"{path} は新しい版 ({version}) の ML-Found が書いたものです. "
            "正解の出どころは分からないものとして扱い, 他は読める項目だけを"
            "使います",
            stacklevel=2
        )
        target = task.get("target")
        task["target"] = dict(target) if isinstance(target, dict) else {}
        task["target"]["origin"] = {
            "type": "unknown", "name": None, "settings": None
        }
        task["newer_version"] = True

    return task
