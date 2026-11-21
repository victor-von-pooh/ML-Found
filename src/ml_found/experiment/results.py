import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
from typing import Any, Optional, Union
import uuid
import wave

from matplotlib import image as mpimg
import numpy as np
import torch

from ml_found.training.status import now_iso

# データ全体の評価と見本を置くディレクトリの名前
EVALUATIONS_DIR = "evaluations"
SAMPLES_DIR = "samples"

# 見本の一覧のファイルの名前
SAMPLES_INDEX = "index.json"

# 見本の一覧 (samples/index.json) の形式の版 (無い一覧は版 1).
# 版 2 の値の決まりの shift は ML-EDA と同じ向き
# (正規化した座標 = 元の座標 + shift). 版 1 は逆の向き
SAMPLES_FORMAT_VERSION = 2

# 学習の後の見本 (エポックを渡さない場合) のファイルの名前 (拡張子の前)
FINAL_STEM = "final"

# ML-Found の形より前にプロジェクトが samples/ の直下に置いた画像
# (samples/epoch_010.png のような番号付きの画像. ML-Dashboard の画像の推移と
# 同じ見分け方)
LEGACY_IMAGE = re.compile(r"^(.*?)(\d+)\.(png|jpe?g|gif|webp)$", re.IGNORECASE)
LEGACY_NAME = "samples"

# 見本の一覧の読み書きを 1 つずつにする鍵 (同じプロセスのスレッドの間)
SAMPLES_LOCK = threading.Lock()

# 評価と見本の名前に使える文字 (ファイルの名前になるため.
# ML-Quest の指標の名前と同じ決まり)
# 先頭の . は一時ファイルと見分けるために禁止
NAME_PATTERN = re.compile(r"[A-Za-z0-9_@-][A-Za-z0-9_.@-]*")

# 評価の種類
EVALUATION_KINDS = ("table", "curve")

# 指標の向き (小さいほど良いか, 大きいほど良いか)
BETTER_VALUES = ("lower", "higher")

# 比べる相手との表の形で, 同じ指標の向きを持つ列の名前の終わり
# (<指標>, <指標>_baseline, <指標>_diff)
METRIC_SUFFIXES = ("_baseline", "_diff")

# 見本の種類と, 写すファイルの拡張子から決める種類
SAMPLE_KINDS = ("image", "text", "audio", "point_cloud", "mask", "file")
KIND_BY_SUFFIX = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".webp": "image", ".bmp": "image", ".txt": "text", ".md": "text",
    ".wav": "audio", ".mp3": "audio", ".flac": "audio", ".ogg": "audio",
    ".ply": "point_cloud"
}

# 配列から書く見本の拡張子
SUFFIX_BY_KIND = {
    "image": ".png", "text": ".txt", "audio": ".wav", "point_cloud": ".ply",
    "mask": ".png"
}

# 領域の番号の範囲 (色の表の大きさ)
MASK_IDS = 256

# 画像のチャンネルの数として扱う大きさ
CHANNELS = (1, 3, 4)

# 音のチャンネルの数の上限 (サンプル数との取り違えの確認)
MAX_CHANNELS = 8


def new_temp(path: Path) -> Path:
    """
    書き込みの一時ファイルを, 普通のファイルと同じ権限で作る関数

    Parameters
    ----------
    path: Path
        最後に置き換えるファイル

    Returns
    ----------
    temp: Path
        同じディレクトリの空の一時ファイル (.<名前>.<乱数>.tmp).
        umask に従う権限 (config.yaml と同じ) で, 別のユーザーの画面も読める
    """
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temp, "x", encoding="utf-8"):
        pass

    return temp


def write_json_file(path: Path, data: Any) -> None:
    """
    読み手が書きかけを読まないように JSON を書く関数

    Parameters
    ----------
    path: Path
        書くファイル
    data: Any
        書く内容

    Returns
    ----------
    None
    """
    # 一時ファイルに書いてから置き換え, 失敗した場合は一時ファイルの削除
    temp = new_temp(path)
    try:
        text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def check_optional_text(value: Any, where: str) -> Optional[str]:
    """
    値が文字か None であることを確かめる関数

    Parameters
    ----------
    value: Any
        確かめる値
    where: str
        誤りのメッセージに出す引数の名前

    Returns
    ----------
    checked: Optional[str]
        確かめた値
    """
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{where} は文字か None にしてください: {value!r}")
    checked = value

    return checked


def run_directory(target: Any) -> Path:
    """
    実験か実験のディレクトリのパスから, 実験のディレクトリを取得する関数

    Parameters
    ----------
    target: Any
        start_experiment が返した実験か, 実験のディレクトリのパス

    Returns
    ----------
    run_dir: Path
        実験のディレクトリ
    """
    # Experiment の run_dir の取り出し
    run_dir = Path(getattr(target, "run_dir", target))
    if not run_dir.is_dir():
        raise FileNotFoundError(f"実験のディレクトリがありません: {run_dir}")

    return run_dir


def check_name(name: Any, where: str) -> str:
    """
    評価と見本の名前がファイルの名前に使えることを確かめる関数

    Parameters
    ----------
    name: Any
        名前
    where: str
        誤りのメッセージに出す名前の種類

    Returns
    ----------
    checked: str
        確かめた名前
    """
    if (
        not isinstance(name, str) or name in (".", "..")
        or not NAME_PATTERN.fullmatch(name)
    ):
        raise ValueError(
            f"{where}の名前は英数字と _ . @ - だけ (先頭は . 以外) に"
            f"してください: {name!r}"
        )
    checked = name

    return checked


def to_plain(value: Any, where: str) -> Any:
    """
    表の値を JSON に書ける値 (数, 文字, 真偽値, None) にする関数

    Parameters
    ----------
    value: Any
        表の値 (0 次元のテンソルと NumPy の数も可)
    where: str
        誤りのメッセージに出す場所

    Returns
    ----------
    plain: Any
        JSON に書ける値, nan と無限大は None
    """
    # テンソルと NumPy の数は Python の数に変換
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{where} の値は 1 つの数にしてください")
        value = value.item()
    elif isinstance(value, (np.ndarray, np.generic)):
        if np.size(value) != 1:
            raise ValueError(f"{where} の値は 1 つの数にしてください")
        value = value.item()
    if value is None or isinstance(value, (bool, str)):
        plain = value
        return plain
    if isinstance(value, (int, float)):
        plain = None if math.isinf(value) or math.isnan(value) else value
        return plain
    raise ValueError(
        f"{where} の値は数, 文字, 真偽値, None にしてください: {value!r}"
    )


def is_column(value: Any) -> bool:
    """
    値が表の列 (並び) かを判定する関数

    Parameters
    ----------
    value: Any
        値

    Returns
    ----------
    result: bool
        リスト, タプル, 1 次元以上のテンソルか配列なら True
    """
    result = isinstance(value, (list, tuple)) or (
        isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim > 0
    )

    return result


def table_records(table: Any) -> tuple[list[str], list[dict]]:
    """
    いろいろな形の表を, 列の名前と行の辞書の並びにする関数

    Parameters
    ----------
    table: Any
        行の辞書の並び, 列の名前から値の並びの辞書, 1 行の辞書,
        pandas の DataFrame (索引は使わない) のどれか

    Returns
    ----------
    columns: list[str]
        列の名前 (最初に現れた順)
    records: list[dict]
        1 行が 1 つの辞書
    """
    # DataFrame は行の辞書の並びに変換
    if hasattr(table, "to_dict") and hasattr(table, "columns"):
        records = table.to_dict(orient="records")
    elif isinstance(table, dict) and not table:
        records = []
    elif isinstance(table, dict) and all(
        is_column(value) for value in table.values()
    ):
        # 列の名前から値の並びの辞書は, 長さをそろえて行に変換
        lengths = {len(value) for value in table.values()}
        if len(lengths) != 1:
            raise ValueError("表の列の長さがそろっていません")
        size = lengths.pop()
        records = [
            {key: value[index] for key, value in table.items()}
            for index in range(size)
        ]
    elif isinstance(table, dict):
        records = [table]
    elif isinstance(table, (list, tuple)) and all(
        isinstance(row, dict) for row in table
    ):
        records = list(table)
    else:
        raise ValueError(
            "表は行の辞書の並び, 列の名前から値の並びの辞書, 1 行の辞書, "
            "DataFrame のどれかにしてください"
        )

    # 列の名前の順は最初に現れた順, 値は JSON に書ける値
    columns: list[str] = []
    for row in records:
        for key in row:
            if not isinstance(key, str):
                raise ValueError(f"表の列の名前は文字にしてください: {key!r}")
            if key not in columns:
                columns.append(key)
    records = [
        {key: to_plain(row.get(key), f"列 {key}") for key in columns}
        for row in records
    ]

    return columns, records


def metric_family(column: str, columns: list[str]) -> list[str]:
    """
    比べる相手との表の形で, 列と同じ指標の列を集める関数

    Parameters
    ----------
    column: str
        列の名前
    columns: list[str]
        表の列

    Returns
    ----------
    family: list[str]
        <指標>, <指標>_baseline, <指標>_diff のうち表にある列 (列の順).
        列自体も含む
    """
    # 終わりを除いた名前を指標とし, 同じ指標の 3 つの列の候補を作成
    metric = column
    for suffix in METRIC_SUFFIXES:
        if column.endswith(suffix) and len(column) > len(suffix):
            metric = column[:-len(suffix)]
            break
    candidates = {metric} | {metric + suffix for suffix in METRIC_SUFFIXES}
    family = [
        item for item in columns if item in candidates or item == column
    ]

    return family


def spread_better(
    better: dict[str, str], columns: list[str], strict: bool
) -> dict[str, str]:
    """
    指標の向きを, 同じ指標の列 (<指標>, _baseline, _diff) に広げる関数

    Parameters
    ----------
    better: dict[str, str]
        列の名前から向き (lower か higher). 列は表にあるもの
    columns: list[str]
        表の列
    strict: bool
        同じ指標の列に違う向きを書いた場合に誤りとするか. False の場合は
        書いた向きを残し, 書いていない列だけを埋める

    Returns
    ----------
    spread: dict[str, str]
        広げた後の向き (列の順)
    """
    # 書いた向きの確認と, 書いていない同じ指標の列の補い
    filled = dict(better)
    for column, value in better.items():
        for item in metric_family(column, columns):
            current = filled.get(item)
            if current is None:
                filled[item] = value
            elif current != value and strict:
                raise ValueError(
                    f"better の {column} と {item} は同じ指標なので, "
                    f"同じ向きにしてください: {value}, {current}"
                )
    spread = {column: filled[column] for column in columns if column in filled}

    return spread


def check_better(better: Any, columns: list[str]) -> dict[str, str]:
    """
    save_evaluation に渡した列ごとの指標の向きを確かめる関数

    Parameters
    ----------
    better: Any
        列の名前から lower か higher の辞書, None の場合は向きを書かない
    columns: list[str]
        表の列

    Returns
    ----------
    checked: dict[str, str]
        同じ指標の列に広げた向き. None の場合は空
    """
    # 辞書で, 列は表にあり, 値は lower か higher に制限
    if better is None:
        checked: dict[str, str] = {}
        return checked
    if not isinstance(better, dict):
        raise ValueError(
            "better は列の名前から lower か higher の辞書にしてください: "
            f"{better!r}"
        )
    for column, value in better.items():
        if column not in columns:
            raise ValueError(f"better の列 {column!r} は表にありません")
        if value not in BETTER_VALUES:
            raise ValueError(
                f"better の {column} は lower か higher にしてください: "
                f"{value!r}"
            )
    checked = spread_better(dict(better), columns, strict=True)

    return checked


def read_better(data: dict) -> dict[str, str]:
    """
    読み込んだ評価から列ごとの指標の向きを取り出す関数

    Parameters
    ----------
    data: dict
        evaluations/<名前>.json の中身

    Returns
    ----------
    better: dict[str, str]
        表の列のうち向きのあるもの (同じ指標の列に広げたもの).
        better を持たないファイルは空, 表に無い列と知らない値の欄は除く
    """
    # 列が文字のリストでない場合と better が辞書でない場合は向きが無い扱い
    columns = data.get("columns")
    raw = data.get("better")
    if (
        not isinstance(columns, list)
        or not all(isinstance(column, str) for column in columns)
        or not isinstance(raw, dict)
    ):
        better: dict[str, str] = {}
        return better
    known = {
        column: value for column, value in raw.items()
        if column in columns and value in BETTER_VALUES
    }
    better = spread_better(known, columns, strict=False)

    return better


def save_evaluation(
    target: Any, name: str, table: Any, split: str,
    description: Optional[str] = None, kind: str = "table",
    x: Optional[str] = None, baseline: Optional[str] = None,
    guard: bool = False, epoch: Optional[int] = None,
    better: Optional[dict[str, str]] = None
) -> Path:
    """
    学習の後のデータ全体の評価を evaluations/<名前>.json に書く関数

    Parameters
    ----------
    target: Any
        start_experiment が返した実験か, 実験のディレクトリのパス
    name: str
        評価の名前 (英数字と _ . @ - だけ. 先頭は . 以外)
    table: Any
        評価の表. 行の辞書の並び, 列の名前から値の並びの辞書, 1 行の辞書,
        pandas の DataFrame のどれか
    split: str
        評価に使った分割 (test, valid など)
    description: Optional[str] = None
        何の評価か
    kind: str = "table"
        table (表) か curve (x の列を横軸にした曲線)
    x: Optional[str] = None
        曲線の横軸の列 (curve だけ)
    baseline: Optional[str] = None
        比べる相手の名前 (今の処理, 前の実験など. 差の列を持つ場合)
    guard: bool = False
        守る指標の表か
    epoch: Optional[int] = None
        評価した重みのエポック
    better: Optional[dict[str, str]] = None
        列ごとの指標の向き. 列の名前から lower (小さいほど良い) か
        higher (大きいほど良い). <指標>, <指標>_baseline, <指標>_diff は
        同じ向きで, どれか 1 つに書けば他にも広げて書く

    Returns
    ----------
    path: Path
        書いたファイル
    """
    # 名前, 分割, 種類, 横軸の確認
    run_dir = run_directory(target)
    check_name(name, "評価")
    if not isinstance(split, str) or split == "":
        raise ValueError(f"split は空でない文字にしてください: {split!r}")
    if kind not in EVALUATION_KINDS:
        raise ValueError(f"kind は table か curve にしてください: {kind!r}")
    columns, records = table_records(table)
    if kind == "curve" and x not in columns:
        raise ValueError(f"曲線の横軸 x は表の列にしてください: {x!r}")
    if kind == "table" and x is not None:
        raise ValueError("x は kind が curve の場合だけ書けます")
    if epoch is not None and (
        not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0
    ):
        raise ValueError(f"epoch は 0 以上の整数にしてください: {epoch!r}")
    check_optional_text(description, "description")
    check_optional_text(baseline, "baseline")
    directions = check_better(better, columns)

    # 一時ファイルから置き換えて書き込み
    evaluation = {
        "name": name,
        "split": split,
        "kind": kind,
        "x": x,
        "columns": columns,
        "rows": records,
        "description": description,
        "baseline": baseline,
        "guard": bool(guard),
        "better": directions,
        "epoch": epoch,
        "created_at": now_iso()
    }
    folder = run_dir / EVALUATIONS_DIR
    folder.mkdir(exist_ok=True)
    path = folder / f"{name}.json"
    write_json_file(path, evaluation)

    return path


def load_evaluations(run_dir: Union[str, os.PathLike]) -> list[dict]:
    """
    実験のディレクトリの評価を全て読む関数

    Parameters
    ----------
    run_dir: Union[str, os.PathLike]
        実験のディレクトリ

    Returns
    ----------
    evaluations: list[dict]
        名前の順の評価. evaluations/ が無い実験 (評価を書かなかった実験など)
        は空. better は列ごとの指標の向きの辞書 (無いファイルは空)
    """
    # 一時ファイル (. で始まる) と読めないファイルは無視
    folder = Path(run_dir) / EVALUATIONS_DIR
    evaluations: list[dict] = []
    if not folder.is_dir():
        return evaluations
    for path in sorted(folder.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            data["better"] = read_better(data)
            evaluations.append(data)

    return evaluations


def to_array(value: Any) -> np.ndarray:
    """
    テンソルか配列を cpu の NumPy の配列にする関数

    Parameters
    ----------
    value: Any
        テンソル, 配列, 並び

    Returns
    ----------
    array: np.ndarray
        配列
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.numpy()
    array = np.asarray(value)

    return array


def image_array(value: Any) -> np.ndarray:
    """
    画像の見本を (高さ, 幅) か (高さ, 幅, チャンネル) の 8 bit の配列にする関数

    Parameters
    ----------
    value: Any
        画像. (高さ, 幅), (高さ, 幅, チャンネル), (チャンネル, 高さ, 幅).
        小数は 0 から 1, 整数は 0 から 255 の値

    Returns
    ----------
    pixels: np.ndarray
        uint8 の配列
    """
    # チャンネルが先頭 (PyTorch の形) なら後ろへ移す
    array = to_array(value)
    if array.ndim == 3 and (
        array.shape[0] in CHANNELS and array.shape[-1] not in CHANNELS
    ):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim not in (2, 3) or (
        array.ndim == 3 and array.shape[-1] not in CHANNELS
    ):
        raise ValueError(
            "画像は (高さ, 幅), (高さ, 幅, チャンネル), (チャンネル, 高さ, 幅) "
            f"の形にしてください (チャンネルは 1, 3, 4): {array.shape}"
        )

    # 小数は 0 から 1, 整数は 0 から 255 として 8 bit に変換
    # nan と無限大は黙って黒や白にせず誤り
    if np.issubdtype(array.dtype, np.floating) or array.dtype == bool:
        if not np.isfinite(array).all():
            raise ValueError("画像に nan か無限大の値があります")
        scaled = np.clip(array.astype(np.float64), 0.0, 1.0) * 255.0
        pixels = np.rint(scaled).astype(np.uint8)
    elif np.issubdtype(array.dtype, np.integer):
        pixels = np.clip(array, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"画像の値の型が使えません: {array.dtype}")

    return pixels


def write_image(path: Path, value: Any) -> None:
    """
    画像の見本を PNG に書く関数

    Parameters
    ----------
    path: Path
        書くファイル
    value: Any
        画像の配列

    Returns
    ----------
    None
    """
    pixels = image_array(value)
    if pixels.ndim == 2:
        mpimg.imsave(path, pixels, cmap="gray", vmin=0, vmax=255, format="png")
    else:
        mpimg.imsave(path, pixels, format="png")


def write_audio(path: Path, value: Any, sample_rate: Optional[int]) -> None:
    """
    音の見本を 16 bit の PCM の WAV に書く関数

    Parameters
    ----------
    path: Path
        書くファイル
    value: Any
        音. (サンプル数,) か (チャンネル, サンプル数) で, -1 から 1 の値
    sample_rate: Optional[int]
        1 秒のサンプル数

    Returns
    ----------
    None
    """
    # 形とサンプリングの周波数の確認
    if (
        not isinstance(sample_rate, int) or isinstance(sample_rate, bool)
        or sample_rate < 1
    ):
        raise ValueError(
            f"音の見本には 1 以上の整数の sample_rate を渡してください: "
            f"{sample_rate!r}"
        )
    # 整数の音は幅 (16 bit, 32 bit など) で -1 から 1 の値が変わるので誤り
    array = to_array(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(
            "音は -1 から 1 の小数の配列にしてください (整数の音は, 型の最大値"
            f"で割って小数にします): {array.dtype}"
        )
    if not np.isfinite(array).all():
        raise ValueError("音に nan か無限大の値があります")
    array = array.astype(np.float64)
    if array.ndim == 1:
        array = array[None]
    if array.ndim != 2:
        raise ValueError(
            "音は (サンプル数,) か (チャンネル, サンプル数) の形にしてください: "
            f"{array.shape}"
        )

    # (サンプル数, チャンネル) の取り違えの確認 (チャンネルは 8 まで)
    if array.shape[0] > MAX_CHANNELS or array.shape[0] > array.shape[1]:
        raise ValueError(
            "音は (チャンネル, サンプル数) の形にしてください. (サンプル数, "
            f"チャンネル) なら .T で入れ替えます: {array.shape}"
        )

    # -1 から 1 を 16 bit の整数にし, チャンネルを交互に並べて書き込み
    samples = np.rint(np.clip(array, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as file:
        file.setnchannels(samples.shape[0])
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes(samples.T.tobytes())


def write_point_cloud(path: Path, value: Any) -> None:
    """
    点群の見本を ascii の PLY に書く関数

    Parameters
    ----------
    path: Path
        書くファイル
    value: Any
        点群. (点の数, 3) の座標

    Returns
    ----------
    None
    """
    array = to_array(value).astype(np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            f"点群は (点の数, 3) の形にしてください: {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("点群に nan か無限大の座標があります")
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {array.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    rows = "".join(
        f"{row[0]:.7g} {row[1]:.7g} {row[2]:.7g}\n" for row in array
    )
    path.write_text(header + rows, encoding="ascii")


def mask_color(class_id: int) -> tuple[int, int, int]:
    """
    領域の番号の色を, 決まった色の表 (PASCAL VOC と同じ) から選ぶ関数

    Parameters
    ----------
    class_id: int
        番号 (0 から 255)

    Returns
    ----------
    color: tuple[int, int, int]
        赤, 緑, 青 (0 から 255). 番号 0 は黒
    """
    # 番号の下位のビットから 3 つずつ取り, 色の上位のビットに順に配置
    red = 0
    green = 0
    blue = 0
    value = class_id
    for shift in range(7, -1, -1):
        red |= (value & 1) << shift
        green |= ((value >> 1) & 1) << shift
        blue |= ((value >> 2) & 1) << shift
        value >>= 3
    color = (red, green, blue)

    return color


def mask_array(value: Any) -> np.ndarray:
    """
    領域の見本を (高さ, 幅) の番号の配列にする関数

    Parameters
    ----------
    value: Any
        領域の番号. (高さ, 幅) か (1, 高さ, 幅) の整数か真偽値の配列

    Returns
    ----------
    mask: np.ndarray
        int64 の配列 (0 から 255)
    """
    # 先頭の大きさ 1 の次元 (チャンネル) の除去と形の確認
    array = to_array(value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(
            "領域の見本は (高さ, 幅) の番号の配列にしてください: "
            f"{array.shape}"
        )

    # 小数は番号の取り違え (確率など) のことが多いので誤り
    if not (
        np.issubdtype(array.dtype, np.integer) or array.dtype == bool
    ):
        raise ValueError(
            "領域の見本は整数か真偽値の配列にしてください (確率なら argmax で"
            f"番号にします): {array.dtype}"
        )
    if array.size == 0:
        raise ValueError(f"領域の見本が空です: {array.shape}")
    mask = array.astype(np.int64)
    if (mask.min() < 0 or mask.max() >= MASK_IDS):
        raise ValueError(
            f"領域の番号は 0 から {MASK_IDS - 1} にしてください: "
            f"{int(mask.min())} から {int(mask.max())}"
        )

    return mask


def class_names(classes: Any) -> dict[int, Optional[str]]:
    """
    渡された番号の名前を, 番号から名前の辞書にする関数

    Parameters
    ----------
    classes: Any
        名前のリスト (番号の順) か, 番号から名前の辞書か None

    Returns
    ----------
    names: dict[int, Optional[str]]
        番号から名前 (名前は文字か None)
    """
    # リストは番号の順, 辞書はそのまま
    if classes is None:
        names: dict[int, Optional[str]] = {}
        return names
    if isinstance(classes, dict):
        pairs = list(classes.items())
    elif isinstance(classes, (list, tuple)):
        pairs = list(enumerate(classes))
    else:
        raise ValueError(
            "classes は名前のリストか, 番号から名前の辞書にしてください: "
            f"{classes!r}"
        )

    # 番号は 0 から 255 の整数, 名前は文字か None
    names = {}
    for class_id, name in pairs:
        if (
            isinstance(class_id, bool)
            or not isinstance(class_id, (int, np.integer))
            or not 0 <= class_id < MASK_IDS
        ):
            raise ValueError(
                f"classes の番号は 0 から {MASK_IDS - 1} の整数にして"
                f"ください: {class_id!r}"
            )
        check_optional_text(name, f"classes の番号 {class_id} の名前")
        names[int(class_id)] = name

    return names


def class_table(mask: np.ndarray, classes: Any) -> list[dict]:
    """
    領域の見本の番号の表 (番号, 名前, 色) を作る関数

    Parameters
    ----------
    mask: np.ndarray
        番号の配列
    classes: Any
        名前のリストか, 番号から名前の辞書か None

    Returns
    ----------
    table: list[dict]
        番号の順の id, name, color (#rrggbb). 名前を渡した番号と,
        配列に現れた番号の全て
    """
    names = class_names(classes)
    present = {int(value) for value in np.unique(mask)}
    table = []
    for class_id in sorted(present | set(names)):
        red, green, blue = mask_color(class_id)
        table.append({
            "id": class_id,
            "name": names.get(class_id),
            "color": f"#{red:02x}{green:02x}{blue:02x}"
        })

    return table


def mask_pixels(mask: np.ndarray) -> np.ndarray:
    """
    番号の配列を, 決まった色の表で色分けした画像にする関数

    Parameters
    ----------
    mask: np.ndarray
        番号の配列 (0 から 255)

    Returns
    ----------
    pixels: np.ndarray
        (高さ, 幅, 3) の uint8 の画像
    """
    palette = np.array(
        [mask_color(class_id) for class_id in range(MASK_IDS)],
        dtype=np.uint8
    )
    pixels = palette[mask]

    return pixels


def check_convention(convention: Any) -> Optional[dict]:
    """
    値の決まりが JSON にできる辞書か None であることを確かめる関数

    Parameters
    ----------
    convention: Any
        値の決まり (点群の座標の単位, 正規化のずらしなど)

    Returns
    ----------
    checked: Optional[dict]
        JSON を通した写し (配列やテンソルは渡せない)
    """
    if convention is None:
        return None
    if not isinstance(convention, dict):
        raise ValueError(
            f"convention は辞書か None にしてください: {convention!r}"
        )
    try:
        text = json.dumps(convention, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "convention は JSON にできる値 (数, 文字, 真偽値, None, リスト, "
            f"辞書. nan と無限大は不可) にしてください: {error}"
        ) from error
    checked = json.loads(text)

    return checked


def sample_kind(value: Any, kind: Optional[str]) -> str:
    """
    見本の種類を決める関数

    Parameters
    ----------
    value: Any
        見本 (文字, ファイルのパス, 配列)
    kind: Optional[str]
        渡した種類, None の場合は値から決める

    Returns
    ----------
    decided: str
        image, text, audio, point_cloud, mask, file のどれか
    """
    # 渡した種類はそのまま (選択肢の確認)
    if kind is not None:
        if kind not in SAMPLE_KINDS:
            raise ValueError(
                f"kind は {', '.join(SAMPLE_KINDS)} のどれかにしてください: "
                f"{kind!r}"
            )
        decided = kind
        return decided

    # 文字は文, パスは拡張子から. 配列は形から決められないので誤り
    if isinstance(value, str):
        decided = "text"
        return decided
    if isinstance(value, os.PathLike):
        decided = KIND_BY_SUFFIX.get(Path(value).suffix.lower(), "file")
        return decided
    raise ValueError(
        "配列の見本には kind (image, audio, point_cloud, mask) を渡して"
        "ください"
    )


def sample_stem(epoch: Optional[int]) -> str:
    """
    見本のファイルの名前 (拡張子の前) を決める関数

    Parameters
    ----------
    epoch: Optional[int]
        エポック, None は学習の後

    Returns
    ----------
    stem: str
        epoch_0005 のような名前か final
    """
    if epoch is None:
        stem = FINAL_STEM
        return stem
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError(f"epoch は 0 以上の整数にしてください: {epoch!r}")
    stem = f"epoch_{epoch:04d}"

    return stem


def sample_folder(root: Path, name: str) -> Path:
    """
    見本の名前ごとのディレクトリを, 他のファイルとぶつからないか確かめて
    作る関数

    Parameters
    ----------
    root: Path
        samples/ のディレクトリ
    name: str
        見本の名前

    Returns
    ----------
    folder: Path
        samples/<名前>/
    """
    # 一覧のファイルの名前と, 前からある同じ名前のファイル (直下の画像など)
    # とのぶつかりは誤り
    folder = root / name
    if name == SAMPLES_INDEX:
        raise ValueError(f"見本の名前に {SAMPLES_INDEX} は使えません")
    if root.exists() and not root.is_dir():
        raise ValueError(f"{root} がディレクトリではありません")
    if folder.exists() and not folder.is_dir():
        raise ValueError(
            f"見本の名前 {name} は samples/ の直下のファイルとぶつかります. "
            "別の名前にしてください"
        )
    folder.mkdir(parents=True, exist_ok=True)

    return folder


def write_sample_file(
    temp: Path, value: Any, kind: str, sample_rate: Optional[int]
) -> None:
    """
    見本を種類に合わせた形でファイルに書く関数

    Parameters
    ----------
    temp: Path
        書く一時ファイル
    value: Any
        見本 (文字, ファイルのパス, 配列)
    kind: str
        見本の種類
    sample_rate: Optional[int]
        音の 1 秒のサンプル数

    Returns
    ----------
    None
    """
    # ファイルは形を変えない写し (中身だけを写し, 権限は一時ファイルのもの)
    if isinstance(value, os.PathLike):
        source = Path(value)
        if not source.is_file():
            raise FileNotFoundError(f"見本のファイルがありません: {source}")
        shutil.copyfile(source, temp)
    elif kind == "text":
        if not isinstance(value, str):
            raise ValueError("文の見本は文字にしてください")
        temp.write_text(value, encoding="utf-8")
    elif kind == "image":
        write_image(temp, value)
    elif kind == "audio":
        write_audio(temp, value, sample_rate)
    elif kind == "point_cloud":
        write_point_cloud(temp, value)
    else:
        raise ValueError(
            "kind が file の見本は, ファイルのパス (pathlib.Path) を渡してください"
        )


def read_samples_index(path: Path) -> tuple[list[dict], int]:
    """
    見本の一覧と形式の版を読む関数

    Parameters
    ----------
    path: Path
        samples/index.json

    Returns
    ----------
    result: tuple[list[dict], int]
        見本の項目の並び (ファイルが無ければ空) と形式の版
        (ファイルが無ければ今の版, 版の無い一覧は 1)
    """
    if not path.is_file():
        result = ([], SAMPLES_FORMAT_VERSION)
        return result
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples") if isinstance(data, dict) else None
    if not isinstance(samples, list):
        raise ValueError(f"見本の一覧の形が正しくありません: {path}")
    version = data.get("format_version", 1)
    if (
        not isinstance(version, int) or isinstance(version, bool)
        or version < 1
    ):
        raise ValueError(
            f"{path} の format_version が 1 以上の整数ではありません: "
            f"{version!r}"
        )
    result = (samples, version)

    return result


def negate_shift(value: Any) -> Any:
    """
    shift の数 (入れ子のリストの中を含む) の符号を反転する関数

    Parameters
    ----------
    value: Any
        版 1 の shift (数か, 数のリスト)

    Returns
    ----------
    negated: Any
        符号を反転した値. 数でない値 (文字, 真偽値, None) はそのまま
    """
    if isinstance(value, list):
        negated = [negate_shift(item) for item in value]
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        negated = -value
    else:
        negated = value

    return negated


def upgrade_sample(item: Any, version: int) -> Any:
    """
    古い版の一覧の項目の値の決まりを, 今の版の意味に写す関数

    Parameters
    ----------
    item: Any
        一覧の項目
    version: int
        一覧の形式の版

    Returns
    ----------
    upgraded: Any
        版 1 の shift (元の座標 = 値 × scale + shift) の符号を反転し,
        今の版 (正規化した座標 = 元の座標 + shift) にした項目の写し.
        今の版の項目と, 辞書でない項目はそのまま
    """
    convention = item.get("convention") if isinstance(item, dict) else None
    if (
        version >= SAMPLES_FORMAT_VERSION
        or not isinstance(convention, dict) or "shift" not in convention
    ):
        upgraded = item
        return upgraded
    upgraded = dict(item)
    upgraded["convention"] = dict(
        convention, shift=negate_shift(convention["shift"])
    )

    return upgraded


def original_coordinates(
    points: Any, convention: Optional[dict]
) -> np.ndarray:
    """
    点群の見本の座標を, 値の決まりから正規化の前の座標に戻す関数

    Parameters
    ----------
    points: Any
        (点の数, 3) の座標 (見本の PLY の頂点など)
    convention: Optional[dict]
        load_samples が返した項目の値の決まり (今の版の意味).
        scale (無ければ 1) と shift (無ければ 0) を使う.
        newer_version が True の項目は None なので, 呼ぶ前に確かめる

    Returns
    ----------
    original: np.ndarray
        元の座標 (値 × scale - shift)
    """
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            f"点群は (点の数, 3) の配列にしてください: {array.shape}"
        )
    if convention is not None and not isinstance(convention, dict):
        raise ValueError(
            f"convention は辞書か None にしてください: {convention!r}"
        )
    given = convention or {}
    scale_value = given.get("scale")
    shift_value = given.get("shift")
    try:
        scale = np.asarray(
            1.0 if scale_value is None else scale_value, dtype=float
        )
        shift = np.asarray(
            0.0 if shift_value is None else shift_value, dtype=float
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "convention の scale と shift は数か 3 つの数のリストに"
            f"してください: {given!r}"
        ) from error
    if scale.shape not in ((), (3,)) or shift.shape not in ((), (3,)):
        raise ValueError(
            "convention の scale と shift は数か 3 つの数のリストに"
            f"してください: {given!r}"
        )
    original = array * scale - shift

    return original


def save_sample(
    target: Any, name: str, value: Any, epoch: Optional[int] = None,
    kind: Optional[str] = None, split: Optional[str] = None,
    caption: Optional[str] = None, sample_rate: Optional[int] = None,
    classes: Any = None, convention: Optional[dict] = None
) -> Path:
    """
    見本の出力を samples/<名前>/ に書き, 一覧に加える関数

    Parameters
    ----------
    target: Any
        start_experiment が返した実験か, 実験のディレクトリのパス
    name: str
        見本の名前 (英数字と _ . @ - だけ. 先頭は . 以外, index.json は不可)
    value: Any
        見本. 文字 (文), pathlib.Path (ファイルを写す), 配列かテンソル
        (画像, 音, 点群. kind が要る)
    epoch: Optional[int] = None
        エポック, None は学習の後
    kind: Optional[str] = None
        image, text, audio, point_cloud, mask, file のどれか.
        None の場合は文字なら text, パスなら拡張子から決める
    split: Optional[str] = None
        見本の入力の分割
    caption: Optional[str] = None
        添える説明
    sample_rate: Optional[int] = None
        音の 1 秒のサンプル数 (配列の音だけ)
    classes: Any = None
        mask の番号の名前. 名前のリスト (番号の順) か, 番号から名前の辞書
        (mask だけ)
    convention: Optional[dict] = None
        値の決まり (点群の座標の単位, 正規化のずらしなど. JSON にできる辞書)

    Returns
    ----------
    path: Path
        書いたファイル (samples/<名前>/epoch_0005.png など)
    """
    # 何も書く前に, 名前, 種類, エポック, 添える文字の確認
    run_dir = run_directory(target)
    check_name(name, "見本")
    decided = sample_kind(value, kind)
    stem = sample_stem(epoch)
    check_optional_text(split, "split")
    check_optional_text(caption, "caption")
    checked_convention = check_convention(convention)
    if isinstance(value, os.PathLike):
        suffix = Path(value).suffix.lower()
    else:
        suffix = SUFFIX_BY_KIND.get(decided, "")

    # 領域の番号は色分けした画像に, 番号の表は一覧に記録
    table = None
    content = value
    written = decided
    if decided == "mask":
        if isinstance(value, os.PathLike):
            raise ValueError(
                "kind が mask の見本は, 番号の配列を渡してください "
                "(色分けは ML-Found が行う)"
            )
        mask = mask_array(value)
        table = class_table(mask, classes)
        content = mask_pixels(mask)
        written = "image"
    elif classes is not None:
        raise ValueError("classes は kind が mask の場合だけ渡せます")

    # 一覧の読み書きは同じプロセスのスレッドの間で 1 つずつ
    root = run_dir / SAMPLES_DIR
    index_path = root / SAMPLES_INDEX
    with SAMPLES_LOCK:
        folder = sample_folder(root, name)
        path = folder / f"{stem}{suffix}"
        samples, version = read_samples_index(index_path)
        if version > SAMPLES_FORMAT_VERSION:
            raise ValueError(
                f"{index_path} の版 {version} は, この ML-Found が知っている"
                f"版 {SAMPLES_FORMAT_VERSION} より新しいです. ML-Found を"
                "新しくしてください"
            )

        # 一時ファイルに書いてから置き換え (書きかけを読ませないため)
        temp = new_temp(path)
        try:
            write_sample_file(temp, content, written, sample_rate)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

        # 同じエポックと名前の前の項目を置き換えた一覧の書き込み
        # (古い版の一覧の項目は, 今の版の意味に写した上での書き直し)
        kept = []
        removed = []
        for item in samples:
            if item.get("epoch") == epoch and item.get("name") == name:
                removed.append(root / str(item.get("file", "")))
                continue
            kept.append(upgrade_sample(item, version))
        kept.append({
            "epoch": epoch,
            "name": name,
            "kind": decided,
            "file": f"{name}/{path.name}",
            "split": split,
            "caption": caption,
            "classes": table,
            "convention": checked_convention,
            "created_at": now_iso()
        })
        write_json_file(index_path, {
            "format_version": SAMPLES_FORMAT_VERSION, "samples": kept
        })

        # 一覧を書けた後に, 拡張子の違う前の見本のファイルを削除
        for old in removed:
            if old != path and old.parent == folder:
                old.unlink(missing_ok=True)

    return path


def legacy_samples(root: Path) -> list[dict]:
    """
    samples/ の直下に置かれた番号付きの画像を, 見本の項目にする関数

    Parameters
    ----------
    root: Path
        samples/ のディレクトリ

    Returns
    ----------
    samples: list[dict]
        名前とエポックの順の項目. 名前は番号の前の部分 (空か epoch なら
        samples), 番号をエポックとし, legacy を True にする
    """
    # ML-Found の形より前のプロジェクトの画像 (samples/epoch_010.png など)
    samples: list[dict] = []
    if not root.is_dir():
        return samples
    for path in root.iterdir():
        match = LEGACY_IMAGE.match(path.name)
        if not path.is_file() or match is None or path.name.startswith("."):
            continue
        prefix = match.group(1).rstrip("_.-")
        samples.append({
            "epoch": int(match.group(2)),
            "name": LEGACY_NAME if prefix in ("", "epoch") else prefix,
            "kind": "image",
            "file": path.name,
            "split": None,
            "caption": None,
            "classes": None,
            "convention": None,
            "created_at": None,
            "legacy": True,
            "newer_version": False
        })
    samples.sort(key=legacy_order)

    return samples


def legacy_order(item: dict) -> tuple[str, int]:
    """
    直下の画像の項目を並べる順を返す関数

    Parameters
    ----------
    item: dict
        見本の項目

    Returns
    ----------
    order: tuple[str, int]
        名前とエポック
    """
    order = (item["name"], item["epoch"])

    return order


def load_samples(run_dir: Union[str, os.PathLike]) -> list[dict]:
    """
    実験のディレクトリの見本の一覧を読む関数

    Parameters
    ----------
    run_dir: Union[str, os.PathLike]
        実験のディレクトリ

    Returns
    ----------
    samples: list[dict]
        一覧の項目 (書いた順) と, samples/ の直下の番号付きの画像
        (ML-Found の形より前のプロジェクトのもの. legacy が True).
        samples/ が無い実験は空. 一覧が読めない場合は直下の画像だけ.
        convention は今の版の意味 (古い版の shift は符号を反転).
        新しい版の一覧の項目は convention を None にし,
        newer_version を True にする
    """
    root = Path(run_dir) / SAMPLES_DIR
    try:
        samples, version = read_samples_index(root / SAMPLES_INDEX)
    except (OSError, ValueError):
        samples, version = [], SAMPLES_FORMAT_VERSION
    newer = version > SAMPLES_FORMAT_VERSION
    samples = [upgrade_sample(item, version) for item in samples]

    # classes と convention の無い項目には null を補完.
    # 新しい版の値の決まりは, 意味が変わったかもしれないので読まない扱い
    for item in samples:
        if isinstance(item, dict):
            item.setdefault("classes", None)
            item.setdefault("convention", None)
            item["newer_version"] = newer
            if newer:
                item["convention"] = None
    samples = samples + legacy_samples(root)

    return samples
