import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Optional, Union
import wave

from matplotlib import image as mpimg
import numpy as np
import torch

from ml_found.training.status import now_iso, write_json

# データ全体の評価と見本を置くディレクトリの名前
EVALUATIONS_DIR = "evaluations"
SAMPLES_DIR = "samples"

# 見本の一覧のファイルの名前
SAMPLES_INDEX = "index.json"

# 学習の後の見本 (エポックを渡さない場合) を置くディレクトリの名前
FINAL_DIR = "final"

# 評価と見本の名前に使える文字 (ファイルの名前になるため.
# ML-Quest の指標の名前と同じ決まり)
NAME_PATTERN = re.compile(r"[A-Za-z0-9_.@-]+")

# 評価の種類
EVALUATION_KINDS = ("table", "curve")

# 見本の種類と, 写すファイルの拡張子から決める種類
SAMPLE_KINDS = ("image", "text", "audio", "point_cloud", "file")
KIND_BY_SUFFIX = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".webp": "image", ".bmp": "image", ".txt": "text", ".md": "text",
    ".wav": "audio", ".mp3": "audio", ".flac": "audio", ".ogg": "audio",
    ".ply": "point_cloud"
}

# 配列から書く見本の拡張子
SUFFIX_BY_KIND = {
    "image": ".png", "text": ".txt", "audio": ".wav", "point_cloud": ".ply"
}

# 画像のチャンネルの数として扱う大きさ
CHANNELS = (1, 3, 4)


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
    # Experiment は run_dir を持つ
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
            f"{where}の名前は英数字と _ . @ - だけにしてください: {name!r}"
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
    if isinstance(value, (torch.Tensor, np.ndarray, np.generic)):
        if getattr(value, "ndim", 0) != 0 and np.size(value) != 1:
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
    elif isinstance(table, dict) and table and all(
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


def save_evaluation(
    target: Any, name: str, table: Any, split: str,
    description: Optional[str] = None, kind: str = "table",
    x: Optional[str] = None, baseline: Optional[str] = None,
    guard: bool = False, epoch: Optional[int] = None
) -> Path:
    """
    学習の後のデータ全体の評価を evaluations/<名前>.json に書く関数

    Parameters
    ----------
    target: Any
        start_experiment が返した実験か, 実験のディレクトリのパス
    name: str
        評価の名前 (英数字と _ . @ - だけ)
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
        not isinstance(epoch, int) or isinstance(epoch, bool)
    ):
        raise ValueError(f"epoch は整数にしてください: {epoch!r}")

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
        "epoch": epoch,
        "created_at": now_iso()
    }
    folder = run_dir / EVALUATIONS_DIR
    folder.mkdir(exist_ok=True)
    path = folder / f"{name}.json"
    write_json(path, evaluation)

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
        名前の順の評価. evaluations/ が無い実験 (足す前の ML-Found が作った
        実験など) は空
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
    if np.issubdtype(array.dtype, np.floating) or array.dtype == bool:
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
    array = to_array(value).astype(np.float64)
    if array.ndim == 1:
        array = array[None]
    if array.ndim != 2:
        raise ValueError(
            "音は (サンプル数,) か (チャンネル, サンプル数) の形にしてください: "
            f"{array.shape}"
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
        image, text, audio, point_cloud, file のどれか
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
        "配列の見本には kind (image, audio, point_cloud) を渡してください"
    )


def sample_folder(epoch: Optional[int]) -> str:
    """
    見本を置くディレクトリの名前を決める関数

    Parameters
    ----------
    epoch: Optional[int]
        エポック, None は学習の後

    Returns
    ----------
    folder: str
        epoch_0005 のような名前か final
    """
    if epoch is None:
        folder = FINAL_DIR
        return folder
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError(f"epoch は 0 以上の整数にしてください: {epoch!r}")
    folder = f"epoch_{epoch:04d}"

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
    # ファイルは形を変えずに写す
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


def read_samples_index(path: Path) -> list[dict]:
    """
    見本の一覧を読む関数

    Parameters
    ----------
    path: Path
        samples/index.json

    Returns
    ----------
    samples: list[dict]
        見本の項目の並び, ファイルが無ければ空
    """
    if not path.is_file():
        samples: list[dict] = []
        return samples
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples") if isinstance(data, dict) else None
    if not isinstance(samples, list):
        raise ValueError(f"見本の一覧の形が正しくありません: {path}")

    return samples


def save_sample(
    target: Any, name: str, value: Any, epoch: Optional[int] = None,
    kind: Optional[str] = None, split: Optional[str] = None,
    caption: Optional[str] = None, sample_rate: Optional[int] = None
) -> Path:
    """
    見本の出力を samples/ に書き, 一覧に加える関数

    Parameters
    ----------
    target: Any
        start_experiment が返した実験か, 実験のディレクトリのパス
    name: str
        見本の名前 (英数字と _ . @ - だけ. 同じエポックの中で 1 つ)
    value: Any
        見本. 文字 (文), pathlib.Path (ファイルを写す), 配列かテンソル
        (画像, 音, 点群. kind が要る)
    epoch: Optional[int] = None
        エポック, None は学習の後
    kind: Optional[str] = None
        image, text, audio, point_cloud, file のどれか.
        None の場合は文字なら text, パスなら拡張子から決める
    split: Optional[str] = None
        見本の入力の分割
    caption: Optional[str] = None
        添える説明
    sample_rate: Optional[int] = None
        音の 1 秒のサンプル数 (配列の音だけ)

    Returns
    ----------
    path: Path
        書いたファイル
    """
    # 名前, 種類, 置き場所の決定
    run_dir = run_directory(target)
    check_name(name, "見本")
    decided = sample_kind(value, kind)
    folder_name = sample_folder(epoch)
    if isinstance(value, os.PathLike):
        suffix = Path(value).suffix.lower()
    else:
        suffix = SUFFIX_BY_KIND[decided]
    root = run_dir / SAMPLES_DIR
    folder = root / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}{suffix}"

    # 同じエポックと名前の前の見本 (拡張子の違うものを含む) は置き換え
    index_path = root / SAMPLES_INDEX
    samples = read_samples_index(index_path)
    kept = []
    for item in samples:
        if item.get("epoch") == epoch and item.get("name") == name:
            old = root / str(item.get("file", ""))
            if old != path and old.parent == folder:
                old.unlink(missing_ok=True)
            continue
        kept.append(item)

    # 一時ファイルに書いてから置き換え (書きかけを読ませないため)
    handle, temp_name = tempfile.mkstemp(
        dir=folder, prefix=f".{name}.", suffix=".tmp"
    )
    os.close(handle)
    temp = Path(temp_name)
    try:
        write_sample_file(temp, value, decided, sample_rate)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

    # 一覧に加えて書き込み
    kept.append({
        "epoch": epoch,
        "name": name,
        "kind": decided,
        "file": f"{folder_name}/{path.name}",
        "split": split,
        "caption": caption,
        "created_at": now_iso()
    })
    write_json(index_path, {"samples": kept})

    return path


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
        書いた順の見本の項目. samples/ が無い実験 (足す前の ML-Found が
        作った実験など) と, 一覧が読めない場合は空
    """
    try:
        samples = read_samples_index(
            Path(run_dir) / SAMPLES_DIR / SAMPLES_INDEX
        )
    except (OSError, ValueError):
        samples = []

    return samples
