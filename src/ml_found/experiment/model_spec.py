from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ml_found.config.source import relative_to_project
from ml_found.experiment.results import new_temp

# モデルの定義の場所を書く config の項目
MODEL_SPEC_KEY = "model.spec"

# 実験に写したモデルの定義のファイルの名前
MODEL_SPEC_FILE = "model_spec.yaml"

# 写す定義の大きさの上限 (モデルの定義の YAML には十分な大きさ.
# 重みのファイルなどを誤って指した場合に丸ごと読まないため)
MAX_SPEC_BYTES = 16 * 1024 * 1024

# 読み込みの 1 回の大きさ
CHUNK_BYTES = 1 << 20

# ディレクトリを表す場所の終わり (m.yaml/ と m.yaml/.)
TRAILING_SEPARATORS = ("/", "/.", os.sep, os.sep + ".")

# Windows のジャンクションの印 (lstat の st_reparse_tag)
JUNCTION_TAG = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)

# 写さなかった理由ごとの, train.log に書く説明
SKIP_REASONS = {
    "missing": "ファイルがありません",
    "outside": "プロジェクトの外を指しています",
    "link": "場所か途中のディレクトリがシンボリックリンクです",
    "not_file": "普通のファイルではありません",
    "unreadable": "読めません",
    "too_large": f"{MAX_SPEC_BYTES // (1024 * 1024)} MiB を超えています",
    "invalid": "場所として読めない書き方です",
    "write_failed": "実験のディレクトリに書けませんでした"
}


@dataclass
class ModelSpecRead:
    # source.yaml の model_spec (path, status, sha256). model.spec が無い,
    # 空, 文字でない場合は None
    record: Optional[dict]

    # 写す中身 (読めた場合だけ)
    content: Optional[bytes]

    # model.spec が文字でないか参照を解決できない場合の, train.log に書く
    # 知らせ
    notice: Optional[str]


def is_link(path: Path) -> bool:
    """
    場所がシンボリックリンクか Windows のジャンクションかを判定する関数

    Parameters
    ----------
    path: Path
        場所

    Returns
    ----------
    linked: bool
        リンクかジャンクションなら True. 無い場合と, 権限などで確かめられない
        場合は False (開くときに無いか読めないものとして扱う)
    """
    # 辿らずに調べ, Windows ではジャンクションの印も確認
    try:
        info = os.lstat(path)
    except (OSError, ValueError):
        linked = False
        return linked
    linked = stat.S_ISLNK(info.st_mode) or (
        getattr(info, "st_reparse_tag", 0) == JUNCTION_TAG
    )

    return linked


def check_climbs(path: Path) -> Optional[str]:
    """
    正規化する前の場所の .. が, OS の辿り方と同じ場所に上がるかを確かめる
    関数

    Parameters
    ----------
    path: Path
        正規化する前の絶対パス

    Returns
    ----------
    reason: Optional[str]
        .. の直前がリンクなら link (POSIX では, 正規化で .. を消すと OS が
        開く場所と違う場所を記録するため), 無いかディレクトリでなければ
        missing (OS では開けないため). 問題が無ければ None
    """
    # 先頭から 1 つずつ辿り, .. の直前の場所を確認
    current = Path(path.anchor)
    for part in path.parts[1:]:
        if part != "..":
            current = current / part
            continue
        if is_link(current):
            reason: Optional[str] = "link"
            return reason
        if not os.path.isdir(current):
            reason = "missing"
            return reason
        current = current.parent
    reason = None

    return reason


def has_link(path: Path, project_dir: Path) -> bool:
    """
    プロジェクトから場所までの途中か場所自体にリンクがあるかを確かめる関数

    Parameters
    ----------
    path: Path
        プロジェクトの中の場所 (正規化した絶対パス)
    project_dir: Path
        プロジェクトのディレクトリ (解決した絶対パス)

    Returns
    ----------
    linked: bool
        途中のディレクトリか場所自体がシンボリックリンクなら True
    """
    # プロジェクトの直下から順に 1 つずつ確認
    current = project_dir
    for part in path.relative_to(project_dir).parts:
        current = current / part
        if is_link(current):
            linked = True
            return linked
    linked = False

    return linked


def rebase_on_project(path: Path, root: Path) -> Path:
    """
    プロジェクトの別の書き方から辿った絶対パスを, 解決したプロジェクトの
    場所からのパスに直す関数

    Parameters
    ----------
    path: Path
        正規化した絶対パス
    root: Path
        プロジェクトのディレクトリ (解決した絶対パス)

    Returns
    ----------
    rebased: Path
        プロジェクトの中なら root からのパス. macOS の /tmp と /private/tmp
        のように, 途中までがプロジェクトそのものを指す場合も中とする.
        それ以外は元のパス
    """
    # 既に root の中ならそのまま, 違えば先頭の側から root と同じ場所を探索
    if path.is_relative_to(root):
        rebased = path
        return rebased
    rebased = path
    for parent in reversed(path.parents):
        if os.path.realpath(parent) == str(root):
            rebased = root / path.relative_to(parent)
            break

    return rebased


def open_spec(path: Path) -> tuple[Optional[int], Optional[str]]:
    """
    モデルの定義のファイルを, 最後の要素のリンクを辿らずに開く関数

    Parameters
    ----------
    path: Path
        定義のファイルの場所

    Returns
    ----------
    fd: Optional[int]
        開いたファイルの番号, 開けなかった場合は None
    reason: Optional[str]
        開けなかった理由 (missing, invalid, link, not_file, unreadable),
        開けた場合は None
    """
    # 確かめた後に入れ替えられても辿らないように O_NOFOLLOW で開き,
    # FIFO で止まらないように O_NONBLOCK で開く
    flags = (
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    )
    fd: Optional[int] = None
    reason: Optional[str] = None
    try:
        fd = os.open(path, flags)
    except (FileNotFoundError, NotADirectoryError):
        reason = "missing"
    except ValueError:
        reason = "invalid"
    except OSError:
        if is_link(path):
            reason = "link"
        elif os.path.exists(path) and not os.path.isfile(path):
            reason = "not_file"
        else:
            reason = "unreadable"

    return fd, reason


def read_open_spec(
    fd: int, path: Path, root: Path
) -> tuple[Optional[bytes], Optional[str]]:
    """
    開いた定義のファイルを, 普通のファイルで上限の大きさまでの場合だけ
    読む関数

    Parameters
    ----------
    fd: int
        開いたファイルの番号 (閉じるのは呼んだ側)
    path: Path
        定義のファイルの場所
    root: Path
        プロジェクトのディレクトリ (解決した絶対パス)

    Returns
    ----------
    content: Optional[bytes]
        ファイルの中身, 読まなかった場合は None
    reason: Optional[str]
        読まなかった理由 (not_file, link, too_large), 読めた場合は None
    """
    # 確かめた後に途中のディレクトリをリンクに入れ替えられた場合に備え,
    # 開いたものがプロジェクトの中の同じ普通のファイルかを確認
    content: Optional[bytes] = None
    opened = os.fstat(fd)
    if not stat.S_ISREG(opened.st_mode):
        reason: Optional[str] = "not_file"
        return content, reason
    real = Path(os.path.realpath(path))
    if not real.is_relative_to(root) or not os.path.samestat(
        opened, os.stat(real)
    ):
        reason = "link"
        return content, reason

    # 上限の 1 バイト先までだけ読み, 超えたら写さない
    chunks = []
    size = 0
    while size <= MAX_SPEC_BYTES:
        chunk = os.read(fd, min(CHUNK_BYTES, MAX_SPEC_BYTES + 1 - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
    if size > MAX_SPEC_BYTES:
        reason = "too_large"
        return content, reason
    content = b"".join(chunks)
    reason = None

    return content, reason


def read_spec_bytes(
    path: Path, root: Path
) -> tuple[Optional[bytes], Optional[str]]:
    """
    モデルの定義のファイルを, リンクを辿らずに普通のファイルだけ読む関数

    Parameters
    ----------
    path: Path
        定義のファイルの場所
    root: Path
        プロジェクトのディレクトリ (解決した絶対パス)

    Returns
    ----------
    content: Optional[bytes]
        ファイルの中身, 読めなかった場合は None
    reason: Optional[str]
        読めなかった理由 (missing, invalid, link, not_file, unreadable,
        too_large), 読めた場合は None
    """
    # 開いて読み, 読み込みの途中の誤りは読めないものとして扱う
    fd, reason = open_spec(path)
    content: Optional[bytes] = None
    if fd is None:
        return content, reason
    try:
        content, reason = read_open_spec(fd, path, root)
    except (OSError, ValueError):
        content, reason = None, "unreadable"
    finally:
        os.close(fd)

    return content, reason


def locate_spec(value: str, root: Path) -> tuple[Path, Path, str]:
    """
    config に書いた場所を, 確かめに使う絶対パスと記録する文字にする関数

    Parameters
    ----------
    value: str
        model.spec に書いた場所
    root: Path
        プロジェクトのディレクトリ (解決した絶対パス)

    Returns
    ----------
    raw: Path
        正規化する前の絶対パス
    path: Path
        リンクを辿らずに正規化した絶対パス
    written: str
        記録する場所 (プロジェクトの中なら相対パス, 外なら絶対パス)
    """
    # 相対パスはプロジェクトから辿り (~ はホームに展開), 誤った書き方
    # (存在しないユーザーの ~, ヌル文字など) は呼んだ側で扱う
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raw = root / raw
    path = rebase_on_project(Path(os.path.normpath(raw)), root)
    written = relative_to_project(path, root)

    return raw, path, written


def check_spec_path(path: Path, raw: Path, root: Path) -> Optional[str]:
    """
    定義の場所が写してよい場所かを確かめる関数

    Parameters
    ----------
    path: Path
        正規化した絶対パス
    raw: Path
        正規化する前の絶対パス
    root: Path
        プロジェクトのディレクトリ (解決した絶対パス)

    Returns
    ----------
    reason: Optional[str]
        写さない理由 (outside, link, missing), 写してよい場合は None
    """
    # プロジェクトの外, リンクの後の .. か辿れない .., 途中か場所自体の
    # リンクは写さない
    reason: Optional[str] = None
    if not path.is_relative_to(root):
        reason = "outside"
    else:
        reason = check_climbs(raw)
    if reason is None and has_link(path, root):
        reason = "link"

    return reason


def read_model_spec(cfg: DictConfig, project_dir: Path) -> ModelSpecRead:
    """
    config の model.spec のモデルの定義を, 写すために読む関数

    出力先を作る前に呼び, 学習を始めるときの定義の中身を持つ.
    写せない場合も例外を出さずに理由を記録に入れる

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config
    project_dir: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    spec: ModelSpecRead
        source.yaml の model_spec の記録と, 写す中身
    """
    # model はプロジェクトの項目なので, 解決できない参照と文字でない値は
    # 写さずに知らせるだけ (場所が無いので記録は null)
    try:
        value = OmegaConf.select(
            cfg, MODEL_SPEC_KEY, throw_on_resolution_failure=True
        )
    except OmegaConfBaseException as error:
        spec = ModelSpecRead(
            record=None, content=None,
            notice=f"{MODEL_SPEC_KEY} の参照を解決できないので, モデルの"
            f"定義を写しません: {str(error).splitlines()[0]!r}"
        )
        return spec
    if value is not None and not isinstance(value, str):
        spec = ModelSpecRead(
            record=None, content=None,
            notice=f"{MODEL_SPEC_KEY} が場所の文字でないので, モデルの定義を"
            f"写しません: {value!r}"
        )
        return spec
    if value is None or not value.strip():
        spec = ModelSpecRead(record=None, content=None, notice=None)
        return spec

    # 場所の書き方の確認 (読めない書き方は invalid)
    root = project_dir.resolve()
    record: dict[str, Any] = {"path": value, "status": None, "sha256": None}
    try:
        raw, path, record["path"] = locate_spec(value, root)
        reason = check_spec_path(path, raw, root)
    except (OSError, ValueError, RuntimeError):
        record["status"] = "invalid"
        spec = ModelSpecRead(record=record, content=None, notice=None)
        return spec

    # リンクの後の .. で link にした場所は, 別のファイルを指して見えない
    # ように書いた文字のまま記録. 末尾の区切り (m.yaml/ と m.yaml/.) は
    # ディレクトリの書き方
    if reason == "link" and check_climbs(raw) == "link":
        record["path"] = value
    if reason is None and value.endswith(TRAILING_SEPARATORS):
        reason = "not_file"

    # 写してよい場所なら中身を読み, 中身の sha256 を記録
    content: Optional[bytes] = None
    if reason is None:
        content, reason = read_spec_bytes(path, root)
    if content is not None:
        record["sha256"] = hashlib.sha256(content).hexdigest()
    record["status"] = reason
    spec = ModelSpecRead(record=record, content=content, notice=None)

    return spec


def write_spec_copy(run_dir: Path, content: bytes) -> None:
    """
    モデルの定義の中身を実験のディレクトリに書く関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ
    content: bytes
        定義のファイルの中身

    Returns
    ----------
    None
    """
    # umask に従う権限の一時ファイルに書いてから置き換え (元のファイルの
    # 権限は写さないので, 学習のスクリプトが後から同じ名前で上書きできる)
    path = run_dir / MODEL_SPEC_FILE
    temp = new_temp(path)
    try:
        temp.write_bytes(content)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def copy_model_spec(spec: ModelSpecRead, run_dir: Path) -> Optional[dict]:
    """
    読んだモデルの定義を実験のディレクトリに写す関数

    Parameters
    ----------
    spec: ModelSpecRead
        read_model_spec が返したもの
    run_dir: Path
        実験のディレクトリ

    Returns
    ----------
    record: Optional[dict]
        source.yaml の model_spec (path, status, sha256).
        config に model.spec が無い, 空, 文字でない, 参照を解決できない場合は
        None
    """
    # 読めた中身だけを写し, 書けなければ理由を残して学習は続行
    record: Optional[dict] = None
    if spec.record is None:
        return record
    record = dict(spec.record)
    if spec.content is None:
        return record
    try:
        write_spec_copy(run_dir, spec.content)
    except OSError:
        record.update(status="write_failed", sha256=None)
        return record
    record["status"] = "copied"

    return record


def describe_model_spec(
    spec: ModelSpecRead, record: Optional[dict]
) -> Optional[str]:
    """
    モデルの定義を写した結果を train.log に書く文にする関数

    Parameters
    ----------
    spec: ModelSpecRead
        read_model_spec が返したもの
    record: Optional[dict]
        copy_model_spec が返した記録

    Returns
    ----------
    message: Optional[str]
        写した場合は写した先と sha256, 写さなかった場合は理由.
        model.spec が無いか空の場合は None
    """
    # 書き方の誤りの知らせと, 改行で偽の行を作れないように repr の場所
    if spec.notice is not None:
        message: Optional[str] = spec.notice
        return message
    if record is None:
        message = None
        return message
    if record["status"] == "copied":
        message = (
            f"モデルの定義 ({record['path']!r}) を {MODEL_SPEC_FILE} に"
            f"写しました (sha256: {record['sha256']})"
        )
        return message
    message = (
        f"モデルの定義 ({record['path']!r}) を写しませんでした: "
        f"{SKIP_REASONS.get(record['status'], record['status'])}"
    )

    return message
