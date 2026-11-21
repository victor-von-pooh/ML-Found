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

# 写さなかった理由ごとの, train.log に書く説明
SKIP_REASONS = {
    "missing": "ファイルがありません",
    "outside": "プロジェクトの外を指しています",
    "link": "場所か途中のディレクトリがシンボリックリンクです",
    "not_file": "普通のファイルではありません",
    "unreadable": "読めません",
    "resumed": "再開した学習なので写しません (定義は再開元の実験にあります)"
}


def read_spec_value(cfg: DictConfig) -> Optional[str]:
    """
    config の model.spec に書いたモデルの定義の場所を読む関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    value: Optional[str]
        書いた場所, 無い, 空, 文字でない, 解決できない場合は None
    """
    # model はプロジェクトの項目なので, 読めない値は書いていない扱い
    try:
        value = OmegaConf.select(
            cfg, MODEL_SPEC_KEY, throw_on_resolution_failure=True
        )
    except OmegaConfBaseException:
        return None
    if not isinstance(value, str) or not value.strip():
        return None

    return value


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
        リンクかジャンクションなら True. 権限などで確かめられない場合は
        False (開くときに読めないものとして扱う)
    """
    # os.path の関数は確かめられない場合も例外を出さずに False を返却
    isjunction = getattr(os.path, "isjunction", None)
    linked = os.path.islink(path) or (
        isjunction is not None and isjunction(path)
    )

    return linked


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
        読めなかった理由 (missing, link, not_file, unreadable), 読めた場合は
        None
    """
    # 確かめた後に入れ替えられても辿らないように O_NOFOLLOW で開き,
    # FIFO で止まらないように O_NONBLOCK で開いてから種類を確認
    flags = (
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    )
    try:
        fd = os.open(path, flags)
    except (FileNotFoundError, NotADirectoryError):
        return None, "missing"
    except OSError:
        if is_link(path):
            return None, "link"
        if os.path.exists(path) and not os.path.isfile(path):
            return None, "not_file"
        return None, "unreadable"

    # 確かめた後に途中のディレクトリをリンクに入れ替えられた場合に備え,
    # 開いたものがプロジェクトの中の同じファイルかを確かめてから読み込み
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            return None, "not_file"
        real = Path(os.path.realpath(path))
        if not real.is_relative_to(root) or not os.path.samestat(
            opened, os.stat(real)
        ):
            return None, "link"
        chunks = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
    except OSError:
        return None, "unreadable"
    finally:
        os.close(fd)
    content = b"".join(chunks)

    return content, None


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


def copy_model_spec(
    cfg: DictConfig, run_dir: Path, project_dir: Path
) -> Optional[dict]:
    """
    config の model.spec のモデルの定義を実験のディレクトリに写す関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config
    run_dir: Path
        実験のディレクトリ
    project_dir: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    record: Optional[dict]
        source.yaml の model_spec (path, status, sha256).
        config に model.spec が無い, 空, 文字でない場合は None
    """
    # 書いていなければ何もしない
    value = read_spec_value(cfg)
    if value is None:
        return None

    # 相対パスはプロジェクトから辿り, リンクを辿らずに正規化した場所を記録
    root = project_dir.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = rebase_on_project(Path(os.path.normpath(path)), root)
    record: dict[str, Any] = {
        "path": relative_to_project(path, root), "status": None,
        "sha256": None
    }

    # 再開した学習, プロジェクトの外, 途中のリンクは写さない
    if cfg.trainer.get("resume_from") is not None:
        record["status"] = "resumed"
        return record
    if not path.is_relative_to(root):
        record["status"] = "outside"
        return record
    if has_link(path, root):
        record["status"] = "link"
        return record

    # 中身を読んで写し, 中身の sha256 を記録
    content, reason = read_spec_bytes(path, root)
    if content is None:
        record["status"] = reason
        return record
    write_spec_copy(run_dir, content)
    record["status"] = "copied"
    record["sha256"] = hashlib.sha256(content).hexdigest()

    return record


def describe_model_spec(record: dict) -> str:
    """
    モデルの定義を写した結果を train.log に書く文にする関数

    Parameters
    ----------
    record: dict
        copy_model_spec が返した記録

    Returns
    ----------
    message: str
        写した場合は写した先と sha256, 写さなかった場合は理由
    """
    # 写した場合は sha256, 写さなかった場合は理由の説明
    if record["status"] == "copied":
        message = (
            f"モデルの定義 ({record['path']}) を {MODEL_SPEC_FILE} に"
            f"写しました (sha256: {record['sha256']})"
        )
        return message
    message = (
        f"モデルの定義 ({record['path']}) を写しませんでした: "
        f"{SKIP_REASONS[record['status']]}"
    )

    return message
