import os
from pathlib import Path
import stat
from typing import Optional

# ファイルの読み込みの 1 回の大きさ
READ_CHUNK_BYTES = 1 << 20

# ファイルを, リンクを辿らず (O_NOFOLLOW) 待たずに (O_NONBLOCK) 開く旗
PLAIN_FILE_FLAGS = (
    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
)

# ディレクトリを, リンクを辿らずにディレクトリとして開く旗
PLAIN_DIR_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)

# Windows のジャンクションの印 (lstat の st_reparse_tag)
JUNCTION_TAG = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)


def can_use_dir_fd() -> bool:
    """
    ディレクトリを開いた番号から, リンクを辿らずに調べて開けるかを判定する
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    usable: bool
        os.open と os.stat が dir_fd を, os.listdir が開いた番号を使え,
        O_NOFOLLOW がある場合は True (Windows などは False)
    """
    usable = (
        hasattr(os, "O_NOFOLLOW") and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd and os.listdir in os.supports_fd
    )

    return usable


def is_plain_dir(info: os.stat_result) -> bool:
    """
    lstat の結果が, リンクでもジャンクションでもないディレクトリかを判定する
    関数

    Parameters
    ----------
    info: os.stat_result
        リンクを辿らずに調べた結果

    Returns
    ----------
    plain: bool
        リンクでもジャンクションでもないディレクトリなら True
    """
    plain = stat.S_ISDIR(info.st_mode) and (
        getattr(info, "st_reparse_tag", 0) != JUNCTION_TAG
    )

    return plain


def open_plain_dir(folder: Path) -> tuple[bool, Optional[int]]:
    """
    ディレクトリを, リンクを辿らずに確かめて開く関数

    Parameters
    ----------
    folder: Path
        ディレクトリの場所

    Returns
    ----------
    opened: tuple[bool, Optional[int]]
        読めるか (無い, ディレクトリでない, リンクの場合は False) と,
        開いたディレクトリの番号 (dir_fd を使えない環境と読めない場合は None.
        閉じるのは呼んだ側)
    """
    # 無い場合, ディレクトリでない場合, リンクの場合は読まない
    try:
        info = os.lstat(folder)
    except (OSError, ValueError):
        opened = (False, None)
        return opened
    if not is_plain_dir(info):
        opened = (False, None)
        return opened
    if not can_use_dir_fd():
        opened = (True, None)
        return opened

    # 確かめた後にリンクへ入れ替えられても辿らないように開き, 同じものかを確認
    try:
        dir_fd = os.open(folder, PLAIN_DIR_FLAGS)
    except (OSError, ValueError):
        opened = (False, None)
        return opened
    try:
        same = os.path.samestat(info, os.fstat(dir_fd))
    except OSError:
        same = False
    if not same:
        os.close(dir_fd)
        opened = (False, None)
        return opened
    opened = (True, dir_fd)

    return opened


def read_limited(fd: int, limit: int) -> Optional[bytes]:
    """
    開いたファイルを, 上限の 1 バイト先までだけ読む関数

    Parameters
    ----------
    fd: int
        開いたファイルの番号 (閉じるのは呼んだ側)
    limit: int
        読む大きさの上限 (バイト)

    Returns
    ----------
    content: Optional[bytes]
        ファイルの中身, 上限を超えた場合は None
    """
    chunks = []
    size = 0
    while size <= limit:
        chunk = os.read(fd, min(READ_CHUNK_BYTES, limit + 1 - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
    if size > limit:
        return None
    content = b"".join(chunks)

    return content


def read_checked_file(
    folder: Path, name: str, limit: int, dir_fd: Optional[int]
) -> tuple[Optional[bytes], bool]:
    """
    ファイルを, 調べたものと開いたものが同じ普通のファイルの場合だけ 1 回
    読む関数

    Parameters
    ----------
    folder: Path
        ファイルを置くディレクトリ
    name: str
        ファイルの名前
    limit: int
        読む大きさの上限 (バイト)
    dir_fd: Optional[int]
        folder を開いた番号, 無ければ場所から開く

    Returns
    ----------
    result: tuple[Optional[bytes], bool]
        ファイルの中身 (読まなかった場合は None) と, 調べた後に別の
        ファイルへ置き換わったか (読み直せば読めることがある)
    """
    # 開く前に, リンク, 普通でないファイル (FIFO は開くと止まる),
    # 大きすぎるファイルを除外
    try:
        if dir_fd is not None:
            info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        else:
            info = os.lstat(folder / name)
    except (OSError, ValueError):
        result = (None, False)
        return result
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        result = (None, False)
        return result

    # 確かめた後に入れ替えられても辿らず止まらないように開き,
    # 開いたものが確かめたものと同じ普通のファイルかを確認
    try:
        if dir_fd is not None:
            fd = os.open(name, PLAIN_FILE_FLAGS, dir_fd=dir_fd)
        else:
            fd = os.open(folder / name, PLAIN_FILE_FLAGS)
    except (OSError, ValueError):
        result = (None, False)
        return result
    try:
        opened = os.fstat(fd)
        if not os.path.samestat(info, opened):
            result = (None, True)
            return result
        if not stat.S_ISREG(opened.st_mode):
            result = (None, False)
            return result
        result = (read_limited(fd, limit), False)
    except OSError:
        result = (None, False)
    finally:
        os.close(fd)

    return result


def read_plain_file(
    folder: Path, name: str, limit: int, dir_fd: Optional[int] = None
) -> Optional[bytes]:
    """
    ファイルを, リンクでない普通のファイルで上限の大きさまでの場合だけ読む
    関数

    調べた後に置き換わった場合 (書き手が一時ファイルから置き換えた瞬間)
    は, 1 回だけ読み直す

    Parameters
    ----------
    folder: Path
        ファイルを置くディレクトリ
    name: str
        ファイルの名前
    limit: int
        読む大きさの上限 (バイト)
    dir_fd: Optional[int] = None
        folder を開いた番号, 無ければ場所から開く

    Returns
    ----------
    content: Optional[bytes]
        ファイルの中身. 無いファイル, リンク, 普通でないファイル (FIFO など),
        上限を超えるファイル, 読めないファイルは None
    """
    content, changed = read_checked_file(folder, name, limit, dir_fd)
    if changed:
        content, _ = read_checked_file(folder, name, limit, dir_fd)

    return content
