"""
CLAUDE.md の一覧に無いリポジトリの名前を探すスクリプト

git の管理下のファイル (submodule の中は含まない) で, 一覧に無い名前を
見つけたら, 場所を出して 1 で終わる

探す語は, 大文字の ML で始まり - か _ が続く語と, 小文字の ml_ で始まる
語の 2 つの形だけ. 小文字の ml- は画面のクラス名 (ml-auto など) と取り
違えないため探さない
"""
import re
import subprocess
import sys
from pathlib import Path

# リポジトリの名前らしい語 (大文字小文字を区別する. 後ろに _ が続く環境
# 変数のような語は除く). 比べるときは名前の形をそろえる
REPO_NAME = re.compile(r"\bML[-_][A-Za-z]+\b(?!_)|\bml_[a-z]+\b(?!_)")

# git grep で候補の行を絞る形 (環境で \b が使えない場合があるので語の境は
# 見ない. 細かい判定は REPO_NAME で行う)
GREP_PATTERN = r"ML[-_][A-Za-z]|ml_[a-z]"


def name_key(name: str) -> str:
    """
    リポジトリの名前を, 大文字小文字と - と _ の違いを除いた形にする関数

    Parameters
    ----------
    name: str
        名前

    Returns
    ----------
    key: str
        小文字で, - と _ を除いた形
    """
    # 小文字にして区切りを除く
    key = name.lower().replace("-", "").replace("_", "")

    return key


def allowed_repos(root: Path) -> set[str]:
    """
    CLAUDE.md の触れてよいリポジトリの一覧を読む関数

    Parameters
    ----------
    root: Path
        リポジトリのルート

    Returns
    ----------
    names: set[str]
        一覧のリポジトリの名前 (name_key の形)

    Raises
    ----------
    ValueError
        CLAUDE.md に一覧 (「(今は ...)」) が無い場合
    """
    # 「(今は A, B, ...)」の括弧の中
    text = (root / "CLAUDE.md").read_text(encoding="utf-8")
    found = re.search(r"\(今は ([^)]*)\)", text)
    if found is None:
        raise ValueError("CLAUDE.md にリポジトリの一覧がありません")
    names = {name_key(name.strip()) for name in found.group(1).split(",")}

    return names


def grep_lines(root: Path) -> list[tuple[str, str, str]]:
    """
    管理下のファイルから名前らしい語を含む行を git grep で探す関数

    Parameters
    ----------
    root: Path
        リポジトリのルート

    Returns
    ----------
    found: list[tuple[str, str, str]]
        ファイル, 行の番号, 行の中身の組 (バイナリのファイルは除く)

    Raises
    ----------
    RuntimeError
        git grep が誤りで終わった場合
    """
    # 見つからない場合の終わりの値は 1, 誤りは 2 以上
    result = subprocess.run(
        ["git", "grep", "-I", "-n", "-z", "-E", GREP_PATTERN],
        cwd=root, capture_output=True, text=True, encoding="utf-8",
        errors="replace"
    )
    if result.returncode > 1:
        raise RuntimeError(f"git grep が失敗しました: {result.stderr}")

    # -z でファイル, 行の番号, 中身を NUL で区切った行
    found = []
    for line in result.stdout.splitlines():
        place, number, content = line.split("\0", 2)
        found.append((place, number, content))

    return found


def name_errors(root: Path, names: set[str]) -> list[str]:
    """
    一覧に無いリポジトリの名前を探す関数

    Parameters
    ----------
    root: Path
        リポジトリのルート
    names: set[str]
        触れてよいリポジトリの名前 (name_key の形)

    Returns
    ----------
    errors: list[str]
        見つけた場所と語
    """
    # 行の中の語ごと
    errors = []
    for place, number, content in grep_lines(root):
        for word in REPO_NAME.findall(content):
            if name_key(word) not in names:
                errors.append(f"{place}:{number}: 一覧に無いリポジトリ {word}")

    return errors


def main() -> int:
    """
    一覧に無いリポジトリの名前を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    code: int
        見つけた場合は 1, 無い場合は 0
    """
    # このスクリプトの 1 つ上をリポジトリのルートとする
    root = Path(__file__).resolve().parent.parent
    errors = name_errors(root, allowed_repos(root))
    for error in errors:
        print(error)
    code = 1 if errors else 0

    return code


if __name__ == "__main__":
    sys.exit(main())
