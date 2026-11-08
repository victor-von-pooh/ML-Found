from importlib import metadata
import json
import logging
from pathlib import Path
import platform
import subprocess

import torch

# ML-Found のパッケージのディレクトリ
PACKAGE_DIR = Path(__file__).resolve().parents[1]


def run_git(args: list, repo_dir: str) -> str:
    """
    git コマンドを実行して結果を取得する関数

    Parameters
    ----------
    args: list
        git に渡す引数
    repo_dir: str
        git コマンドを実行するディレクトリ

    Returns
    -------
    output: str
        コマンドの出力, 取得できない場合は空文字
    """
    # git リポジトリでない場合や git が無い場合は空文字にする
    try:
        output = subprocess.check_output(
            ["git", *args], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        output = ""

    return output


def get_git_info(repo_dir: str) -> str:
    """
    git リポジトリの情報を取得する関数

    Parameters
    ----------
    repo_dir: str
        情報を取得するリポジトリ内のディレクトリ

    Returns
    -------
    git_info: str
        commit id, ブランチ, ユーザー名, 未コミットの変更の有無
    """
    # commit id を取得し, 取得できない場合は git 管理外とみなす
    commit_id = run_git(["rev-parse", "HEAD"], repo_dir)
    if commit_id == "":
        git_info = f"\tgit 管理外のディレクトリです: {repo_dir}\n"
        return git_info

    # ブランチ名, タグからの位置, ユーザー名を取得する
    # submodule はブランチを持たない状態が普通なのでタグからの位置も記録する
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
    describe = run_git(["describe", "--always", "--tags"], repo_dir)
    username = run_git(["config", "user.name"], repo_dir)

    # 未コミットの変更があると commit id だけでは再現できないので記録する
    changes = run_git(["status", "--porcelain"], repo_dir)
    n_changes = len(changes.splitlines())

    # 1 つの文字列にまとめる
    spec = [
        f"\tcommit id: {commit_id}",
        f"\tbranch: {branch}",
        f"\tdescribe: {describe}",
        f"\tusername: {username}",
        f"\tuncommitted changes: {n_changes} files"
    ]
    git_info = "".join(f"{item}\n" for item in spec)

    return git_info


def get_git_diff(repo_dir: str) -> str:
    """
    未コミットの変更の差分を取得する関数

    Parameters
    ----------
    repo_dir: str
        差分を取得するリポジトリ内のディレクトリ

    Returns
    -------
    git_diff: str
        最新の commit からの差分, 変更が無い場合や git 管理外の場合は空文字
    """
    # ステージ済みかどうかに関わらず最新の commit からの差分を取得する
    git_diff = run_git(["diff", "HEAD"], repo_dir)

    return git_diff


def get_ml_found_info() -> str:
    """
    ML-Found のバージョンの情報を取得する関数

    Parameters
    ----------
    None

    Returns
    -------
    ml_found_info: str
        git 情報, または pip でインストールした場合のバージョン
    """
    # submodule などで git 管理されている場合は git の情報を使う
    # pip でインストールした場合はプロジェクト内にあっても git 管理外なので区別する
    tracked = run_git(
        ["ls-files", "--error-unmatch", "__init__.py"], str(PACKAGE_DIR)
    )
    if tracked != "":
        ml_found_info = get_git_info(str(PACKAGE_DIR))
        return ml_found_info

    # pip でインストールした場合はバージョンを取得する
    try:
        dist = metadata.distribution("ml-found")
        version = dist.version
        direct_url = dist.read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        version = "不明"
        direct_url = None

    # git からインストールした場合は commit id も取得する
    commit_id = "不明"
    if direct_url is not None:
        vcs_info = json.loads(direct_url).get("vcs_info", {})
        commit_id = vcs_info.get("commit_id", "不明")

    # 1 つの文字列にまとめる
    spec = [
        f"\tversion: {version}",
        f"\tcommit id: {commit_id}",
        f"\tlocation: {PACKAGE_DIR}"
    ]
    ml_found_info = "".join(f"{item}\n" for item in spec)

    return ml_found_info


def get_os_info() -> str:
    """
    OS と Python の情報を取得する関数

    Parameters
    ----------
    None

    Returns
    -------
    os_info: str
        OS, プロセッサ, マシン, ノード, Python のバージョン
    """
    # OS の情報を並べる
    spec = [
        f"\tOS: {platform.system()} {platform.release()}",
        f"\tProcessor: {platform.processor()}",
        f"\tMachine: {platform.machine()}",
        f"\tNode: {platform.node()}",
        f"\tPython Version: {platform.python_version()}"
    ]

    # 1 つの文字列にまとめる
    os_info = "".join(f"{item}\n" for item in spec)

    return os_info


def get_torch_info() -> str:
    """
    PyTorch と GPU の情報を取得する関数

    Parameters
    ----------
    None

    Returns
    -------
    torch_info: str
        PyTorch のバージョンと使える GPU の情報
    """
    # PyTorch のバージョンと各 GPU が使えるかどうかを並べる
    spec = [
        f"\tPyTorch Version: {torch.__version__}",
        f"\tCUDA Available: {torch.cuda.is_available()}",
        f"\tMPS Available: {torch.backends.mps.is_available()}"
    ]

    # CUDA が使える場合はバージョンと GPU 名も並べる
    if torch.cuda.is_available():
        spec.append(f"\tCUDA Version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            spec.append(f"\tGPU {i}: {torch.cuda.get_device_name(i)}")

    # 1 つの文字列にまとめる
    torch_info = "".join(f"{item}\n" for item in spec)

    return torch_info


def get_package_list() -> str:
    """
    インストール済みの Python パッケージの一覧を取得する関数

    Parameters
    ----------
    None

    Returns
    -------
    package_list: str
        パッケージ名とバージョンの一覧
    """
    # pip が無い環境でも取得できるように実行中の Python から直接調べる
    # メタデータが壊れていて名前が取れないものは除く
    packages = sorted(
        (dist.metadata["Name"], dist.version)
        for dist in metadata.distributions()
        if dist.metadata["Name"] is not None
    )

    # 1 つの文字列にまとめる
    package_list = "".join(
        f"\t{name}=={version}\n" for name, version in packages
    )

    return package_list


def collect_env_info(project_dir: str) -> dict[str, str]:
    """
    実験環境の情報をまとめて取得する関数

    Parameters
    ----------
    project_dir: str
        ML-Found を使っているプロジェクトのディレクトリ

    Returns
    -------
    env_info: dict[str, str]
        項目の名前と情報
    """
    # プロジェクトと ML-Found のバージョン, OS, PyTorch, パッケージの情報を集める
    env_info = {
        "プロジェクトの git 情報": get_git_info(project_dir),
        "ML-Found の情報": get_ml_found_info(),
        "OS の情報": get_os_info(),
        "PyTorch の情報": get_torch_info(),
        "パッケージ一覧": get_package_list()
    }

    return env_info


def log_env_info(logger: logging.Logger, env_info: dict[str, str]) -> None:
    """
    実験環境の情報をログに記録する関数

    Parameters
    ----------
    logger: logging.Logger
        ロガー
    env_info: dict[str, str]
        collect_env_info で取得した項目の名前と情報

    Returns
    -------
    None
    """
    # 項目ごとに記録する
    for title, info in env_info.items():
        logger.info(f"{title}:\n\n{info}")
