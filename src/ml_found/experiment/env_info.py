from importlib import metadata
import logging
from pathlib import Path
import platform
import subprocess

import torch


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

    # ブランチ名とユーザー名を取得する
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
    username = run_git(["config", "user.name"], repo_dir)

    # 未コミットの変更があると commit id だけでは再現できないので記録する
    changes = run_git(["status", "--porcelain"], repo_dir)
    n_changes = len(changes.splitlines())

    # 1 つの文字列にまとめる
    spec = [
        f"\tcommit id: {commit_id}",
        f"\tbranch: {branch}",
        f"\tusername: {username}",
        f"\tuncommitted changes: {n_changes} files"
    ]
    git_info = "".join(f"{item}\n" for item in spec)

    return git_info


def get_os_info() -> str:
    """
    OS と Python の情報を取得する関数

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

    Returns
    -------
    package_list: str
        パッケージ名とバージョンの一覧
    """
    # pip が無い環境でも取得できるように実行中の Python から直接調べる
    packages = sorted(
        (dist.metadata["Name"], dist.version)
        for dist in metadata.distributions()
    )

    # 1 つの文字列にまとめる
    package_list = "".join(
        f"\t{name}=={version}\n" for name, version in packages
    )

    return package_list


def log_env_info(logger: logging.Logger, project_dir: str) -> None:
    """
    実験環境の情報をまとめてログに記録する関数

    Parameters
    ----------
    logger: logging.Logger
        ロガー
    project_dir: str
        ML-Found を使っているプロジェクトのディレクトリ

    Returns
    -------
    None
    """
    # ML-Found 自身のディレクトリを取得する
    ml_found_dir = str(Path(__file__).resolve().parent)

    # プロジェクトと ML-Found のそれぞれの git 情報を記録する
    logger.info(f"project git information:\n\n{get_git_info(project_dir)}")
    logger.info(f"ML-Found git information:\n\n{get_git_info(ml_found_dir)}")

    # OS, PyTorch, パッケージの情報を記録する
    logger.info(f"OS information:\n\n{get_os_info()}")
    logger.info(f"PyTorch information:\n\n{get_torch_info()}")
    logger.info(f"package list:\n\n{get_package_list()}")
