from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig
import torch

from ml_found.config.loader import format_diff, save_config
from ml_found.experiment.env_info import (
    collect_env_info, get_git_diff, log_env_info
)
from ml_found.experiment.logger import get_logger
from ml_found.experiment.seed import fix_seed
from ml_found.utils.device import get_device


@dataclass
class Experiment:
    # 学習に使う config
    cfg: DictConfig

    # 実験結果を保存するディレクトリ
    run_dir: Path

    # 実験のロガー
    logger: logging.Logger

    # 学習に使うデバイス
    device: torch.device


def check_name(name: str) -> None:
    """
    実験の名前が出力先のディレクトリ名として使えることを確かめる関数

    Parameters
    ----------
    name: str
        実験の名前

    Returns
    -------
    None
    """
    # 出力先の外にディレクトリを作らないように区切り文字や .. を禁止する
    if name in ("", ".", "..") or "/" in name or "\\" in name:
        raise ValueError(
            f"experiment.name はディレクトリ名として使える名前にしてください: "
            f"{name}"
        )


def make_run_dir(
    project_dir: Path, output_dir: str, name: str, now: datetime
) -> Path:
    """
    実験結果を保存するディレクトリを作る関数

    Parameters
    ----------
    project_dir: Path
        ML-Found を使っているプロジェクトのディレクトリ
    output_dir: str
        出力先のディレクトリ, 相対パスの場合はプロジェクトのディレクトリから辿る
    name: str
        実験の名前
    now: datetime
        実験を開始した日時

    Returns
    -------
    run_dir: Path
        <output_dir>/<name>/<日付>/<時刻>/ のディレクトリ
    """
    # 実験の名前を確かめる
    check_name(name)

    # 日付のディレクトリまで作る
    date_dir = project_dir / output_dir / name / now.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    # 同じ秒に複数の実験を始めた場合は連番を付けて上書きを防ぐ
    # 並列に実行しても衝突しないように, 作れなかった場合に次の番号を試す
    time_str = now.strftime("%H-%M-%S")
    count = 0
    while True:
        dir_name = time_str if count == 0 else f"{time_str}_{count}"
        run_dir = date_dir / dir_name
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            count += 1

    return run_dir


def start_experiment(
    cfg: DictConfig, diff: dict, project_dir: Optional[str] = None
) -> Experiment:
    """
    実験を開始する関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config
    diff: dict
        default から変わった項目
    project_dir: Optional[str] = None
        ML-Found を使っているプロジェクトのディレクトリ, None の場合は実行した場所

    Returns
    -------
    experiment: Experiment
        config, 保存先, ロガー, デバイスをまとめたもの
    """
    # プロジェクトのディレクトリを決める
    if project_dir is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_dir).resolve()

    # 空の出力先が残らないように, ディレクトリを作る前にデバイスを確かめる
    device = get_device(cfg.experiment.device)

    # 出力先を作ると未コミットの変更として数えられるので, 先に環境の情報を集める
    env_info = collect_env_info(str(project_root))
    git_diff = get_git_diff(str(project_root))

    # 実験結果を保存するディレクトリを作る
    run_dir = make_run_dir(
        project_root, cfg.experiment.output_dir, cfg.experiment.name,
        datetime.now()
    )

    # 学習に使った config と差分を保存する
    save_config(cfg, diff, str(run_dir))

    # 未コミットの変更がある場合は再現できるように差分を保存する
    if git_diff != "":
        (run_dir / "git_diff.patch").write_text(
            git_diff + "\n", encoding="utf-8"
        )

    # ロガーを設定する
    logger = get_logger(
        str(run_dir / "train.log"), cfg.experiment.log_format
    )
    logger.info(f"保存先: {run_dir}")
    logger.info(f"default との差分:\n\n{format_diff(diff)}")

    # 乱数シードを固定する
    fix_seed(cfg.experiment.seed, cfg.experiment.deterministic)
    logger.info(f"乱数シード: {cfg.experiment.seed}")

    # 実験環境の情報を記録する
    log_env_info(logger, env_info)
    logger.info(f"デバイス: {device}")

    # まとめて返す
    experiment = Experiment(
        cfg=cfg, run_dir=run_dir, logger=logger, device=device
    )

    return experiment
