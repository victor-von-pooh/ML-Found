from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig
import torch

from ml_found.config.loader import format_diff, save_config
from ml_found.experiment.env_info import log_env_info
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
    # 日付と時刻のディレクトリのパスを作る
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    run_dir = project_dir / output_dir / name / date_str / time_str

    # 同じ秒に複数の実験を始めた場合は連番を付けて上書きを防ぐ
    base_dir = run_dir
    count = 1
    while run_dir.exists():
        run_dir = base_dir.with_name(f"{base_dir.name}_{count}")
        count += 1

    # ディレクトリを作る
    run_dir.mkdir(parents=True)

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
        ML-Found を使っているプロジェクトのディレクトリ, null の場合は実行した場所

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

    # 実験結果を保存するディレクトリを作る
    run_dir = make_run_dir(
        project_root, cfg.experiment.output_dir, cfg.experiment.name,
        datetime.now()
    )

    # 学習に使った config と差分を保存する
    save_config(cfg, diff, str(run_dir))

    # ロガーを設定する
    logger = get_logger(
        str(run_dir / "train.log"), cfg.experiment.log_format
    )
    logger.info(f"run directory: {run_dir}")
    logger.info(f"diff:\n\n{format_diff(diff)}")

    # 乱数シードを固定する
    fix_seed(cfg.experiment.seed, cfg.experiment.deterministic)
    logger.info(f"seed: {cfg.experiment.seed}")

    # 実験環境の情報を記録する
    log_env_info(logger, str(project_root))

    # 学習に使うデバイスを決める
    device = get_device(cfg.experiment.device)
    logger.info(f"device: {device}")

    # まとめて返す
    experiment = Experiment(
        cfg=cfg, run_dir=run_dir, logger=logger, device=device
    )

    return experiment
