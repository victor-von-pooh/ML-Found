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
from ml_found.training.optim import check_optim_config
from ml_found.training.registry import check_optim_params
from ml_found.utils.device import get_device


@dataclass
class Experiment:
    # 学習に使う config
    cfg: DictConfig

    # ML-Found を使っているプロジェクトのディレクトリ
    project_dir: Path

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
    ----------
    None
    """
    # 出力先の外にディレクトリを作らないように区切り文字や .. を禁止
    if name in ("", ".", "..") or "/" in name or "\\" in name:
        raise ValueError(
            "experiment.name はディレクトリ名として使える名前にしてください: "
            f"{name}"
        )


def check_trainer_config(cfg: DictConfig) -> None:
    """
    学習の設定の組み合わせが正しいことを確かめる関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    None
    """
    # last.pt を保存する間隔は 1 以上に制限
    trainer_cfg = cfg.trainer
    if trainer_cfg.checkpoint.every_n_epochs < 1:
        raise ValueError(
            "trainer.checkpoint.every_n_epochs は 1 以上にしてください: "
            f"{trainer_cfg.checkpoint.every_n_epochs}"
        )

    # 同じ値の場合に比べる指標の向きは min か max に制限
    if trainer_cfg.monitor_tiebreak_mode not in ("min", "max"):
        raise ValueError(
            "trainer.monitor_tiebreak_mode は min か max にしてください: "
            f"{trainer_cfg.monitor_tiebreak_mode}"
        )

    # 自分で更新する設定では ML-Found がクリップと記録をできないのでエラー
    uses_grad = (
        trainer_cfg.grad_clip_norm is not None or trainer_cfg.log_grad_norm
    )
    if trainer_cfg.manual_optimization and uses_grad:
        raise ValueError(
            "manual_optimization では grad_clip_norm と log_grad_norm は"
            "使えません"
        )


def check_resume_path(cfg: DictConfig, project_dir: Path) -> None:
    """
    再開元のチェックポイントが存在することを確かめる関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config
    project_dir: Path
        ML-Found を使っているプロジェクトのディレクトリ

    Returns
    ----------
    None
    """
    # 再開しない場合は確認不要
    path = cfg.trainer.resume_from
    if path is None:
        return

    # 相対パスは Trainer と同じくプロジェクトのディレクトリから辿って確認
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = project_dir / resolved
    if not resolved.exists():
        raise FileNotFoundError(
            f"trainer.resume_from のファイルがありません: {resolved} "
            "(相対パスはプロジェクトのディレクトリから辿ります)"
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
    ----------
    run_dir: Path
        <output_dir>/<name>/<日付>/<時刻>/ のディレクトリ
    """
    # 実験の名前の確認
    check_name(name)

    # 日付のディレクトリまで作成
    date_dir = project_dir / output_dir / name / now.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    # 同じ秒に複数の実験を始めた場合は連番を付けて上書きの防止
    # 並列に実行しても衝突しないように, 作れなかった場合に次の番号の試行
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
    ----------
    experiment: Experiment
        config, プロジェクトのディレクトリ, 保存先, ロガー, デバイスをまとめたもの
    """
    # プロジェクトのディレクトリの決定
    if project_dir is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_dir).resolve()

    # 空の出力先が残らないように, ディレクトリを作る前にデバイス,
    # Optimizer と Scheduler (クラスと引数) と学習の設定, 再開元の確認
    device = get_device(cfg.experiment.device)
    check_optim_config(cfg)
    check_optim_params(cfg)
    check_trainer_config(cfg)
    check_resume_path(cfg, project_root)

    # 出力先を作ると未コミットの変更として数えられるので, 先に環境の情報の収集
    env_info = collect_env_info(str(project_root))
    git_diff = get_git_diff(str(project_root))

    # 実験結果を保存するディレクトリの作成
    run_dir = make_run_dir(
        project_root, cfg.experiment.output_dir, cfg.experiment.name,
        datetime.now()
    )

    # 学習に使った config と差分を保存
    save_config(cfg, diff, str(run_dir))

    # 未コミットの変更がある場合は再現できるように差分を保存
    if git_diff != b"":
        (run_dir / "git_diff.patch").write_bytes(git_diff)

    # ロガーを設定し, 保存先と差分を記録
    logger = get_logger(
        str(run_dir / "train.log"), cfg.experiment.log_format
    )
    logger.info(f"保存先: {run_dir}")
    logger.info(f"default との差分:\n\n{format_diff(diff)}")

    # 乱数シードを固定して記録
    fix_seed(cfg.experiment.seed, cfg.experiment.deterministic)
    logger.info(f"乱数シード: {cfg.experiment.seed}")

    # 実験環境の情報とデバイスを記録
    log_env_info(logger, env_info)
    logger.info(f"デバイス: {device}")

    # まとめて返却
    experiment = Experiment(
        cfg=cfg, project_dir=project_root, run_dir=run_dir, logger=logger,
        device=device
    )

    return experiment
