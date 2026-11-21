from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional
import warnings

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException
import torch

from ml_found.config.loader import format_diff, save_config
from ml_found.config.source import save_source
from ml_found.experiment.env_info import (
    collect_env_info, get_git_diff, log_env_info
)
from ml_found.experiment.logger import get_logger
from ml_found.experiment.model_spec import (
    copied_spec_path, copied_spec_source, copy_model_spec,
    describe_model_spec, read_model_spec
)
from ml_found.experiment.seed import EXTRA_RNGS, fix_seed
from ml_found.experiment.task import (
    TASK_KEYS, build_task, describe_origin, save_task
)
from ml_found.experiment.uncaught import watch_uncaught_errors
from ml_found.training.optim import (
    check_optim_config, class_uses_gradients, resolve_optimizer_class,
    try_build_optim
)
from ml_found.training.registry import check_optim_params
from ml_found.utils.device import get_device
from ml_found.utils.requirements import REQUIREMENTS, require


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

    # 実験に写したモデルの定義 (run_dir の model_spec.yaml). 写さなかった
    # 場合と config に model.spec が無い場合は None
    model_spec_path: Optional[Path] = None

    # 写したモデルの定義の元のファイル (プロジェクトの中. 定義の中の相対の
    # 参照を辿る起点). 写さなかった場合と config に model.spec が無い場合は
    # None
    model_spec_source: Optional[Path] = None


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
    # エポック数は 1 以上, 監視する指標の向きは min か max に制限
    trainer_cfg = cfg.trainer
    if not isinstance(trainer_cfg.epochs, int) or trainer_cfg.epochs < 1:
        raise ValueError(
            f"trainer.epochs は 1 以上の整数にしてください: {trainer_cfg.epochs}"
        )
    if trainer_cfg.mode not in ("min", "max"):
        raise ValueError(
            f"trainer.mode は min か max にしてください: {trainer_cfg.mode}"
        )

    # 再開時の Scheduler の扱いは checkpoint か config に制限
    if trainer_cfg.resume_scheduler not in ("checkpoint", "config"):
        raise ValueError(
            "trainer.resume_scheduler は checkpoint か config にしてください: "
            f"{trainer_cfg.resume_scheduler}"
        )
    if trainer_cfg.resume_scheduler == "config" and cfg.get("optimizers"):
        raise ValueError(
            "optimizers を使う場合は trainer.resume_scheduler: config を使えません"
            " (再開元の Scheduler の状態を使います)"
        )

    # last.pt を保存する間隔は 1 以上に制限
    if trainer_cfg.checkpoint.every_n_epochs < 1:
        raise ValueError(
            "trainer.checkpoint.every_n_epochs は 1 以上にしてください: "
            f"{trainer_cfg.checkpoint.every_n_epochs}"
        )

    # 検証の間隔は 1 以上の整数に制限
    valid_every = trainer_cfg.valid_every_n_epochs
    if (
        not isinstance(valid_every, int) or isinstance(valid_every, bool)
        or valid_every < 1
    ):
        raise ValueError(
            "trainer.valid_every_n_epochs は 1 以上の整数にしてください: "
            f"{valid_every}"
        )

    # last.pt にベストモデルの重みを入れず best.pt も残さないと, 再開した
    # 学習でベストモデルに戻せないので警告
    checkpoint_cfg = trainer_cfg.checkpoint
    if (
        not checkpoint_cfg.last_with_best and not checkpoint_cfg.save_best
        and trainer_cfg.select_best
    ):
        message = (
            "trainer.checkpoint.last_with_best: false と save_best: false を"
            "合わせると, 再開した学習ではベストモデルの重みに戻せません"
        )
        warnings.warn(message, stacklevel=2)

    # 学習用データを評価し直さない設定では train_eval/ の指標は記録されない
    for name in (trainer_cfg.monitor, trainer_cfg.monitor_tiebreak):
        if (
            isinstance(name, str) and name.startswith("train_eval/")
            and not trainer_cfg.eval_train
        ):
            raise ValueError(
                f"{name} は trainer.eval_train: true の場合だけ記録されます"
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

    # 勾配を使わない Optimizer (config で指定したもの) も同じ組み合わせは不可
    optimizer_cls = resolve_optimizer_class(cfg.optimizer.name)
    if not class_uses_gradients(optimizer_cls) and uses_grad:
        raise ValueError(
            f"{cfg.optimizer.name} は勾配を使わないので, trainer.grad_clip_norm"
            " と log_grad_norm は使えません"
        )

    # ベストモデルを選ばない設定では, 指標の改善で止める Early Stopping は不可
    if not trainer_cfg.select_best and trainer_cfg.early_stopping.use:
        raise ValueError(
            "trainer.select_best: false では trainer.early_stopping.use は"
            "使えません (改善を判定しないため)"
        )
    if not trainer_cfg.select_best and trainer_cfg.monitor_tiebreak:
        raise ValueError(
            "trainer.select_best: false では trainer.monitor_tiebreak は"
            "使えません (ベストモデルを選ばないため)"
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

    # 重みだけのチェックポイント (best_weights_only の best.pt) は再開に使えない
    # 大きなファイルでも速いように, テンソルは読み込まずに参照
    state = torch.load(
        resolved, map_location="cpu", weights_only=True, mmap=True
    )
    if isinstance(state, dict) and state.get("weights_only"):
        raise ValueError(
            f"{resolved} は重みだけのチェックポイント (best_weights_only) なので"
            "再開には使えません. last.pt を指定してください"
        )

    # final_model.pt のような学習の状態を含まないファイルも再開に使えない
    if not isinstance(state, dict) or "monitor" not in state:
        raise ValueError(
            f"{resolved} は学習を再開するためのチェックポイントではありません "
            "(final_model.pt などのモデルの重みだけのファイル). "
            "last.pt を指定してください"
        )


def config_task(cfg: DictConfig, project_dir: Path) -> Optional[dict]:
    """
    config の task から task.yaml の中身を作る関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config
    project_dir: Path
        ML-Found を使っているプロジェクトのディレクトリ

    Returns
    ----------
    task: Optional[dict]
        task.yaml の中身, config に task の辞書が無ければ None
    """
    # 辞書でない task と, task.yaml の項目を 1 つも持たない task
    # (プロジェクトが別の意味で使う項目) は警告して対象外
    section = cfg.get("task")
    if section is None or not isinstance(section, DictConfig):
        return None
    if not any(key in TASK_KEYS for key in section.keys()):
        warnings.warn(
            "config の task に task.yaml の項目が無いので, task.yaml を"
            "書きません (プロジェクトの項目として扱います. 項目の名前の誤り"
            f"なら直してください: {', '.join(TASK_KEYS)})",
            stacklevel=3
        )
        return None
    task = build_task(
        OmegaConf.to_container(section, resolve=True), project_dir
    )

    return task


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


def read_requires(cfg: DictConfig) -> list[str]:
    """
    config の一番上の requires から, 学習に要る機能の名前を読む関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    names: list[str]
        機能の名前 (書いた順), 書いていない場合と null の場合は空
    """
    # 値の参照 (${...}) も解決して読み, 書いていなければ何も要らない扱い
    try:
        value = cfg.get("requires")
        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_container(value, resolve=True)
    except OmegaConfBaseException as e:
        raise ValueError(f"requires を読めません: {e}") from e
    if value is None:
        names = []
        return names

    # 文字のリスト (1 つの文字だけは不可) で, 同じ名前が無いことの確認
    if not isinstance(value, list) or not all(
        isinstance(name, str) and name.strip() for name in value
    ):
        raise ValueError(
            "requires は機能の名前のリストで書いてください "
            f"(例: requires: [rl]): {value!r}"
        )
    if len(set(value)) != len(value):
        raise ValueError(f"requires に同じ名前が 2 回以上あります: {value!r}")

    # 登録簿に無い名前は, 登録済みの名前を添えてエラー
    unknown = [name for name in value if name not in REQUIREMENTS]
    if unknown:
        raise ValueError(
            f"requires の {', '.join(unknown)} は登録簿にありません. "
            f"登録済み: {', '.join(sorted(REQUIREMENTS))} (プロジェクトの"
            "機能は start_experiment の前に register_requirement で登録)"
        )
    names = list(value)

    return names


def check_requires(cfg: DictConfig) -> None:
    """
    config の requires に書いた機能に要るものが入っていることを確かめる関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    None
    """
    # 書いた順に確かめ, 足りなければ入れ方を示す MissingRequirementError
    for name in read_requires(cfg):
        require(name)


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
        config, プロジェクトのディレクトリ, 保存先, ロガー, デバイス,
        モデルの定義の写しと元のファイルの場所をまとめたもの
    """
    # プロジェクトのディレクトリの決定
    if project_dir is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_dir).resolve()

    # 空の出力先が残らないように, ディレクトリを作る前に config の
    # requires に書いた機能の要るもの, デバイス,
    # Optimizer と Scheduler (クラス, 引数, 仮のパラメータで作れるか) と
    # 学習の設定, 再開元, config の task の確認
    check_requires(cfg)
    device = get_device(cfg.experiment.device)
    check_optim_config(cfg)
    check_optim_params(cfg)
    check_trainer_config(cfg)
    try_build_optim(cfg, device)
    check_resume_path(cfg, project_root)
    task = config_task(cfg, project_root)

    # 学習を始めるときのモデルの定義 (config に model.spec があれば) を,
    # 出力先を作る前に読み込み (写せない場合は理由を記録に残して続行)
    model_spec = read_model_spec(cfg, project_root)

    # 出力先を作ると未コミットの変更として数えられるので, 先に環境の情報の収集
    env_info = collect_env_info(str(project_root))
    git_diff = get_git_diff(str(project_root))

    # 実験結果を保存するディレクトリの作成
    run_dir = make_run_dir(
        project_root, cfg.experiment.output_dir, cfg.experiment.name,
        datetime.now()
    )

    # 学習に使った config と差分, モデルの定義の写し (config に model.spec
    # があれば), 起動のしかた (スクリプト, config のパス, 上書き, 定義の
    # 写しの記録), config に task があれば何の実験か (task.yaml) を保存
    save_config(cfg, diff, str(run_dir))
    spec_record = copy_model_spec(model_spec, run_dir)
    save_source(cfg, str(run_dir), project_root, spec_record)
    if task is not None:
        save_task(run_dir, task)

    # 未コミットの変更がある場合は再現できるように差分を保存
    if git_diff != b"":
        (run_dir / "git_diff.patch").write_bytes(git_diff)

    # ロガーを設定し, 保存先と差分, 正解の出どころを記録
    logger = get_logger(
        str(run_dir / "train.log"), cfg.experiment.log_format
    )
    logger.info(f"保存先: {run_dir}")
    logger.info(f"default との差分:\n\n{format_diff(diff)}")
    spec_message = describe_model_spec(model_spec, spec_record)
    if spec_message is not None:
        logger.info(spec_message)
    if task is not None:
        logger.info(
            f"task.yaml を書きました: {describe_origin(task['target']['origin'])}"
        )

    # 乱数シードを固定して記録
    fix_seed(cfg.experiment.seed, cfg.experiment.deterministic)
    logger.info(f"乱数シード: {cfg.experiment.seed}")
    if EXTRA_RNGS:
        logger.info(f"登録された外部ライブラリの乱数: {sorted(EXTRA_RNGS)}")

    # 実験環境の情報とデバイスを記録
    log_env_info(logger, env_info)
    logger.info(f"デバイス: {device}")

    # まとめて返却
    experiment = Experiment(
        cfg=cfg, project_dir=project_root, run_dir=run_dir, logger=logger,
        device=device, model_spec_path=copied_spec_path(spec_record, run_dir),
        model_spec_source=copied_spec_source(model_spec, spec_record)
    )

    # fit の外 (データの用意など) で処理されなかったエラーも, ダッシュボード
    # から分かるように train.log と run.json に記録
    watch_uncaught_errors(experiment)

    return experiment
