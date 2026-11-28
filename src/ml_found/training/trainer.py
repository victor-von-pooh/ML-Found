from collections import Counter
import contextlib
import copy
import functools
import math
from pathlib import Path
import shutil
import time
import warnings
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Union

from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn, optim
from tqdm import tqdm

from ml_found.experiment.seed import (
    EXTRA_RNGS, FIXED_RNGS, get_rng_state, set_rng_state, set_seed
)
from ml_found.training.callbacks import (
    Checkpoint, EarlyStopping, copy_state_dict, load_checkpoint,
    save_checkpoint, without_best_model
)
from ml_found.training.ema import (
    build_ema_model, ema_state_dict, reset_ema_model
)
from ml_found.training.finetune import (
    autocast_context, complete_weights, frozen_parameters, listed,
    load_init_weights, omit_keys
)
from ml_found.training.history import (
    History, LrSteps, MetricTracker, to_float
)
from ml_found.training.multi import (
    NamedOptimizers, NamedSchedulers, split_parameters
)
from ml_found.training.optim import (
    auto_params, build_optimizer, build_scheduler, check_scheduler,
    class_uses_gradients, missing_data_size_params, named_errors, named_views,
    needs_data, requires_closure, step_scheduler, to_kwargs,
    uses_named_optimizers
)
from ml_found.training.heartbeat import PostHeartbeat, end_heartbeats
from ml_found.training.status import STOP_FILE, RunStatus, record_not_started
from ml_found.utils.batch import find_batch_size, move_to_device
from ml_found.utils.model_summary import (
    count_parameters, save_model_summary, summarize_model
)
from ml_found.utils.plotting import plot_history, plot_lr_steps

# Experiment は型ヒントだけに使用 (experiment は training を使うので循環を回避)
if TYPE_CHECKING:
    from ml_found.experiment.setup import Experiment


def loader_generators(loader: Iterable) -> list[torch.Generator]:
    """
    DataLoader とその sampler が持つ乱数生成器を取り出す関数

    Parameters
    ----------
    loader: Iterable
        DataLoader など

    Returns
    ----------
    generators: list[torch.Generator]
        重複を除いた乱数生成器, 持たない場合は空
    """
    # DataLoader の generator と, sampler に渡した generator を重複なく収集
    generators: list[torch.Generator] = []
    candidates = [
        getattr(loader, "generator", None),
        getattr(getattr(loader, "sampler", None), "generator", None)
    ]
    for generator in candidates:
        if isinstance(generator, torch.Generator) and all(
            generator is not other for other in generators
        ):
            generators.append(generator)

    return generators


def has_loader_state(loader: Any) -> bool:
    """
    チェックポイントに状態を保存して戻せる loader かを判定する関数

    Parameters
    ----------
    loader: Any
        学習か検証に使う loader, 検証しない場合は None

    Returns
    ----------
    stateful: bool
        checkpoint_state と load_checkpoint_state を持つ場合は True
        (RolloutLoader など)
    """
    # 2 つのメソッドを両方持つものだけを対象
    stateful = callable(getattr(loader, "checkpoint_state", None)) and (
        callable(getattr(loader, "load_checkpoint_state", None))
    )

    return stateful


def clone_to_cpu(value: Any) -> Any:
    """
    テンソルを含む状態を, cpu の写しにする関数

    Parameters
    ----------
    value: Any
        テンソル, またはテンソルを含む辞書, リスト, タプル

    Returns
    ----------
    cloned: Any
        テンソルを cpu に写した同じ形のもの, それ以外の値は深い写し
    """
    # 入れ物は中身を順に写し, テンソルは cpu に写して他は深い写し
    if isinstance(value, torch.Tensor):
        cloned = value.detach().cpu().clone()
    elif isinstance(value, dict):
        cloned = {key: clone_to_cpu(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        items = [clone_to_cpu(item) for item in value]
        cloned = items if isinstance(value, list) else tuple(items)
    else:
        cloned = copy.deepcopy(value)

    return cloned


class StopWithinEpoch(Exception):
    # エポックの途中で止める要求を受けたことを伝える例外
    pass


# Scheduler の状態のうち, 学習の進み具合で変わる値 (引数ではないもの)
PROGRESS_KEYS = {
    "last_epoch", "base_lrs", "T_cur", "T_i", "best", "num_bad_epochs",
    "cooldown_counter"
}

# Scheduler の状態のうち, 内部の名前でも引数から決まる値
# (OneCycleLR の pct_start, three_phase, anneal_strategy)
ARGUMENT_PRIVATE_KEYS = {"_schedule_phases", "_anneal_func_type"}

# Scheduler の状態のうち, 他の引数から決まる値 (ReduceLROnPlateau の mode
# から決まる値と, min_lr から決まる値)
DERIVED_KEYS = {"mode_worse", "default_min_lr"}

# Scheduler が Optimizer のパラメータグループに書き込む引数の値
SCHEDULER_GROUP_KEYS = ("max_lr", "min_lr", "base_momentum", "max_momentum")


def scheduler_classes(
    scheduler: Optional[optim.lr_scheduler.LRScheduler]
) -> list[str]:
    """
    Scheduler の組み立て (クラス名, warmup を付けた場合は中のクラス名) を返す
    関数

    Parameters
    ----------
    scheduler: Optional[optim.lr_scheduler.LRScheduler]
        Scheduler, 使わない場合は None

    Returns
    ----------
    names: list[str]
        クラス名の並び (例: SequentialLR, LinearLR, CosineAnnealingLR)
    """
    # 使わない場合は空, SequentialLR は中のクラスも並べる
    if scheduler is None:
        return []
    names = [type(scheduler).__name__]
    if isinstance(scheduler, optim.lr_scheduler.SequentialLR):
        names += [type(inner).__name__ for inner in scheduler._schedulers]

    return names


def scheduler_label(
    scheduler: Optional[optim.lr_scheduler.LRScheduler]
) -> Optional[str]:
    """
    run.json に書く Scheduler の名前を返す関数

    Parameters
    ----------
    scheduler: Optional[optim.lr_scheduler.LRScheduler]
        Scheduler, 使わない場合と学習を始めるまで作らない場合は None

    Returns
    ----------
    label: Optional[str]
        クラス名, warmup を付けた場合は中のクラス名も付けた名前
        (例: SequentialLR(LinearLR, CosineAnnealingLR)), 無い場合は None
    """
    names = scheduler_classes(scheduler)
    if not names:
        return None
    label = names[0]
    if len(names) > 1:
        label += f"({', '.join(names[1:])})"

    return label


def json_value(value: Any) -> Any:
    """
    run.json に書ける値に直す関数

    Parameters
    ----------
    value: Any
        Scheduler の状態の値

    Returns
    ----------
    converted: Any
        数, 文字, 真偽値, null と, それらのリストと辞書. 無限大と非数は
        文字, 回数の辞書 (MultiStepLR の milestones) は小さい順の並び,
        それ以外の書けない値は文字の表し
    """
    # 配列と NumPy の数は Python の値に直してから変換
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()
    if isinstance(value, Counter):
        converted = json_value(sorted(value.elements()))
    elif value is None or isinstance(value, (bool, int, str)):
        converted = value
    elif isinstance(value, float):
        converted = value if math.isfinite(value) else str(value)
    elif isinstance(value, dict):
        converted = {str(key): json_value(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        converted = [json_value(item) for item in value]
    else:
        converted = str(value)

    return converted


def describe_callable(function: Callable) -> Union[str, dict]:
    """
    Scheduler に渡した関数を, run.json に書ける形で表す関数

    Parameters
    ----------
    function: Callable
        lr_lambda や scale_fn に渡した関数か, 呼び出せるオブジェクト

    Returns
    ----------
    described: Union[str, dict]
        関数の名前 (無名の関数は <lambda>), オブジェクトはクラスの名前.
        functools.partial は, 元の関数 (function) と固定した引数 (args,
        keywords) の辞書
    """
    # partial は元の関数と固定した引数 (transformers の warmup など)
    if isinstance(function, functools.partial):
        described = {
            "function": describe_callable(function.func),
            "args": json_value(function.args),
            "keywords": json_value(function.keywords)
        }
        return described

    # 関数の名前, 名前を持たないオブジェクトはクラスの名前
    described = getattr(function, "__name__", None)
    if not isinstance(described, str):
        described = type(function).__name__

    return described


def scheduler_params(
    scheduler: Optional[optim.lr_scheduler.LRScheduler]
) -> Optional[dict]:
    """
    run.json に書く Scheduler の主な引数を返す関数

    Scheduler の状態から, 学習の進み具合で変わる値と内部の値を除き,
    状態の名前のまま並べる. 引数は名前を変えて持つものがある (OneCycleLR の
    epochs は total_steps など) ので, クラスの引数ではなく状態から取る

    Parameters
    ----------
    scheduler: Optional[optim.lr_scheduler.LRScheduler]
        Scheduler, 使わない場合と学習を始めるまで作らない場合は None

    Returns
    ----------
    params: Optional[dict]
        引数の名前と値. 中に Scheduler を持つもの (SequentialLR,
        ChainedScheduler) は, 中の Scheduler ごとのクラスの名前と引数を
        schedulers に並べた辞書. 使わない場合は None
    """
    if scheduler is None:
        return None

    # 中に Scheduler を持つものは, 区切りと中の Scheduler ごとの引数
    lr_scheduler = optim.lr_scheduler
    if isinstance(
        scheduler, (lr_scheduler.SequentialLR, lr_scheduler.ChainedScheduler)
    ):
        params = {}
        if isinstance(scheduler, lr_scheduler.SequentialLR):
            params["milestones"] = json_value(scheduler._milestones)
        params["schedulers"] = [
            {
                "scheduler": type(inner).__name__,
                "params": scheduler_params(inner)
            }
            for inner in scheduler._schedulers
        ]
        return params

    # 学習の進み具合で変わる値と内部の値以外の状態
    # CyclicLR の base_lrs は Optimizer の学習率ではなく引数の base_lr
    state = scheduler.state_dict()
    params = {
        key: json_value(value) for key, value in state.items()
        if (
            not key.startswith("_") and key not in PROGRESS_KEYS
            and key not in DERIVED_KEYS
        ) or key in ARGUMENT_PRIVATE_KEYS
    }
    if isinstance(scheduler, lr_scheduler.CyclicLR):
        params["base_lrs"] = json_value(state.get("base_lrs"))

    # 渡した関数 (状態には残らない) の名前と, partial で固定した引数
    if hasattr(scheduler, "lr_lambdas"):
        params["lr_lambdas"] = [
            describe_callable(function)
            for function in scheduler.lr_lambdas
        ]
    custom = getattr(scheduler, "_scale_fn_custom", None)
    if custom is not None:
        params["scale_fn"] = describe_callable(custom)

    # OneCycleLR が Optimizer に書き込む引数の値 (max_lr など)
    if isinstance(scheduler, lr_scheduler.OneCycleLR):
        params["groups"] = json_value(
            scheduler_group_values(scheduler.optimizer)
        )

    return params


def saved_scheduler_classes(state: dict) -> Optional[list[str]]:
    """
    チェックポイントに残る Scheduler の組み立てを返す関数

    Parameters
    ----------
    state: dict
        チェックポイントの内容

    Returns
    ----------
    names: Optional[list[str]]
        クラス名の並び. 組み立てを記録する前のチェックポイントでは,
        SequentialLR なら ["SequentialLR"], そうでなければ None (不明)
    """
    # 記録がある場合はそのまま, 無い場合は状態の形から推定
    names = state.get("scheduler_classes")
    if names is None and "_schedulers" in (state.get("scheduler") or {}):
        names = ["SequentialLR"]

    return names


def same_classes(old: Optional[list[str]], new: list[str]) -> bool:
    """
    再開元と今の Scheduler の組み立てが同じかを確かめる関数

    Parameters
    ----------
    old: Optional[list[str]]
        再開元の組み立て (saved_scheduler_classes の戻り値)
    new: list[str]
        今の組み立て

    Returns
    ----------
    same: bool
        同じと見なせる場合は True
    """
    # 記録の無い古いチェックポイントは, SequentialLR かどうかだけを比較
    if old is None:
        same = not (new and new[0] == "SequentialLR")
    elif old == ["SequentialLR"]:
        same = bool(new) and new[0] == "SequentialLR"
    else:
        same = old == new

    return same


def sequential_arguments(state: dict) -> dict:
    """
    SequentialLR (warmup を付けた Scheduler) の状態を, 比べられる形に並べる
    関数

    Parameters
    ----------
    state: dict
        SequentialLR の状態

    Returns
    ----------
    arguments: dict
        区切りの回数 (milestones) と, 中の Scheduler ごとの引数の値
        (名前は scheduler<番号>.<引数>)
    """
    # 区切りと, 中の Scheduler の進み具合と内部の値以外を収集
    arguments = {"milestones": state.get("_milestones")}
    for i, inner in enumerate(state.get("_schedulers", [])):
        for key, value in inner.items():
            if (
                not key.startswith("_") and key not in PROGRESS_KEYS
            ) or key in ARGUMENT_PRIVATE_KEYS:
                arguments[f"scheduler{i}.{key}"] = value

    return arguments


def scheduler_group_values(optimizer: optim.Optimizer) -> list[dict]:
    """
    Scheduler が Optimizer のパラメータグループに書き込んだ値を取り出す関数

    Parameters
    ----------
    optimizer: optim.Optimizer
        Scheduler を付けた Optimizer

    Returns
    ----------
    values: list[dict]
        グループごとの, OneCycleLR の max_lr などの値
    """
    # グループごとに, 書き込まれている値だけを収集
    values = [
        {key: group[key] for key in SCHEDULER_GROUP_KEYS if key in group}
        for group in optimizer.param_groups
    ]

    return values


# extra に使えない名前 (記録の他の列と重なるもの)
RESERVED_EXTRA = {"epoch", "lr"}


def check_extra(extra: Any) -> None:
    """
    step の戻り値の extra (平均せずに記録だけする値) を確かめる関数

    Parameters
    ----------
    extra: Any
        step の戻り値の extra, 無い場合は None

    Returns
    ----------
    None
    """
    if extra is None:
        return
    if not isinstance(extra, dict):
        raise ValueError(
            "step の extra は「名前: 値」の辞書で返してください"
        )
    for key, value in extra.items():
        # 名前は空でない文字列で, 学習曲線のファイル名と train.log の
        # 「名前=値, 」の形を壊す文字と, 記録の他の列の名前は禁止
        bad_name = (
            not isinstance(key, str) or not key or key in RESERVED_EXTRA
            or any(char in key for char in "/,=")
        )
        if bad_name:
            raise ValueError(
                f"extra の名前は空でない文字列で, /, カンマ, = と "
                f"{sorted(RESERVED_EXTRA)} は使えません: {key!r}"
            )

        # 値は数か 1 つの値のテンソル
        is_number = isinstance(value, (int, float)) and not isinstance(
            value, bool
        )
        is_scalar = isinstance(value, torch.Tensor) and value.numel() == 1
        if not (is_number or is_scalar):
            raise ValueError(
                f"extra の {key} は数か 1 つの値のテンソルにしてください: "
                f"{value!r}"
            )


def record_init_error(
    experiment: "Experiment", error: BaseException
) -> None:
    """
    Trainer の作成で起きたエラーを, train.log と run.json に記録する関数

    出力先は作られた後なので, 何が起きたかを後から分かるように残す.
    記録に失敗しても元のエラーを隠さないように, 記録の失敗は警告だけにする

    Parameters
    ----------
    experiment: Experiment
        始めた実験
    error: BaseException
        起きたエラー, キーボードの割り込みは止めたものとして記録

    Returns
    ----------
    None
    """
    interrupted = isinstance(error, KeyboardInterrupt)
    try:
        if interrupted:
            experiment.logger.warning(
                "キーボードの割り込みで Trainer の作成が止まりました"
            )
        else:
            experiment.logger.exception("Trainer の作成でエラーが起きました")
        record_not_started(
            experiment.run_dir, experiment.logger, experiment.cfg,
            str(experiment.device), error
        )
    except Exception as record_error:
        warnings.warn(
            f"Trainer の作成のエラーを記録できませんでした: {record_error}",
            stacklevel=2
        )


class Trainer:
    def __init__(
        self, model: nn.Module, experiment: "Experiment",
        step_fn: Optional[Callable[[nn.Module, Any, str], dict]] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    ):
        # 作成で失敗した場合も, 出力先に記録を残してから投げ直し
        # (継承したクラスの setup などと重ならないように, 部品の用意は
        # _build_parts で実行)
        try:
            self._build_parts(model, experiment, step_fn, optimizer, scheduler)
        except (Exception, KeyboardInterrupt) as error:
            record_init_error(experiment, error)
            raise

    def _build_parts(
        self, model: nn.Module, experiment: "Experiment",
        step_fn: Optional[Callable[[nn.Module, Any, str], dict]],
        optimizer: Optional[optim.Optimizer],
        scheduler: Optional[optim.lr_scheduler.LRScheduler]
    ) -> None:
        """
        学習に使う部品を用意する関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        experiment: Experiment
            start_experiment で始めた実験
        step_fn: Optional[Callable[[nn.Module, Any, str], dict]]
            1 バッチ分の計算を行う関数, step を上書きする場合は None
        optimizer: Optional[optim.Optimizer]
            自分で作った Optimizer, None の場合は config から作成
        scheduler: Optional[optim.lr_scheduler.LRScheduler]
            自分で作った Scheduler, None の場合は config から作成

        Returns
        ----------
        None
        """
        # 実験, 学習の config, ロガー, デバイスを保存
        self.experiment = experiment
        self.trainer_cfg = experiment.cfg.trainer
        self.logger = experiment.logger
        self.device = experiment.device

        # start_experiment の後に登録した外部ライブラリの乱数は固定されて
        # いないので警告
        late = sorted(set(EXTRA_RNGS) - FIXED_RNGS)
        if late:
            message = (
                f"乱数 {late} は start_experiment の後に登録したので, シードで"
                "固定されていません. start_experiment の前に register_rng を"
                "呼んでください"
            )
            self.logger.warning(message)
            warnings.warn(message, stacklevel=3)

        # モデルをデバイスに移して保存
        self.model = model.to(self.device)

        # config に学習を始める前に読む重みがあれば, EMA の写しを作る前に
        # 読み込み
        # (再開する場合も, 再開元に無い学習しないパラメータの値に使用)
        self.init_weights_report: Optional[dict] = None
        init_cfg = self.trainer_cfg.init_weights
        if init_cfg.path is not None:
            self.init_weights_report = load_init_weights(
                self.model, init_cfg, experiment.project_dir, self.logger
            )

        # config で EMA を使う場合は, 重みの指数移動平均を持つモデルを作成
        # (学習するモデルの写しを同じデバイスに用意)
        self.ema_model: Optional[optim.swa_utils.AveragedModel] = None
        ema_cfg = self.trainer_cfg.ema
        if ema_cfg.decay is not None:
            self.ema_model = build_ema_model(
                self.model, ema_cfg.decay, ema_cfg.use_buffers
            )

        # 1 ステップの計算を行う関数を保存
        self.step_fn = step_fn

        # パラメータを持たないモデル (バッファだけのモデル) かどうかを保存
        # (勾配で学習しないので, Optimizer と Scheduler を作らない)
        self.no_params = next(iter(self.model.parameters()), None) is None

        # Optimizer と Scheduler の作成 (名前付きの複数の Optimizer は別に作成)
        self.named = (
            uses_named_optimizers(experiment.cfg) and not self.no_params
        )
        self.auto_values: dict[str, int] = {}
        self.warmup_from_ratio = False
        self.warmup_ratio_labels: set[str] = set()
        # 予定の総ステップ数 (OneCycleLR の総ステップ数との食い違いの確認に使用)
        self.planned_total_steps: Optional[int] = None
        if self.no_params:
            self.build_no_param_parts(optimizer, scheduler)
        elif self.named:
            self.build_named_parts(optimizer, scheduler)
        else:
            self.build_single_parts(optimizer, scheduler)

        # Scheduler と間隔の組み合わせの確認 (Scheduler が無い場合は間隔だけ)
        for _, scheduler_entry, _ in self.scheduler_entries():
            check_scheduler(
                scheduler_entry, self.scheduler_interval,
                experiment.cfg.trainer.mode
            )

        # Early Stopping を使う場合は作成
        self.early_stopping = None
        if self.trainer_cfg.early_stopping.use:
            self.early_stopping = EarlyStopping(
                patience=self.trainer_cfg.early_stopping.patience,
                min_delta=self.trainer_cfg.early_stopping.min_delta,
                mode=self.trainer_cfg.mode,
                tiebreak_mode=self.trainer_cfg.monitor_tiebreak_mode
            )

        # ベストモデルと最後のモデルを保存する仕組みの作成
        self.checkpoint = Checkpoint(
            experiment.run_dir, mode=self.trainer_cfg.mode,
            save_best=self.trainer_cfg.checkpoint.save_best,
            save_last=self.trainer_cfg.checkpoint.save_last,
            tiebreak_mode=self.trainer_cfg.monitor_tiebreak_mode,
            best_weights_only=self.trainer_cfg.checkpoint.best_weights_only,
            last_with_best=self.trainer_cfg.checkpoint.last_with_best
        )

        # エポックごとの記録と, 最初と最後のエポックを保存
        # (同じ Trainer で fit を呼び直した場合は, 前の fit の続きのエポックから
        # trainer.epochs の分だけの範囲)
        self.history = History()
        self.start_epoch = 1
        self.end_epoch = self.trainer_cfg.epochs
        self.loop_started = False
        self.last_monitor: Optional[str] = None
        self.fit_finished = False

        # fit の終わりにベストモデルなどの重みに置き換えた場合の, 置き換える前
        # (最後のエポック) の重みの入れ物 (fit を呼び直した場合の続きの起点)
        self.continue_weights: Optional[dict] = None

        # エポックの終わりに呼ぶ関数 (add_callback で追加)
        self.callbacks: list[Callable[[dict], Optional[bool]]] = []

        # step ごとに学習率を変える場合の, 更新ごとの学習率の記録
        self.lr_steps = LrSteps(self.lr_columns)

        # step から参照できるように, 今のエポックとパラメータの更新回数を保存
        self.epoch = 0
        self.global_step = 0

        # LBFGS のように損失を計算し直す closure が必要な Optimizer かどうかと,
        # 勾配を使わない Optimizer かどうかを保存
        # (名前付きの Optimizer は, どちらでもないことを config の確認で保証)
        # (パラメータを持たないモデルは更新しないので, どちらも見ない)
        self.use_closure = (
            not self.named and not self.no_params
            and requires_closure(self.optimizer)
        )
        self.use_gradients = (
            self.named or self.no_params
            or class_uses_gradients(type(self.optimizer))
        )
        if not self.use_gradients and (
            self.trainer_cfg.grad_clip_norm is not None
            or self.trainer_cfg.log_grad_norm
        ):
            raise ValueError(
                f"{type(self.optimizer).__name__} は勾配を使わないので, "
                "trainer.grad_clip_norm と log_grad_norm は使えません"
            )

        # 勾配の蓄積は, ML-Found が逆伝播して 1 つの Optimizer で更新する場合
        # だけ (自分で作って渡した Optimizer も同じ確認)
        self.accumulate_grad_batches = int(
            self.trainer_cfg.accumulate_grad_batches
        )
        if self.accumulate_grad_batches > 1 and (
            self.named or not self.use_gradients or self.use_closure
            or self.trainer_cfg.manual_optimization
        ):
            raise ValueError(
                "trainer.accumulate_grad_batches は, 勾配を使う 1 つの Optimizer"
                " を ML-Found が更新する場合だけ使えます (closure を使う "
                "Optimizer, 勾配を使わない Optimizer, optimizers, "
                "manual_optimization では使えません)"
            )
        # 今の更新に向けて勾配を貯めたバッチの数
        self.accumulated = 0

        # closure で使う学習時のバッチと, 最後に求めた勾配のノルムの入れ物を用意
        self.current_batch: Any = None
        self.last_grad_norm: Optional[Union[float, dict[str, float]]] = None

        # 学習のバッチがエポックの最後のものかどうかの印 (勾配の端数の更新用)
        self.last_batch_of_epoch = False

        # closure の中で step を呼んでいるかどうかの印を用意
        # closure は step を何度も呼ぶので, 予測を貯める処理などを飛ばすのに使用
        self.in_closure = False

        # 外から読める実験の状態と進み具合を書く仕組みの作成
        self.run_status = RunStatus(experiment.run_dir, self.logger)

        # 学習の後の処理の心拍 (fit を終えたときに作る) の入れ物を用意
        self.heartbeat: Optional[PostHeartbeat] = None

        # 外からの要求で学習を止めたかどうかの印を用意
        self.stopped_by_request = False

        # step の extra で返した, 平均せずに記録だけする値の入れ物を用意
        self.epoch_extra: dict[str, float] = {}

        # バッチのサンプル数を推定できなかったことを 1 回だけ警告するための印
        self.warned_batch_size = False

        # エポックの途中で止めたときに戻す, 学習を始めた時点の重みと,
        # 戻せずに途中までの重みが残ったかどうかの印を用意
        self.start_state: Optional[dict] = None
        self.epoch_start_step = 0
        self.partial_weights = False

        # チェックポイントに状態を保存する, 今の fit の loader の入れ物を用意
        self.loaders: dict[str, Any] = {}

    def build_no_param_parts(
        self, optimizer: Optional[optim.Optimizer],
        scheduler: Optional[optim.lr_scheduler.LRScheduler]
    ) -> None:
        """
        パラメータを持たないモデルのために, Optimizer と Scheduler を作らずに
        学習率の記録などの入れ物を用意する関数

        Parameters
        ----------
        optimizer: Optional[optim.Optimizer]
            使えないので None (自分で作った Optimizer は渡せない)
        scheduler: Optional[optim.lr_scheduler.LRScheduler]
            使えないので None (自分で作った Scheduler は渡せない)

        Returns
        ----------
        None
        """
        # 自分で作った Optimizer と Scheduler は, 更新するものが無いので不可
        if optimizer is not None or scheduler is not None:
            raise ValueError(
                "パラメータを持たないモデルは学習しないので, Trainer に "
                "optimizer と scheduler を渡さないでください"
            )

        # Optimizer と Scheduler は作らず, 学習率の列も無し
        self.optimizer = None
        self.scheduler = None
        self.scheduler_from_config = True
        self.pending_scheduler = False
        self.scheduler_interval = "epoch"
        self.lr_columns = []
        self.logger.info(
            "パラメータを持たないモデルなので, Optimizer と Scheduler を作り"
            "ません (config の optimizer, scheduler, optimizers は使いません)"
        )

        # 毎エポック同じ検証になるので, エポック数が 1 でない場合は警告
        epochs = self.trainer_cfg.epochs
        if epochs != 1:
            message = (
                f"パラメータを持たないモデルは学習しないので, trainer.epochs "
                f"({epochs}) の毎エポックで同じ検証をします. "
                "trainer.epochs: 1 にしてください"
            )
            self.logger.warning(message)
            warnings.warn(message, stacklevel=4)

    def build_single_parts(
        self, optimizer: Optional[optim.Optimizer],
        scheduler: Optional[optim.lr_scheduler.LRScheduler]
    ) -> None:
        """
        config の optimizer と scheduler から, 1 つの Optimizer と Scheduler を
        用意する関数

        Parameters
        ----------
        optimizer: Optional[optim.Optimizer]
            自分で作った Optimizer, None の場合は config から作成
        scheduler: Optional[optim.lr_scheduler.LRScheduler]
            自分で作った Scheduler, None の場合は config から作成

        Returns
        ----------
        None
        """
        # Optimizer が渡されていない場合は config から作成
        cfg = self.experiment.cfg
        if optimizer is None:
            optimizer = build_optimizer(cfg.optimizer, self.model.parameters())
        self.optimizer = optimizer

        # Scheduler が渡されていない場合は config から作成し, どちらかを保存
        # データの量で決まる引数 (OneCycleLR の steps_per_epoch など) が
        # 無い場合は, 学習用データが分かる fit の初めまで作成を保留
        self.scheduler_from_config = scheduler is None
        self.pending_scheduler = scheduler is None and needs_data(
            cfg.scheduler
        )
        if scheduler is None and not self.pending_scheduler:
            scheduler = build_scheduler(
                cfg.scheduler, self.optimizer, cfg.trainer.mode
            )
        elif (
            scheduler is not None and scheduler.optimizer is not self.optimizer
        ):
            raise ValueError(
                "scheduler は Trainer が使う optimizer で作ってください"
            )
        self.scheduler = scheduler

        # 学習率を更新する間隔と, 更新ごとの学習率の記録 (列は lr)
        self.scheduler_interval = cfg.scheduler.interval
        self.lr_columns = ["lr"]

    def build_named_parts(
        self, optimizer: Optional[optim.Optimizer],
        scheduler: Optional[optim.lr_scheduler.LRScheduler]
    ) -> None:
        """
        config の optimizers から, 名前付きの複数の Optimizer と Scheduler を
        用意する関数

        学習するパラメータを modules で Optimizer ごとに分け, それぞれの
        Optimizer と Scheduler を作って NamedOptimizers と NamedSchedulers に
        まとめる

        Parameters
        ----------
        optimizer: Optional[optim.Optimizer]
            使えないので None (自分で作った Optimizer は渡せない)
        scheduler: Optional[optim.lr_scheduler.LRScheduler]
            使えないので None

        Returns
        ----------
        None
        """
        # 自分で作った Optimizer と Scheduler は, 名前との対応が無いので不可
        if optimizer is not None or scheduler is not None:
            raise ValueError(
                "config の optimizers を使う場合は, Trainer に optimizer と "
                "scheduler を渡さないでください"
            )

        # 学習するパラメータを Optimizer ごとに分けて作成
        cfg = self.experiment.cfg
        modules = {
            str(name): [str(path) for path in entry.modules]
            for name, entry in cfg.optimizers.items()
        }
        self.named_params = split_parameters(self.model, modules)
        optimizers = {}
        for name, view in named_views(cfg):
            with named_errors(name):
                optimizers[name] = build_optimizer(
                    view.optimizer, self.named_params[name]
                )
        self.optimizer = NamedOptimizers(optimizers)

        # Scheduler は Optimizer ごとに作成し, データの量で決まる引数がある
        # ものは fit の初めまで保留
        self.scheduler_from_config = True
        schedulers: dict[str, Optional[optim.lr_scheduler.LRScheduler]] = {}
        self.pending_named: set[str] = set()
        intervals = set()
        for name, entry in cfg.optimizers.items():
            name = str(name)
            schedulers[name] = None
            if entry.scheduler.name is None and not (
                entry.scheduler.get("warmup_steps")
                or entry.scheduler.get("warmup_ratio")
            ):
                continue
            intervals.add(entry.scheduler.interval)
            if needs_data(entry.scheduler):
                self.pending_named.add(name)
                continue
            with named_errors(name):
                schedulers[name] = build_scheduler(
                    entry.scheduler, optimizers[name], cfg.trainer.mode
                )
        self.pending_scheduler = bool(self.pending_named)
        self.scheduler = NamedSchedulers(schedulers)

        # 学習率を更新する間隔 (全ての Scheduler で同じ) と, 更新ごとの学習率の
        # 記録の列 (lr_<名前>)
        self.scheduler_interval = intervals.pop() if intervals else "epoch"
        self.lr_columns = [f"lr_{name}" for name in optimizers]

    def scheduler_entries(
        self
    ) -> list[tuple[str, Optional[optim.lr_scheduler.LRScheduler], Any]]:
        """
        Scheduler を, 名前, Scheduler, scheduler の config の組で返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        entries: list[tuple[str, Optional[LRScheduler], Any]]
            1 つの Optimizer の場合は名前が空の 1 組, 名前付きの場合は
            Optimizer ごとの組, パラメータを持たないモデルの場合は空
        """
        cfg = self.experiment.cfg
        if self.no_params:
            entries = []
            return entries
        if not self.named:
            entries = [("", self.scheduler, cfg.scheduler)]
            return entries
        schedulers = (
            self.scheduler.schedulers
            if isinstance(self.scheduler, NamedSchedulers) else {}
        )
        entries = [
            (str(name), schedulers.get(str(name)), entry.scheduler)
            for name, entry in cfg.optimizers.items()
        ]

        return entries

    def step(self, batch: Any, phase: str) -> dict:
        """
        1 バッチ分の損失と指標を求める関数

        Parameters
        ----------
        batch: Any
            デバイスに移したバッチ
        phase: str
            学習時は train, 検証時は valid, 学習用データの評価し直し
            (trainer.eval_train) では train_eval

        Returns
        ----------
        outputs: dict
            loss は必須, その他の指標と batch_size は任意. 平均せずに
            記録だけする値は extra に辞書で入れる (検証でも返した場合は
            検証の値, エポックの最後の値を記録. 学習用データの評価し直しで
            返した値は記録しない)
        """
        # 関数が渡されていない場合は継承して上書きが必要
        if self.step_fn is None:
            raise NotImplementedError(
                "step_fn を渡すか step を上書きしてください"
            )

        # 渡された関数で計算, 検証では評価に使うモデルを使用
        model = self.model if phase == "train" else self.eval_model()
        outputs = self.step_fn(model, batch, phase)

        return outputs

    def autocast(self) -> contextlib.AbstractContextManager:
        """
        trainer.precision に合わせた自動の混合精度の範囲を作る関数

        Parameters
        ----------
        None

        Returns
        ----------
        context: contextlib.AbstractContextManager
            bf16 の場合は torch.autocast, fp32 の場合は何もしない範囲
        """
        # 学習のデバイスで, 設定の精度の範囲を作成
        context = autocast_context(self.trainer_cfg.precision, self.device)

        return context

    def add_callback(self, fn: Callable[[dict], Optional[bool]]) -> None:
        """
        エポックの終わりに呼ぶ関数を加える関数

        関数はエポックごとに, epoch (エポック), record (記録の写し),
        improved (ベストモデルを更新したか), best_epoch, best_value,
        early_stopping (Early Stopping で止めるか), judged (検証して判定した
        エポックか) の辞書を受け取る. 真の値
        (True など) を返すと, そのエポックで学習を止める (状態は stopped,
        予定の最後のエポックなら completed)

        Parameters
        ----------
        fn: Callable[[dict], Optional[bool]]
            エポックの結果を受け取る関数

        Returns
        ----------
        None
        """
        # 関数以外は, 学習の途中ではなくここでエラー
        if not callable(fn):
            raise TypeError(f"関数を渡してください: {fn!r}")
        self.callbacks.append(fn)

    def run_callbacks(
        self, epoch: int, record: dict, improved: bool, should_stop: bool,
        judged: bool = True
    ) -> bool:
        """
        登録した関数にエポックの結果を渡す関数

        Parameters
        ----------
        epoch: int
            終えたエポック
        record: dict
            そのエポックの記録
        improved: bool
            ベストモデルを更新したかどうか
        should_stop: bool
            Early Stopping で止めるかどうか
        judged: bool = True
            検証してベストモデルなどを判定したエポックかどうか

        Returns
        ----------
        stop: bool
            どれかの関数が真の値を返して止める要求をしたかどうか
        """
        # 関数ごとに写しを渡し, 記録を書き換えられないように用意
        stop = False
        for fn in self.callbacks:
            info = {
                "epoch": epoch, "record": dict(record),
                "improved": bool(improved),
                "best_epoch": self.checkpoint.best_epoch or None,
                "best_value": self.checkpoint.best_value,
                "early_stopping": bool(should_stop),
                "judged": bool(judged)
            }
            # numpy や torch の真偽値も, 真の値なら止める要求として扱う
            if fn(info):
                stop = True

        return stop

    def eval_model(self) -> nn.Module:
        """
        検証, ベストモデルの判定, final_model.pt に使うモデルを返す関数

        自分で持つ平均の重みで検証する場合などに継承して上書きする. 返す
        モデルは self.model と同じ名前の重み (state_dict のキー) を持つ必要が
        ある.
        step を上書きする場合は, 学習以外 (phase が valid か train_eval) で
        このモデルを使う

        Parameters
        ----------
        None

        Returns
        ----------
        model: nn.Module
            評価に使うモデル, 既定は学習するモデル (self.model) そのもの.
            trainer.ema.decay を使う場合は EMA の重みを持つモデル
        """
        # 既定は EMA のモデルか, 学習するモデルをそのまま使用
        if self.ema_model is not None:
            model = self.ema_model.module
        else:
            model = self.model

        return model

    def check_eval_model(self) -> None:
        """
        評価に使うモデルが学習するモデルと同じ名前と形の重みを持つかを
        確かめる関数

        ベストモデルの重みと学習後の重みを self.model に読み込むため, 学習を
        始める前に確認

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 学習するモデルそのものなら確認不要
        model = self.eval_model()
        if model is self.model:
            return

        # 重みの名前か形が違う場合は, 学習の後で読み込めないのでエラー
        expected = self.model.state_dict()
        actual = model.state_dict()
        if expected.keys() != actual.keys():
            missing = sorted(expected.keys() - actual.keys())[:5]
            unexpected = sorted(actual.keys() - expected.keys())[:5]
            raise ValueError(
                "eval_model が返すモデルの重みの名前が self.model と違います "
                f"(足りない: {missing}, 余分: {unexpected})"
            )
        for key, value in expected.items():
            other = actual[key]
            if (
                isinstance(value, torch.Tensor)
                and isinstance(other, torch.Tensor)
                and value.shape != other.shape
            ):
                raise ValueError(
                    f"eval_model が返すモデルの重み {key} の形が self.model と"
                    f"違います ({tuple(other.shape)} と {tuple(value.shape)})"
                )

    def on_epoch_start(self, phase: str) -> None:
        """
        エポックの学習または検証を始める前に呼ばれる関数

        Parameters
        ----------
        phase: str
            学習時は train, 検証時は valid, 学習用データの評価し直しでは
            train_eval

        Returns
        ----------
        None
        """
        # 何もしない, エポック全体の予測を集める場合は継承して入れ物を用意
        pass

    def on_epoch_end(self, phase: str) -> dict:
        """
        エポックの学習または検証を終えた後に呼ばれる関数

        Parameters
        ----------
        phase: str
            学習時は train, 検証時は valid, 学習用データの評価し直しでは
            train_eval

        Returns
        ----------
        metrics: dict
            エポック全体から求めた指標, 接頭辞を付けて記録される
        """
        # 何も返さない, AUC などバッチの平均で求められない指標は継承して計算
        metrics = {}

        return metrics

    def estimate_batch_size(self, batch: Any) -> int:
        """
        指標の平均の重みに使う, バッチのサンプル数を推定する関数

        Parameters
        ----------
        batch: Any
            テンソル, またはテンソルを含むリスト, タプル, 辞書など

        Returns
        ----------
        size: int
            最初に見つかったテンソルの 0 次元目, 見つからない場合は 1
        """
        size = find_batch_size(batch)
        if size is not None:
            return size

        # 1 として平均すると, 小さなバッチがサンプル数の割に重く効くので警告
        if not self.warned_batch_size:
            message = (
                "バッチのサンプル数を推定できないので, 1 として指標を平均します. "
                "step の戻り値に batch_size を入れてください"
            )
            self.logger.warning(message)
            warnings.warn(message, stacklevel=2)
            self.warned_batch_size = True
        size = 1

        return size

    def check_outputs(self, outputs: Any, phase: str) -> torch.Tensor:
        """
        step の戻り値が正しいことを確かめて損失を取り出す関数

        Parameters
        ----------
        outputs: Any
            step の戻り値
        phase: str
            train または valid

        Returns
        ----------
        loss: torch.Tensor
            1 バッチ分の損失
        """
        # 辞書で loss を含んでいることの確認
        if not isinstance(outputs, dict) or "loss" not in outputs:
            raise ValueError("step は loss を含む辞書を返してください")

        # 学習曲線の画像のファイル名に使うので / を含む名前を禁止し,
        # バッチごとの平均を求めるので指標は 1 つの値に制限
        # (平均せずに記録だけする値の extra は, 中の値を別に確認)
        check_extra(outputs.get("extra"))
        for key, value in outputs.items():
            if key == "extra":
                continue
            if "/" in key:
                raise ValueError(f"指標の名前に / は使えません: {key}")
            if isinstance(value, torch.Tensor) and value.numel() != 1:
                raise ValueError(
                    f"指標 {key} は 1 つの値にしてください: "
                    f"形状 {tuple(value.shape)}"
                )

        # 逆伝播できるように損失はテンソルに限定
        loss = outputs["loss"]
        if not isinstance(loss, torch.Tensor):
            raise ValueError("loss は torch.Tensor で返してください")

        # 損失が発散していないことの確認
        if not math.isfinite(to_float(loss)):
            raise ValueError(f"{phase} の loss が {to_float(loss)} になりました")

        return loss

    def clip_gradients(
        self, params: Optional[list[nn.Parameter]] = None
    ) -> Optional[float]:
        """
        勾配のノルムを求め, 上限が設定されていればクリップする関数

        Parameters
        ----------
        params: Optional[list[nn.Parameter]] = None
            対象のパラメータ, None の場合は Optimizer が更新する全てのもの

        Returns
        ----------
        grad_norm: Optional[float]
            クリップ前の勾配のノルム, クリップも記録もしない設定の場合は None
        """
        # クリップも記録もしない場合は計算なし
        max_norm = self.trainer_cfg.grad_clip_norm
        if max_norm is None and not self.trainer_cfg.log_grad_norm:
            grad_norm = None
            return grad_norm

        # 指定が無い場合は, Optimizer が更新するパラメータだけを対象に設定
        if params is None:
            params = [
                p for group in self.optimizer.param_groups
                for p in group["params"]
            ]

        # 上限が無い場合は勾配を変えずにノルムだけ計算
        # (torch 2.6 より前にも対応するため, パラメータごとのノルムから計算)
        if max_norm is None:
            norms = [
                torch.linalg.vector_norm(p.grad.detach())
                for p in params if p.grad is not None
            ]
            grad_norm = 0.0
            if len(norms) > 0:
                grad_norm = to_float(
                    torch.linalg.vector_norm(torch.stack(norms))
                )
            return grad_norm

        # 上限でクリップし, クリップ前のノルムを取得
        grad_norm = to_float(nn.utils.clip_grad_norm_(params, max_norm))

        return grad_norm

    def make_closure(self) -> Callable[[], torch.Tensor]:
        """
        保存した学習時のバッチで損失と勾配を計算し直す関数を作る関数

        Parameters
        ----------
        None

        Returns
        ----------
        closure: Callable[[], torch.Tensor]
            LBFGS などの Optimizer の step に渡す関数
        """
        def closure() -> torch.Tensor:
            """
            勾配を初期化して損失を計算し直し, 逆伝播とクリップを行う関数

            Parameters
            ----------
            None

            Returns
            ----------
            loss: torch.Tensor
                計算し直した損失
            """
            # 勾配を計算する範囲で損失を計算し直して逆伝播
            self.optimizer.zero_grad()
            self.in_closure = True
            try:
                with torch.enable_grad():
                    with self.autocast():
                        outputs = self.step(self.current_batch, "train")
                    loss = self.check_outputs(outputs, "train")
                    loss.backward()
            finally:
                self.in_closure = False

            # 勾配のノルムを求めてクリップ
            self.last_grad_norm = self.clip_gradients()

            return loss

        return closure

    def make_loss_closure(self) -> Callable[[], torch.Tensor]:
        """
        保存した学習時のバッチで, 勾配を使わずに損失だけを計算し直す関数を
        作る関数

        Parameters
        ----------
        None

        Returns
        ----------
        closure: Callable[[], torch.Tensor]
            勾配を使わない Optimizer の step に渡す関数
        """
        def closure() -> torch.Tensor:
            """
            勾配を計算せずに損失を計算し直す関数

            Parameters
            ----------
            None

            Returns
            ----------
            loss: torch.Tensor
                計算し直した損失
            """
            # 勾配を計算しない範囲で, 今のパラメータの損失を計算
            self.in_closure = True
            try:
                with torch.no_grad(), self.autocast():
                    outputs = self.step(self.current_batch, "train")
                    loss = self.check_outputs(outputs, "train")
            finally:
                self.in_closure = False

            return loss

        return closure

    def current_lrs(self) -> dict[str, float]:
        """
        今の学習率を, 記録の列の名前ごとに返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        lrs: dict[str, float]
            1 つの Optimizer は lr, 名前付きの Optimizer は lr_<名前> ごとの
            最初のパラメータグループの学習率. Optimizer が無い場合は空
        """
        if self.optimizer is None:
            lrs = {}
        elif isinstance(self.optimizer, NamedOptimizers):
            lrs = {
                f"lr_{name}": float(optimizer.param_groups[0]["lr"])
                for name, optimizer in self.optimizer.items()
            }
        else:
            lrs = {"lr": float(self.optimizer.param_groups[0]["lr"])}

        return lrs

    def expand_named_loss(self, outputs: Any) -> Any:
        """
        名前付きの Optimizer の場合に, step が返した名前ごとの損失を取り出し,
        記録する指標 (loss_<名前> と合計の loss) に展開する関数

        Parameters
        ----------
        outputs: Any
            step の戻り値

        Returns
        ----------
        outputs: Any
            loss を合計に置き換え, loss_<名前> を加えた戻り値. 名前付きの
            Optimizer を使わない場合はそのまま
        """
        # 1 つの Optimizer の場合と, 辞書でない戻り値はそのまま (後で確認)
        self.named_losses: Optional[dict[str, torch.Tensor]] = None
        if not self.named or not isinstance(outputs, dict):
            return outputs
        losses = outputs.get("loss")

        # 自分で更新する設定では 1 つの損失も可, それ以外は名前ごとの辞書
        if not isinstance(losses, dict):
            if self.trainer_cfg.manual_optimization:
                return outputs
            raise ValueError(
                "optimizers を使う場合は, step の loss を Optimizer の名前ごとの"
                f"損失の辞書にしてください ({list(self.optimizer)})"
            )
        if set(losses) != set(self.optimizer):
            raise ValueError(
                f"step の loss の名前 {sorted(losses)} が optimizers の名前 "
                f"{sorted(self.optimizer)} と違います"
            )

        # 名前ごとの損失と合計を, 他の指標と重ならない名前で記録
        expanded = dict(outputs)
        for name in self.optimizer:
            loss = losses[name]
            if not isinstance(loss, torch.Tensor):
                raise ValueError(
                    f"loss の {name} は torch.Tensor で返してください"
                )
            key = f"loss_{name}"
            if key in outputs:
                raise ValueError(
                    f"optimizers を使う場合は step で {key} を返さないでください"
                )
            expanded[key] = loss.detach()
        expanded["loss"] = sum(losses[name] for name in self.optimizer)
        self.named_losses = {name: losses[name] for name in self.optimizer}

        return expanded

    def update_named(self) -> None:
        """
        名前付きの Optimizer で, 損失ごとに受け持つパラメータの勾配を求めて
        更新する関数

        全ての損失の勾配を, 更新の前の同じパラメータで求めてから, config に
        書いた順に更新する (同時の更新). 勾配は受け持つパラメータだけに求める
        ので, 例えば生成器の損失の勾配は識別器のパラメータに入らない

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 名前ごとの損失 (expand_named_loss で取り出したもの)
        losses = self.named_losses
        if losses is None:
            raise ValueError("optimizers の損失が取り出されていません")

        # 損失ごとに, 受け持つパラメータだけの勾配を計算
        # 計算グラフは, 最後の損失の勾配を求めるまで保持
        names = list(self.optimizer)
        grads = {}
        for index, name in enumerate(names):
            loss = losses[name]
            if not loss.requires_grad:
                raise ValueError(
                    f"loss の {name} が勾配を持ちません (パラメータから計算した"
                    "損失を返してください)"
                )
            # 学習の途中で凍結したパラメータは, 勾配を求めずに対象外
            params = [p for p in self.named_params[name] if p.requires_grad]
            values = torch.autograd.grad(
                loss, params, retain_graph=index < len(names) - 1,
                allow_unused=True
            ) if params else ()
            grads[name] = dict(zip(map(id, params), values))

        # 勾配を書き込み, Optimizer ごとにクリップして更新
        norms = {}
        for name in names:
            for parameter in self.named_params[name]:
                parameter.grad = grads[name].get(id(parameter))
            norm = self.clip_gradients(self.named_params[name])
            if norm is not None:
                norms[name] = norm
        for name in names:
            self.optimizer[name].step()
        self.last_grad_norm = norms or None

    def update_parameters(self, loss: torch.Tensor) -> None:
        """
        損失から勾配を求めてパラメータを更新する関数

        学習のバッチごとに呼ぶ. 勾配を貯める設定 (accumulate_grad_batches が
        2 以上) では, 貯める数か, エポックの最後のバッチに達した回だけ更新し,
        更新した回だけ global_step が増える

        Parameters
        ----------
        loss: torch.Tensor
            1 バッチ分の損失

        Returns
        ----------
        None
        """
        # 自分で更新する設定の場合は, 逆伝播と更新を行わずに回数のカウントだけ
        if self.trainer_cfg.manual_optimization:
            self.last_grad_norm = None

        # 勾配を使わない Optimizer は, 損失だけを求める関数を渡して更新
        # (パラメータを直接書き換えられるように, 勾配の記録を止めて実行)
        elif not self.use_gradients:
            self.last_grad_norm = None
            with torch.no_grad():
                self.optimizer.step(self.make_loss_closure())

        # 名前付きの Optimizer は, 損失ごとに受け持つパラメータの勾配を求めて更新
        elif self.named:
            self.update_named()

        # closure が必要な Optimizer は損失を計算し直す関数を渡して更新
        elif self.use_closure:
            self.optimizer.step(self.make_closure())

        # 勾配を貯める場合は, 決めた数か, エポックの最後のバッチまで貯めて更新
        # (損失を貯める数で割り, 貯めた勾配が損失の平均の勾配になるように)
        elif self.accumulate_grad_batches > 1:
            if self.accumulated == 0:
                self.optimizer.zero_grad()
            (loss / self.accumulate_grad_batches).backward()
            self.accumulated += 1
            if (
                self.accumulated < self.accumulate_grad_batches
                and not self.last_batch_of_epoch
            ):
                self.last_grad_norm = None
                return
            self.apply_accumulated()
            return

        # 勾配法では勾配を初期化して逆伝播し, クリップしてから更新
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.last_grad_norm = self.clip_gradients()
            self.optimizer.step()

        # EMA, 更新回数, Scheduler の進行
        self.after_update()

    def apply_accumulated(self) -> None:
        """
        貯めた勾配でパラメータを更新する関数

        貯めたバッチの数が accumulate_grad_batches より少ない (エポックの最後の
        端数) 場合は, 勾配を貯めた数の平均に直してから更新する

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 端数は, 損失を割った数と貯めた数の比で勾配を補正
        count = self.accumulated
        self.accumulated = 0
        if count < self.accumulate_grad_batches:
            scale = self.accumulate_grad_batches / count
            for group in self.optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)

        # クリップしてから更新し, 更新の後の処理
        self.last_grad_norm = self.clip_gradients()
        self.optimizer.step()
        self.after_update()

    def after_update(self) -> None:
        """
        パラメータを 1 回更新した後に, EMA, 更新回数, Scheduler を進める関数

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # EMA を使う場合は, 更新した重みで指数移動平均を更新
        if self.ema_model is not None:
            self.ema_model.update_parameters(self.model)

        # パラメータの更新回数のカウント
        self.global_step += 1

        # step ごとに学習率を更新する場合は, この更新に使った学習率を記録して
        # から更新 (エポックの記録の lr はエポックを始めた時点の値だけ)
        if self.scheduler_interval == "step":
            if self.scheduler is not None:
                self.lr_steps.append(
                    self.global_step, self.epoch,
                    list(self.current_lrs().values())
                )
            step_scheduler(self.scheduler)

    def run_epoch(
        self, loader: Iterable, train: bool, phase: Optional[str] = None
    ) -> dict[str, float]:
        """
        1 エポック分の学習または検証を行う関数

        Parameters
        ----------
        loader: Iterable
            バッチを順に返すもの, DataLoader や [data] のようなリスト
        train: bool
            学習する場合は True, 検証する場合は False
        phase: Optional[str] = None
            フェーズの名前, None の場合は train か valid. 学習用データの
            評価し直しでは train_eval

        Returns
        ----------
        metrics: dict[str, float]
            フェーズの名前 (train/ など) を付けた指標ごとの平均
        """
        # 学習モードと推論モードの切り替えと, 指標の集計の準備
        # 検証では評価に使うモデルも推論モード
        self.model.train(train)
        if not train:
            self.eval_model().eval()
        if phase is None:
            phase = "train" if train else "valid"
        tracker = MetricTracker()
        self.on_epoch_start(phase)

        # 勾配を貯める数は, 学習のエポックごとに数え直し
        if train:
            self.accumulated = 0

        # 学習時 (勾配を使う Optimizer か, 自分で更新する設定の場合) と,
        # 検証時にも勾配を使う設定の場合だけ勾配を計算
        if train:
            grad_enabled = (
                self.use_gradients or self.trainer_cfg.manual_optimization
            )
        else:
            grad_enabled = self.trainer_cfg.valid_grad

        # 進み具合を外に知らせるため, 分かる場合は 1 エポックのバッチの数を取得
        total_batches: Optional[int] = None
        try:
            total_batches = len(loader)
        except (TypeError, NotImplementedError):
            pass
        # 1 エポックが短い学習の邪魔にならないように, フェーズの始めも間隔を確保
        self.run_status.progress(
            phase, self.epoch, self.end_epoch, 0, total_batches
        )

        # バッチごとに計算, 設定に応じてエポックの中の進捗バーも表示
        show = (
            self.trainer_cfg.progress_bar
            and self.trainer_cfg.progress_bar_batches
        )
        n_batches = 0
        with tqdm(
            loader, desc=phase, leave=False, disable=not show
        ) as batches:
            for batch in batches:
                batch = move_to_device(batch, self.device)
                n_batches += 1

                # 決めた範囲だけ勾配を計算 (bf16 では自動の混合精度)
                with torch.set_grad_enabled(grad_enabled), self.autocast():
                    outputs = self.expand_named_loss(self.step(batch, phase))

                # 戻り値を確かめて, 学習時は closure 用にバッチを保存して更新
                # 勾配を貯める場合は, エポックの最後のバッチで端数も更新
                is_last = total_batches is not None and (
                    n_batches >= total_batches
                )
                loss = self.check_outputs(outputs, phase)
                if train:
                    self.current_batch = batch
                    self.last_batch_of_epoch = is_last
                    self.update_parameters(loss)

                # サンプル数で重み付けして指標の加算
                metrics = dict(outputs)
                # 学習用データの評価し直しで返した値は, 学習中の値を残すため
                # 記録しない
                extra = metrics.pop("extra", None) or {}
                if phase != "train_eval":
                    self.epoch_extra.update(
                        {key: to_float(value) for key, value in extra.items()}
                    )
                n = metrics.pop("batch_size", None)
                if n is None:
                    n = self.estimate_batch_size(batch)
                if train and self.trainer_cfg.log_grad_norm:
                    self.add_grad_norm(metrics)
                tracker.update(metrics, int(n))

                # 一定の間隔で進み具合を書き出し
                self.run_status.progress(
                    phase, self.epoch, self.end_epoch, n_batches,
                    total_batches
                )

                # エポックの途中で止める要求 (STOP に now) の確認
                # 最後のバッチの後は, エポックの区切りで止める処理に任せる
                if not is_last and self.run_status.stop_now_requested():
                    raise StopWithinEpoch()

        # バッチの数が分からない loader で貯めた勾配の端数を, エポックの
        # 終わりに更新 (勾配のノルムは最後のバッチの重みで記録)
        if train and self.accumulated > 0:
            self.apply_accumulated()
            if self.trainer_cfg.log_grad_norm:
                grad_metrics: dict = {}
                self.add_grad_norm(grad_metrics)
                tracker.update(grad_metrics, int(n))
        self.last_batch_of_epoch = False

        # フェーズの終わりの進み具合の書き出し (間隔の中なら次の機会に)
        self.run_status.progress(
            phase, self.epoch, self.end_epoch, n_batches,
            total_batches
        )

        # 学習時のバッチを検証の間まで残さないように解放
        self.current_batch = None

        # バッチが 1 つも無い場合は平均を求められないのでエラー
        if n_batches == 0:
            raise ValueError(f"{phase} の loader が空です")

        # エポックの終わりの処理 (継承した場合だけ. FID など高価な評価が
        # 入りうるので, 間隔に関係なく段階を書き出し, 終えたら戻す)
        overridden = type(self).on_epoch_end is not Trainer.on_epoch_end
        if overridden:
            self.run_status.progress(
                phase, self.epoch, self.end_epoch, n_batches,
                total_batches, force=True, stage="epoch_end"
            )
        extra_metrics = self.on_epoch_end(phase)
        if overridden:
            self.run_status.progress(
                phase, self.epoch, self.end_epoch, n_batches,
                total_batches, force=True
            )

        # バッチの平均とエポック全体から求めた指標を集約
        epoch_metrics = tracker.compute()
        for key, value in extra_metrics.items():
            if "/" in key:
                raise ValueError(f"指標の名前に / は使えません: {key}")
            if key in epoch_metrics:
                raise ValueError(
                    f"on_epoch_end の指標 {key} が step の指標と重複しています"
                )
            epoch_metrics[key] = to_float(value)

        # 接頭辞を付けて返却
        metrics = {
            f"{phase}/{key}": value for key, value in epoch_metrics.items()
        }

        return metrics

    def add_grad_norm(self, metrics: dict) -> None:
        """
        最後に求めた勾配のノルムを指標に加える関数

        Parameters
        ----------
        metrics: dict
            1 バッチ分の指標, grad_norm を書き加える

        Returns
        ----------
        None
        """
        # step の指標と名前が重なる場合はエラー
        if "grad_norm" in metrics:
            raise ValueError(
                "log_grad_norm を使う場合は step で grad_norm を返さないでください"
            )

        # 勾配を計算しなかった場合 (自分で更新する設定など) は追加なし
        # 名前付きの Optimizer は Optimizer ごと (grad_norm_<名前>)
        if isinstance(self.last_grad_norm, dict):
            for name, value in self.last_grad_norm.items():
                key = f"grad_norm_{name}"
                if key in metrics:
                    raise ValueError(
                        f"log_grad_norm を使う場合は step で {key} を返さないで"
                        "ください"
                    )
                metrics[key] = value
        elif self.last_grad_norm is not None:
            metrics["grad_norm"] = self.last_grad_norm

    def run_valid_epoch(
        self, loader: Iterable, phase: str = "valid"
    ) -> dict[str, float]:
        """
        設定に応じて乱数を固定して検証を 1 エポック行う関数

        Parameters
        ----------
        loader: Iterable
            検証用のバッチを返すもの
        phase: str = "valid"
            フェーズの名前, 学習用データの評価し直しでは train_eval

        Returns
        ----------
        metrics: dict[str, float]
            フェーズの名前 (valid/ など) を付けた指標ごとの平均
        """
        # 乱数を固定せず, 学習の乱数にも影響しない場合はそのまま検証
        # 学習用データの評価し直しは, 並べ替える DataLoader などが学習の
        # 乱数を使うので, 固定しない場合も前後で状態を復帰
        seed = self.trainer_cfg.valid_seed
        if seed is None and phase == "valid":
            metrics = self.run_epoch(loader, train=False, phase=phase)
            return metrics

        # 学習の乱数の状態を保存してから固定し, 検証の後に元の状態へ復帰
        # 学習用データの評価し直しでは, DataLoader 自身の乱数生成器も復帰
        state = get_rng_state()
        generators = (
            loader_generators(loader) if phase == "train_eval" else []
        )
        generator_states = [
            generator.get_state() for generator in generators
        ]
        if seed is not None:
            set_seed(seed)
        try:
            metrics = self.run_epoch(loader, train=False, phase=phase)
        finally:
            set_rng_state(state)
            for generator, generator_state in zip(
                generators, generator_states
            ):
                generator.set_state(generator_state)

        return metrics

    def get_monitor_value(self, record: dict, monitor: str) -> float:
        """
        判定するエポックの記録から監視する指標の値を取得する関数

        Parameters
        ----------
        record: dict
            そのエポックの記録
        monitor: str
            監視する指標の名前

        Returns
        ----------
        value: float
            監視する指標の値, 無い場合と nan や inf の場合はエラー
        """
        # 監視する指標があり, 有限の値であることの確認
        if monitor not in record:
            raise ValueError(
                f"監視する指標 {monitor} がありません: "
                f"{list(record.keys())}"
            )
        value = record[monitor]
        if not math.isfinite(value):
            raise ValueError(
                f"監視する指標 {monitor} が {value} になりました"
            )

        return value

    def get_tiebreak(self, record: dict) -> Optional[float]:
        """
        監視する指標が同じ値の場合に比べる指標の値を取得する関数

        Parameters
        ----------
        record: dict
            今回のエポックの記録

        Returns
        ----------
        tiebreak: Optional[float]
            比べる指標の値, 設定していない場合は None
        """
        # 設定していない場合は比べない
        name = self.trainer_cfg.monitor_tiebreak
        if name is None:
            tiebreak = None
            return tiebreak

        # 指標があり, 有限の値であることの確認
        if name not in record:
            raise ValueError(
                f"同じ値の場合に比べる指標 {name} がありません: "
                f"{list(record.keys())}"
            )
        tiebreak = record[name]
        if not math.isfinite(tiebreak):
            raise ValueError(
                f"同じ値の場合に比べる指標 {name} が {tiebreak} になりました"
            )

        return tiebreak

    def get_monitor(self, has_valid: bool) -> str:
        """
        ベストモデルの判定に使う指標の名前を決める関数

        Parameters
        ----------
        has_valid: bool
            検証用のデータがあるかどうか

        Returns
        ----------
        monitor: str
            指標の名前
        """
        # 指定されている場合はそのまま使い, 無い場合は検証用のデータの有無で決定
        if self.trainer_cfg.monitor is not None:
            monitor = self.trainer_cfg.monitor
        elif has_valid:
            monitor = "valid/loss"
        else:
            monitor = "train/loss"

        return monitor

    def is_judged_epoch(self, epoch: int) -> bool:
        """
        検証し, ベストモデルなどを判定するエポックかを返す関数

        エポックの番号が trainer.valid_every_n_epochs で割り切れるエポックと,
        予定の最後のエポック. 番号は通しなので, 再開や fit の呼び直しでも
        同じエポックを判定する

        Parameters
        ----------
        epoch: int
            エポックの番号 (1 から)

        Returns
        ----------
        judged: bool
            判定するエポックなら True
        """
        # 間隔で割り切れるエポックと予定の最後のエポック
        every = self.trainer_cfg.valid_every_n_epochs
        judged = epoch % every == 0 or epoch == self.end_epoch

        return judged

    def state_dict(self, epoch: int, monitor: str) -> dict:
        """
        チェックポイントとして保存する内容をまとめる関数

        Parameters
        ----------
        epoch: int
            今回のエポック
        monitor: str
            ベストモデルの判定に使う指標の名前

        Returns
        ----------
        state: dict
            モデル, Optimizer, Scheduler, 記録, 乱数などの状態
        """
        # 学習を再開するのに必要な状態を全て集約
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "monitor": monitor,
            "mode": self.trainer_cfg.mode,
            "monitor_tiebreak": self.trainer_cfg.monitor_tiebreak,
            "monitor_tiebreak_mode": self.trainer_cfg.monitor_tiebreak_mode,
            "select_best": self.trainer_cfg.select_best,
            "valid_every_n_epochs": self.trainer_cfg.valid_every_n_epochs,
            "model": self.model.state_dict(),
            "optimizer": (
                None if self.optimizer is None
                else self.optimizer.state_dict()
            ),
            "scheduler": (
                None if self.scheduler is None
                else self.scheduler.state_dict()
            ),
            "scheduler_classes": (
                {
                    name: scheduler_classes(scheduler)
                    for name, scheduler in self.scheduler.items()
                } if isinstance(self.scheduler, NamedSchedulers)
                else scheduler_classes(self.scheduler)
            ),
            "history": self.history.state_dict(),
            "lr_steps": self.lr_steps.state_dict(),
            "early_stopping": (
                None if self.early_stopping is None
                else self.early_stopping.state_dict()
            ),
            "checkpoint": self.checkpoint.state_dict(),
            "separate_best_model": self.checkpoint.separate_best_model,
            "rng": get_rng_state(),
            "loaders": self.loader_states(),
            "ema": self.ema_state(),
            "extra": self.extra_state_dict()
        }

        # 学習するパラメータだけを残す設定では, 学習しないパラメータを除外
        if self.trainer_cfg.checkpoint.trainable_only:
            state = self.omit_frozen(state)

        return state

    def omit_frozen(self, state: dict) -> dict:
        """
        チェックポイントの内容から学習しないパラメータの重みを除く関数

        モデル, ベストモデル, EMA の重みから除き, 除いた名前と形を
        trainable_only に残す (再開では今のモデルの値で補う)

        Parameters
        ----------
        state: dict
            チェックポイントとして保存する内容

        Returns
        ----------
        light: dict
            学習しないパラメータを除いた写し (他の値は共有)
        """
        # 今のモデルの学習しないパラメータを, 重みの辞書ごとに除外
        frozen = frozen_parameters(self.model)
        light = dict(state)
        light["model"] = omit_keys(state["model"], frozen)
        if isinstance(state.get("checkpoint"), dict):
            light["checkpoint"] = {
                **state["checkpoint"],
                "best_model": omit_keys(
                    state["checkpoint"].get("best_model"), frozen
                )
            }
        if isinstance(state.get("ema"), dict):
            light["ema"] = {
                **state["ema"],
                "state": omit_keys(state["ema"]["state"], frozen, "module.")
            }
        light["trainable_only"] = {"frozen": frozen}

        return light

    def complete_frozen(self, state: dict) -> None:
        """
        学習するパラメータだけのチェックポイントの重みを, 今のモデルの学習
        しないパラメータの値で補う関数

        Parameters
        ----------
        state: dict
            再開元のチェックポイントの内容, モデル, ベストモデル, EMA の重みを
            書き換える

        Returns
        ----------
        None
        """
        # 学習しないパラメータを除いていないチェックポイントはそのまま
        meta = state.get("trainable_only")
        if not isinstance(meta, dict):
            return
        frozen = meta.get("frozen") or {}

        # 再開元では凍結していて今は学習する重みは, 今のモデルの値から始まる
        # ので通知
        trainable = {
            name for name, parameter in self.model.named_parameters(
                remove_duplicate=False
            ) if parameter.requires_grad
        }
        unfrozen = sorted(name for name in frozen if name in trainable)
        if unfrozen:
            self.logger.warning(
                "再開元では凍結していて今は学習する重みを, 今のモデルの値で"
                f"補います: {listed(unfrozen)}"
            )

        # モデルとベストモデルは今のモデルから, EMA は今の EMA から補完
        reference = self.model.state_dict()
        state["model"] = complete_weights(
            state["model"], reference, frozen
        )
        checkpoint_state = state.get("checkpoint")
        if (
            isinstance(checkpoint_state, dict)
            and checkpoint_state.get("best_model") is not None
        ):
            checkpoint_state["best_model"] = complete_weights(
                checkpoint_state["best_model"], reference, frozen,
                label="checkpoint.best_model", copy=True
            )
        ema = state.get("ema")
        if isinstance(ema, dict) and self.ema_model is not None:
            ema["state"] = complete_weights(
                ema["state"], self.ema_model.state_dict(), frozen,
                prefix="module.", label="ema"
            )

    def loader_states(self) -> dict:
        """
        状態を保存できる loader の状態を集める関数

        Parameters
        ----------
        None

        Returns
        ----------
        states: dict
            loader の役割 (train か valid) ごとの checkpoint_state の値
        """
        # 状態を持つ loader だけを対象
        states = {
            name: loader.checkpoint_state()
            for name, loader in self.loaders.items()
            if has_loader_state(loader)
        }

        return states

    def load_loader_states(
        self, states: Optional[dict], resuming: bool
    ) -> None:
        """
        保存した状態を loader に戻す関数

        Parameters
        ----------
        states: Optional[dict]
            loader_states で集めた状態, 古いチェックポイントでは None
        resuming: bool
            resume_from での再開の場合は True (戻せない状態を警告として
            表示. どちらも train.log には残す)

        Returns
        ----------
        None
        """
        # 状態を持つ loader に, 同じ役割の状態があれば復元
        # (loader が返した, 戻せなかった状態の警告の文も集める)
        states = states or {}
        messages = []
        for name, loader in self.loaders.items():
            if not has_loader_state(loader):
                continue
            if name in states:
                returned = loader.load_checkpoint_state(states[name])
                messages += [
                    f"{name} の loader: {message}"
                    for message in returned or []
                ]
            else:
                messages.append(
                    f"再開元に {name} の loader の状態 (環境の状態と乱数など) "
                    "が無いので戻せません. 再開した学習は通しで学習した場合と"
                    "一致しません"
                )

        # 保存した状態を受け取る loader が無い場合も通知
        for name in states:
            if not has_loader_state(self.loaders.get(name)):
                messages.append(
                    f"再開元の {name} の loader の状態 (環境の状態と乱数など) "
                    "を受け取る loader が無いので戻せません"
                )

        # train.log に残し, 再開では警告も表示 (途中で止めたエポックの
        # 捨てでは同じ fit の状態なので, 表示しない)
        for message in messages:
            self.logger.warning(message)
            if resuming:
                warnings.warn(message, stacklevel=2)

    def ema_state(self) -> Optional[dict]:
        """
        チェックポイントに入れる EMA の状態を取得する関数

        Parameters
        ----------
        None

        Returns
        ----------
        state: Optional[dict]
            減衰率, use_buffers, 重みと更新回数. EMA を使わない場合は None
        """
        # 今の設定と EMA の重みをまとめて取得
        ema_cfg = self.trainer_cfg.ema
        state = ema_state_dict(
            self.ema_model, ema_cfg.decay, ema_cfg.use_buffers
        )

        return state

    def resume_ema(self, state: Optional[dict]) -> None:
        """
        再開元のチェックポイントから EMA の状態を戻す関数

        再開元と設定が違う場合は警告し, 再開元に状態が無ければ再開した重み
        から EMA を始める

        Parameters
        ----------
        state: Optional[dict]
            再開元の ema, EMA を使わなかった実験と古いチェックポイントでは
            None

        Returns
        ----------
        None
        """
        # 再開元と今回の EMA の使い方に応じて復元し, 違いがあれば通知の文を用意
        message = None
        ema_cfg = self.trainer_cfg.ema
        if self.ema_model is None and state is not None:
            message = (
                "再開元は EMA を使っていますが, 今回は trainer.ema.decay が "
                "null なので EMA の状態を使いません. 再開元のベストモデルは "
                "再開元で検証したモデル (eval_model) の重みと値のままです"
            )
        elif self.ema_model is not None and state is None:
            reset_ema_model(self.ema_model, self.model)
            message = (
                "再開元に EMA の状態が無いので, 再開した重みから EMA を始めます."
                " 再開元のベストモデルは再開元で検証したモデル (eval_model) の"
                "重みと値のままです"
            )
        elif self.ema_model is not None:
            self.ema_model.load_state_dict(state["state"])
            old = (state.get("decay"), state.get("use_buffers"))
            new = (ema_cfg.decay, ema_cfg.use_buffers)
            if old != new:
                message = (
                    f"再開元と EMA の設定が違います: 再開元は decay={old[0]}, "
                    f"use_buffers={old[1]}, 今回は decay={new[0]}, "
                    f"use_buffers={new[1]}. 戻した EMA の重みから今回の設定で"
                    "続けます"
                )

        # 違いがあれば train.log にも残して警告
        if message is not None:
            self.logger.warning(message)
            warnings.warn(message, stacklevel=3)

    def extra_state_dict(self) -> dict:
        """
        継承したクラスがチェックポイントに加える状態を取得する関数

        Parameters
        ----------
        None

        Returns
        ----------
        state: dict
            加える状態, 2 つ目の Optimizer や EMA の重みなどは継承して追加
        """
        # 加える状態の無い空の辞書
        state = {}

        return state

    def load_extra_state_dict(self, state: dict) -> None:
        """
        extra_state_dict で加えた状態をチェックポイントから戻す関数

        Parameters
        ----------
        state: dict
            extra_state_dict で取得した状態, 古いチェックポイントでは空

        Returns
        ----------
        None
        """
        # 何もしない, extra_state_dict を上書きした場合は合わせて上書き
        pass

    def resume(self, path: str, monitor: str) -> None:
        """
        チェックポイントから学習を再開できる状態に戻す関数

        Parameters
        ----------
        path: str
            チェックポイントのパス, 相対パスの場合はプロジェクトのディレクトリから辿る
        monitor: str
            ベストモデルの判定に使う指標の名前

        Returns
        ----------
        None
        """
        # 出力先と同じく, 相対パスはプロジェクトのディレクトリを基準に解決
        checkpoint_path = Path(path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.experiment.project_dir / checkpoint_path
        path = str(checkpoint_path)

        # チェックポイントの読み込み
        state = load_checkpoint(path, self.device)
        if state.get("weights_only"):
            raise ValueError(
                f"{path} は重みだけのチェックポイント (best_weights_only) なので"
                "再開には使えません. last.pt を指定してください"
            )

        # 学習するパラメータだけのチェックポイントは, 今のモデルで補完
        self.complete_frozen(state)

        # 監視する指標や向きが変わると最良値を比べられないので同じものに制限
        if (state["monitor"], state["mode"]) != (
            monitor, self.trainer_cfg.mode
        ):
            raise ValueError(
                f"再開元と監視する指標が違います: 再開元は {state['monitor']} "
                f"({state['mode']}), 今回は {monitor} "
                f"({self.trainer_cfg.mode})"
            )

        # 同じ値の場合に比べる指標と向きも同じものに制限
        # この設定を入れる前のチェックポイントには無いので比較しない
        old_tiebreak = (
            state.get("monitor_tiebreak"),
            state.get("monitor_tiebreak_mode", "min")
        )
        new_tiebreak = (
            self.trainer_cfg.monitor_tiebreak,
            self.trainer_cfg.monitor_tiebreak_mode
        )
        if (
            "monitor_tiebreak" in state and old_tiebreak != new_tiebreak
            and self.trainer_cfg.select_best
        ):
            raise ValueError(
                "再開元と同じ値の場合に比べる指標が違います: 再開元は "
                f"{old_tiebreak[0]} ({old_tiebreak[1]}), 今回は "
                f"{new_tiebreak[0]} ({new_tiebreak[1]})"
            )

        # 検証の間隔が変わると, 待つ回数が表すエポック数が変わるので警告
        # (この設定を入れる前のチェックポイントは毎エポック検証)
        old_every = state.get("valid_every_n_epochs", 1)
        new_every = self.trainer_cfg.valid_every_n_epochs
        if old_every != new_every:
            message = (
                f"再開元と検証の間隔が違います: 再開元は {old_every}, "
                f"今回は {new_every}. Early Stopping と ReduceLROnPlateau の"
                "待つ回数は判定の回数のまま続けます"
            )
            warnings.warn(message, stacklevel=2)
            self.logger.warning(message)

        # パラメータを持たないモデルの学習 (Optimizer の状態が無いもの) かは,
        # 再開元と同じものに制限
        saved_no_params = state.get("optimizer") is None
        if saved_no_params != self.no_params:
            raise ValueError(
                "再開元と今回で, パラメータを持たないモデルかどうかが違います"
                f" (再開元: {'持たない' if saved_no_params else '持つ'}, "
                f"今回: {'持たない' if self.no_params else '持つ'})"
            )

        # モデル, Optimizer, Scheduler, 記録の復元
        # Scheduler が Optimizer に書き込む値 (OneCycleLR の max_lr など) も
        # config と比べるため, 読み込む前の値を保持
        # (パラメータを持たないモデルは Optimizer と Scheduler が無し)
        self.model.load_state_dict(state["model"])
        if self.named:
            self.resume_named(state)
        elif not self.no_params:
            self.resume_single(state)
        self.history.load_state_dict(state["history"])
        self.lr_steps = LrSteps(self.lr_columns)
        if state.get("lr_steps") is not None:
            self.lr_steps.load_state_dict(state["lr_steps"])

        # Early Stopping の状態の復元
        if (
            self.early_stopping is not None
            and state["early_stopping"] is not None
        ):
            self.early_stopping.load_state_dict(state["early_stopping"])

        # チェックポイントに含まれるベストモデルの重みと最良値の復元
        # 重みを省いた場合は, 今のモデルか再開元の best.pt からの復元
        # ベストモデルを選ばない設定では, 再開元のベストモデルの状態を捨てる
        self.checkpoint.load_state_dict(state["checkpoint"])
        self.warn_select_best_change(state)
        if not self.trainer_cfg.select_best:
            self.checkpoint.best_value = None
            self.checkpoint.best_epoch = 0
            self.checkpoint.best_tiebreak = None
            self.checkpoint.best_model_state = None
        elif self.checkpoint.best_model_state is None:
            self.restore_best_weights(state, checkpoint_path)

        # EMA と, 継承したクラスが加えた状態の復元
        self.resume_ema(state.get("ema"))
        self.load_extra_state_dict(state.get("extra", {}))

        # 続きの乱数を使えるように乱数の状態の復元
        # 戻せない外部ライブラリの乱数は train.log にも記録
        for message in set_rng_state(state["rng"]):
            self.logger.warning(message)
            warnings.warn(message, stacklevel=2)

        # 環境の状態と乱数など, loader の状態の復元
        self.load_loader_states(state.get("loaders"), resuming=True)

        # 新しい保存先にも, 再開元の best.pt と読み込んだ状態を保存
        run_dir = self.experiment.run_dir
        best_path = checkpoint_path.with_name("best.pt")
        if (
            best_path.exists() and self.trainer_cfg.checkpoint.save_best
            and self.trainer_cfg.select_best
        ):
            self.copy_best(best_path, run_dir / "best.pt")
        if self.trainer_cfg.checkpoint.save_last:
            # 戻したベストモデルの重みを入れ直し, 今回の設定に合わせて保存
            state["checkpoint"] = self.checkpoint.state_dict()
            if not self.trainer_cfg.checkpoint.last_with_best:
                state = without_best_model(state)
            if self.trainer_cfg.checkpoint.trainable_only:
                state = self.omit_frozen(state)
            save_checkpoint(state, run_dir / "last.pt")

        # 続きのエポックとパラメータの更新回数から開始
        self.start_epoch = state["epoch"] + 1
        self.epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.logger.info(f"再開元: {path} (エポック {state['epoch']})")

    def resume_single(self, state: dict) -> None:
        """
        チェックポイントから, 1 つの Optimizer と Scheduler の状態を戻す関数

        Parameters
        ----------
        state: dict
            チェックポイントの内容

        Returns
        ----------
        None
        """
        # 名前付きの Optimizer で保存した状態は読み込み不可
        if "optimizers" in (state.get("optimizer") or {}):
            raise ValueError(
                "再開元は optimizers を使った学習なので, optimizer の学習の"
                "状態として読み込めません"
            )
        fresh_groups = scheduler_group_values(self.optimizer)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state["scheduler"] is not None:
            # Scheduler の組み立て (warmup の有無など) が再開元と違う場合は,
            # 状態を読み込めないので, config の値で作り直すかエラー
            fresh_state = self.scheduler.state_dict()
            old, new = saved_scheduler_classes(state), scheduler_classes(
                self.scheduler
            )
            if not same_classes(old, new):
                old_text = " + ".join(old) if old else "warmup なし"
                change = f"{old_text} から {' + '.join(new)}"
                if self.trainer_cfg.resume_scheduler != "config":
                    raise ValueError(
                        f"Scheduler の組み立てが再開元と違う ({change}) ので, "
                        "再開元の状態を読み込めません (warmup の有無を変えた"
                        "場合など). config の値で作り直す場合は "
                        "trainer.resume_scheduler: config にしてください"
                    )
                self.replay_scheduler(state["scheduler"], [change])
            else:
                self.scheduler.load_state_dict(state["scheduler"])
                diffs = self.scheduler_param_diffs(
                    fresh_state, state["scheduler"], fresh_groups
                )
                if diffs and self.trainer_cfg.resume_scheduler == "config":
                    self.replay_scheduler(state["scheduler"], diffs)
                else:
                    self.check_auto_params()
                    self.warn_scheduler_params(diffs)

            # 再開元の総ステップ数の確認 (ファイルを写す前)
            self.check_total_steps()

    def resume_named(self, state: dict) -> None:
        """
        チェックポイントから, 名前付きの Optimizer と Scheduler の状態を戻す
        関数

        Scheduler の組み立て (クラス, warmup の有無) は名前ごとに再開元と
        同じものに限る. 引数の違いの確認と config の値での作り直し
        (resume_scheduler: config) には対応しない

        Parameters
        ----------
        state: dict
            チェックポイントの内容

        Returns
        ----------
        None
        """
        # Optimizer の状態を名前ごとに復元 (名前が違う場合はエラー)
        self.optimizer.load_state_dict(state["optimizer"])

        # Scheduler は名前ごとに組み立てを比べてから復元
        if all(scheduler is None for _, scheduler in self.scheduler.items()):
            if state.get("scheduler") is not None and any(
                value is not None
                for value in state["scheduler"].get("schedulers", {}).values()
            ):
                raise ValueError(
                    "再開元は Scheduler を使った学習ですが, 今回の optimizers "
                    "には Scheduler がありません"
                )
            return
        saved = state.get("scheduler_classes") or {}
        for name, scheduler in self.scheduler.items():
            new = scheduler_classes(scheduler)
            old = saved.get(name, []) if isinstance(saved, dict) else None
            if old is not None and old != new:
                raise ValueError(
                    f"optimizers.{name} の Scheduler の組み立てが再開元と違います"
                    f": 再開元は {' + '.join(old) or 'なし'}, 今回は "
                    f"{' + '.join(new) or 'なし'}"
                )
        self.scheduler.load_state_dict(state["scheduler"])
        self.check_named_auto_params()
        self.check_total_steps()

    def check_named_auto_params(self) -> None:
        """
        名前付きの Optimizer の Scheduler で, auto や割合で補った値が再開元の
        値と同じかを確かめる関数

        再開元の値のまま学習すると, config.yaml と train.log に残る補った値と
        実際の値が食い違うのでエラー

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 名前ごとに, 補った値と再開元から戻した Scheduler の値を比較
        cfg = self.experiment.cfg
        diffs = []
        for name, scheduler in self.scheduler.items():
            if scheduler is None:
                continue
            inner = scheduler
            if isinstance(scheduler, optim.lr_scheduler.SequentialLR):
                inner = scheduler._schedulers[-1]
            prefix = f"{name}."
            for key, value in self.auto_values.items():
                param = key[len(prefix):]
                if key.startswith(prefix) and getattr(
                    inner, param, value
                ) != value:
                    diffs.append(
                        f"optimizers.{name}.scheduler の {param} (再開元 "
                        f"{getattr(inner, param)}, 補った値 {value})"
                    )
            warmup = cfg.optimizers[name].scheduler.get("warmup_steps")
            if name in self.warmup_ratio_labels and isinstance(
                scheduler, optim.lr_scheduler.SequentialLR
            ) and scheduler._milestones[0] != warmup:
                diffs.append(
                    f"optimizers.{name}.scheduler の warmup_steps (再開元 "
                    f"{scheduler._milestones[0]}, 補った値 {warmup})"
                )
        if diffs:
            raise ValueError(
                "auto や warmup_ratio で補った Scheduler の引数が再開元と"
                f"違います: {', '.join(diffs)}. optimizers ではエポック数などを"
                "変えて再開できないので, 再開元と同じ設定にしてください"
            )

    def warn_select_best_change(self, state: dict) -> None:
        """
        再開元と今回で select_best が違う場合に警告する関数

        Parameters
        ----------
        state: dict
            再開元のチェックポイントの内容

        Returns
        ----------
        None
        """
        # この設定を入れる前のチェックポイントはベストモデルを選んでいたもの
        old = state.get("select_best", True)
        new = self.trainer_cfg.select_best
        if old == new:
            return
        if new:
            message = (
                "再開元はベストモデルを選ばない設定 (select_best: false) なので, "
                "ベストモデルは再開した後のエポックから選びます"
            )
        else:
            message = (
                "再開元のベストモデルは使わずに, 最後のエポックの重みを残します "
                "(select_best: false)"
            )
        self.logger.warning(message)
        warnings.warn(message, stacklevel=2)

    def restore_best_weights(self, state: dict, checkpoint_path: Path) -> None:
        """
        チェックポイントに無いベストモデルの重みを, 今のモデルか best.pt から
        戻す関数

        Parameters
        ----------
        state: dict
            再開元のチェックポイントの内容
        checkpoint_path: Path
            再開元のチェックポイントのパス

        Returns
        ----------
        None
        """
        best_epoch = self.checkpoint.best_epoch
        if best_epoch == 0:
            return

        # 再開元のエポックがベストモデルなら, そのモデルの重みを使用
        # 再開元がベストモデルの重みを eval_model から取っていた場合は, 今の
        # モデルと違うので best.pt から
        separate = self.source_separate(state)
        if state["epoch"] == best_epoch and not separate:
            self.checkpoint.best_model_state = copy_state_dict(state["model"])
            return

        # 同じディレクトリの best.pt が同じエポックのものなら, その重みを使用
        # 大きなモデルでもデバイスのメモリを使わないように, cpu の上で必要な
        # 部分だけを参照
        best_path = checkpoint_path.with_name("best.pt")
        reason = f"{best_path.name} がありません"
        if best_path.exists():
            best = torch.load(
                best_path, map_location="cpu", weights_only=True, mmap=True
            )
            # 重みだけの best.pt はベストモデルの重みそのもの, そうでなければ
            # 今のモデルの重み (再開元の eval_model が別の場合は best_model に
            # 保存. best_model は別の場合だけ入るので, あればそれを使用)
            weights = None
            if best["epoch"] == best_epoch:
                stored = (best.get("checkpoint") or {}).get("best_model")
                if best.get("weights_only"):
                    weights = best["model"]
                elif stored is not None:
                    weights = stored
                elif not separate:
                    weights = best["model"]
            if weights is not None:
                # 学習するパラメータだけの best.pt は今のモデルで補完
                meta = best.get("trainable_only")
                if isinstance(meta, dict):
                    weights = complete_weights(
                        weights, self.model.state_dict(),
                        meta.get("frozen") or {}, label=best_path.name
                    )
                self.checkpoint.best_model_state = copy_state_dict(weights)
                self.logger.info(
                    f"ベストモデル (エポック {best_epoch}) の重みを "
                    f"{best_path} から戻しました"
                )
                return
            elif best["epoch"] != best_epoch:
                reason = (
                    f"{best_path.name} はエポック {best['epoch']} のもので"
                    "合いません (every_n_epochs で last.pt の後に更新された"
                    "場合など)"
                )
            else:
                reason = (
                    f"{best_path.name} に eval_model のベストモデルの重みが"
                    "ありません"
                )

        # 見つからない場合は, 学習の後にベストモデルへ戻せないことを通知
        message = (
            f"ベストモデル (エポック {best_epoch}) の重みが last.pt に無く, "
            f"再開元の {reason}. 再開後に改善しなければ, 最後のエポックの重みが"
            "残ります"
        )
        self.logger.warning(message)
        warnings.warn(message, stacklevel=2)

    def source_separate(self, state: dict) -> bool:
        """
        再開元がベストモデルの重みを学習するモデルと別のモデルから取って
        いたかを判定する関数

        Parameters
        ----------
        state: dict
            再開元のチェックポイントの内容

        Returns
        ----------
        separate: bool
            再開元のチェックポイントの separate_best_model. 無い古い
            チェックポイントでは, config の EMA を使っていたか, 継承して
            eval_model を上書きして別のモデルを返す場合 (再開元も同じ上書きと
            みなす) は True
        """
        # 再開元が記録していればその値を使用
        if "separate_best_model" in state:
            separate = bool(state["separate_best_model"])
            return separate

        # 記録の無いチェックポイントでは, config の EMA は再開元の ema で,
        # 継承した上書きは今の設定で判定
        overridden = self.eval_model() is not self.model and not (
            self.ema_model is not None
            and self.eval_model() is self.ema_model.module
        )
        separate = state.get("ema") is not None or overridden

        return separate

    def copy_best(self, source: Path, target: Path) -> None:
        """
        再開元の best.pt が再開する状態のベストモデルの場合だけ写す関数

        Parameters
        ----------
        source: Path
            再開元の best.pt
        target: Path
            新しい保存先の best.pt

        Returns
        ----------
        None
        """
        # エポックだけを読むので, テンソルは読み込まずに cpu の上で参照
        best_epoch = torch.load(
            source, map_location="cpu", weights_only=True, mmap=True
        )["epoch"]

        # last.pt の保存の間隔を空けた場合などは, best.pt が再開で捨てる履歴の
        # ベストモデルのことがあるので, 再開する状態のベストのエポックと一致する場合だけコピー
        if best_epoch != self.checkpoint.best_epoch:
            self.logger.warning(
                f"再開元の best.pt (エポック {best_epoch}) は再開する状態の"
                f"ベストモデル (エポック {self.checkpoint.best_epoch}) と違うので"
                "写しません"
            )
            return

        # 重みだけにする設定で, 再開元が重み以外も含む場合は重みだけを保存
        if self.trainer_cfg.checkpoint.best_weights_only:
            best = torch.load(
                source, map_location="cpu", weights_only=True, mmap=True
            )
            if not best.get("weights_only"):
                # ベストモデルの重みを別のモデル (EMA など) から取った場合は,
                # model (学習しているモデル) ではなく best_model の重み
                stored = (best.get("checkpoint") or {}).get("best_model")
                weights = best["model"] if stored is None else stored
                light = {
                    "weights_only": True, "epoch": best["epoch"],
                    "monitor": best["monitor"],
                    "best_value": self.checkpoint.best_value,
                    "model": weights
                }
                if "trainable_only" in best:
                    light["trainable_only"] = best["trainable_only"]
                save_checkpoint(light, target)
                return
        shutil.copyfile(source, target)

    def log_settings(self, monitor: str) -> None:
        """
        学習の設定をログに記録する関数

        Parameters
        ----------
        monitor: str
            ベストモデルの判定に使う指標の名前

        Returns
        ----------
        None
        """
        # モデルの構造とパラメータ数を記録
        # print(model) で中身が分からない層もあるので, パラメータとバッファの
        # 一覧を model_summary.csv に保存
        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"モデルの構造:\n\n{self.model}\n")
        self.logger.info(f"パラメータ数: {n_params}")
        rows = summarize_model(self.model)
        save_model_summary(
            rows, str(self.experiment.run_dir / "model_summary.csv")
        )
        counts = count_parameters(rows)
        self.logger.info(
            f"学習するパラメータ数: {counts['trainable']}, 学習しない"
            f"パラメータ数: {counts['frozen']}, バッファの要素数: "
            f"{counts['buffer']} (一覧は model_summary.csv)"
        )

        # Optimizer, Scheduler, 監視する指標を記録
        # (パラメータを持たないモデルは, 作らないことを作成時に記録済み)
        if not self.no_params:
            self.logger.info(f"Optimizer:\n\n{self.optimizer}\n")
        if self.named:
            self.log_named_settings()
        elif not self.no_params:
            self.log_scheduler(self.scheduler)
        self.log_monitor(monitor)

        # 計算の精度, 勾配を貯めるバッチ数, チェックポイントの重みの範囲を記録
        trainable_only = self.trainer_cfg.checkpoint.trainable_only
        self.logger.info(
            f"計算の精度: {self.trainer_cfg.precision}, 勾配を貯めるバッチ数: "
            f"{self.accumulate_grad_batches}, チェックポイントの重み: "
            f"{'学習するパラメータとバッファだけ' if trainable_only else '全て'}"
        )

    def optimizer_info(self) -> list[dict]:
        """
        run.json に書く, Optimizer ごとの名前, クラス, 受け持つ部分, Scheduler
        と引数をまとめる関数

        Parameters
        ----------
        None

        Returns
        ----------
        info: list[dict]
            Optimizer ごとの name, optimizer, modules, scheduler,
            scheduler_params の辞書. 1 つの Optimizer の場合は name と
            modules が None の 1 つ. 学習を始めるまで作らない Scheduler は,
            作るまで None. パラメータを持たないモデルの場合は空
        """
        # パラメータを持たないモデルは Optimizer が無いので空
        if self.no_params:
            info = []
            return info

        # 名前付きの場合は config の順に, 受け持つ部分と Scheduler を付加
        if self.named:
            schedulers = (
                dict(self.scheduler.items())
                if isinstance(self.scheduler, NamedSchedulers) else {}
            )
            cfg = self.experiment.cfg
            info = [
                {
                    "name": name,
                    "optimizer": type(self.optimizer[name]).__name__,
                    "modules": [
                        str(module) for module in cfg.optimizers[name].modules
                    ],
                    "scheduler": scheduler_label(schedulers.get(name)),
                    "scheduler_params": scheduler_params(schedulers.get(name))
                }
                for name in self.optimizer
            ]
            return info

        # 1 つの Optimizer の場合
        info = [{
            "name": None, "optimizer": type(self.optimizer).__name__,
            "modules": None,
            "scheduler": scheduler_label(self.scheduler),
            "scheduler_params": scheduler_params(self.scheduler)
        }]

        return info

    def log_named_settings(self) -> None:
        """
        名前付きの Optimizer ごとの受け持つ部分と Scheduler を記録する関数

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # Scheduler の名前を 1 行にまとめ, 中の状態は名前ごとに記録
        cfg = self.experiment.cfg
        schedulers = (
            dict(self.scheduler.items())
            if isinstance(self.scheduler, NamedSchedulers) else {}
        )
        names = ", ".join(
            f"{name}={type(schedulers[name]).__name__}"
            if schedulers.get(name) is not None else f"{name}=なし"
            for name in self.optimizer
        )
        self.logger.info(f"Scheduler: {names}")
        for name in self.optimizer:
            modules = list(cfg.optimizers[name].modules)
            count = sum(p.numel() for p in self.named_params[name])
            self.logger.info(
                f"Optimizer {name} が受け持つ部分: {modules} "
                f"(学習するパラメータ数: {count})"
            )
            if schedulers.get(name) is not None:
                self.log_scheduler(schedulers[name], name)

    def log_scheduler(
        self, scheduler: Optional[optim.lr_scheduler.LRScheduler],
        name: str = ""
    ) -> None:
        """
        Scheduler の名前と状態を記録する関数

        Parameters
        ----------
        scheduler: Optional[optim.lr_scheduler.LRScheduler]
            記録する Scheduler, 使わない場合は None
        name: str = ""
            名前付きの Optimizer の名前, 1 つの Optimizer の場合は空

        Returns
        ----------
        None
        """
        # 1 つの Optimizer の場合は Scheduler の名前の行も記録
        label = f" ({name})" if name else ""
        if not name:
            scheduler_name = (
                "なし" if scheduler is None else type(scheduler).__name__
            )
            self.logger.info(f"Scheduler: {scheduler_name}")
        if scheduler is not None:
            # warmup を付けた場合は中の Scheduler ごとの状態も記録
            inners = [scheduler]
            if isinstance(scheduler, optim.lr_scheduler.SequentialLR):
                inners += list(scheduler._schedulers)
                self.logger.info(
                    f"Scheduler の区切り{label}: "
                    f"{scheduler.state_dict()['_milestones']}"
                )
            for inner in inners:
                scheduler_state = {
                    key: value for key, value in inner.state_dict().items()
                    if not key.startswith("_")
                }
                self.logger.info(
                    f"Scheduler の状態{label} ({type(inner).__name__}): "
                    f"{scheduler_state}"
                )

    def log_monitor(self, monitor: str) -> None:
        """
        監視する指標, 評価に使うモデル, 更新の手順を記録する関数

        Parameters
        ----------
        monitor: str
            ベストモデルの判定に使う指標の名前

        Returns
        ----------
        None
        """
        self.logger.info(f"監視する指標: {monitor} ({self.trainer_cfg.mode})")
        every = self.trainer_cfg.valid_every_n_epochs
        if every > 1:
            self.logger.info(
                f"検証の間隔: {every} エポックごとと予定の最後のエポック "
                f"({self.end_epoch}). 判定もこのエポックだけ"
            )
        if self.trainer_cfg.monitor_tiebreak is not None:
            self.logger.info(
                "同じ値の場合に比べる指標: "
                f"{self.trainer_cfg.monitor_tiebreak} "
                f"({self.trainer_cfg.monitor_tiebreak_mode})"
            )

        # 評価に使うモデルが別の場合は記録
        ema_cfg = self.trainer_cfg.ema
        if (
            self.ema_model is not None
            and self.eval_model() is self.ema_model.module
        ):
            self.logger.info(
                f"評価に使うモデル: EMA (decay={ema_cfg.decay}, "
                f"use_buffers={ema_cfg.use_buffers})"
            )
        elif self.eval_model() is not self.model:
            self.logger.info(
                "評価に使うモデル: eval_model が返すモデル "
                f"({type(self.eval_model()).__name__})"
            )

        # 更新の手順を記録
        if self.trainer_cfg.manual_optimization:
            self.logger.info("更新の手順: step の中で自分で更新")
        elif self.named:
            self.logger.info(
                "更新の手順: 名前付きの Optimizer ごとに, 損失の勾配を受け持つ"
                "パラメータだけに求めて, config の順に更新"
            )
        elif not self.use_gradients:
            self.logger.info(
                "更新の手順: 勾配を使わず, 損失を求める closure を Optimizer "
                "に渡して更新"
            )
        elif self.use_closure:
            self.logger.info("更新の手順: closure を Optimizer に渡して更新")

    def scheduler_param_diffs(
        self, fresh_state: dict, loaded_state: dict,
        fresh_groups: list[dict]
    ) -> list[str]:
        """
        再開元の Scheduler の引数のうち config と違うものを求める関数

        引数は名前を変えて持つものがある (CyclicLR の max_lr は max_lrs,
        OneCycleLR の epochs は total_steps, max_lr は Optimizer の
        max_lr など) ので, config の名前ではなく, config から作った直後の
        状態と再開元の状態を比較

        Parameters
        ----------
        fresh_state: dict
            config から作った直後の Scheduler の状態
        loaded_state: dict
            再開元のチェックポイントの Scheduler の状態
        fresh_groups: list[dict]
            config から作った直後の, Scheduler が Optimizer に書き込む値

        Returns
        ----------
        diffs: list[str]
            違う引数と, 再開元と config の値の説明
        """
        # 自分で作って渡した Scheduler は config と比べられないので確認なし
        if not self.scheduler_from_config:
            diffs = []
            return diffs

        # 学習の進み具合で変わる値と内部の値を除き, Scheduler の状態を比較
        # CyclicLR の base_lrs は Optimizer の学習率ではなく引数の base_lr
        # warmup を付けた場合 (SequentialLR) は, 中の Scheduler ごとに比較
        if isinstance(self.scheduler, optim.lr_scheduler.SequentialLR):
            fresh_state = sequential_arguments(fresh_state)
            loaded_state = sequential_arguments(loaded_state)
        compared = {
            key for key in fresh_state
            if (not key.startswith("_") and key not in PROGRESS_KEYS)
            or key in ARGUMENT_PRIVATE_KEYS
        }
        if isinstance(self.scheduler, optim.lr_scheduler.CyclicLR):
            compared.add("base_lrs")
        # 内部の値は, 元の引数の名前で表示 (中身は長いので省略)
        diffs = []
        for key in sorted(compared):
            loaded = loaded_state.get(key)
            if key not in loaded_state or fresh_state[key] == loaded:
                continue
            if key in ARGUMENT_PRIVATE_KEYS:
                diffs.append("pct_start, three_phase, anneal_strategy のどれか")
            else:
                diffs.append(
                    f"{key} (再開元 {loaded_state[key]}, config "
                    f"{fresh_state[key]})"
                )

        # Optimizer に書き込む値の比較
        loaded_groups = scheduler_group_values(self.optimizer)
        for i, (fresh, loaded) in enumerate(zip(fresh_groups, loaded_groups)):
            for key in fresh:
                if key in loaded and fresh[key] != loaded[key]:
                    diffs.append(
                        f"{key} (グループ {i}, 再開元 {loaded[key]}, "
                        f"config {fresh[key]})"
                    )

        return diffs

    def check_auto_params(self) -> None:
        """
        auto で補った Scheduler の引数が, 再開元の値と同じかを確かめる関数

        再開元の値のまま学習すると, config.yaml と train.log に残る補った
        値と実際の値が食い違うのでエラー

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 補った値と, 再開元から戻した Scheduler の値を比較
        # warmup を付けた場合は後ろの Scheduler の値と, 割合から補った回数
        scheduler = self.scheduler
        if isinstance(scheduler, optim.lr_scheduler.SequentialLR):
            scheduler = scheduler._schedulers[-1]
        diffs = [
            f"{name} (再開元 {getattr(scheduler, name)}, 補った値 {value})"
            for name, value in self.auto_values.items()
            if getattr(scheduler, name, value) != value
        ]
        warmup = self.experiment.cfg.scheduler.get("warmup_steps")
        if self.warmup_from_ratio and isinstance(
            self.scheduler, optim.lr_scheduler.SequentialLR
        ) and self.scheduler._milestones[0] != warmup:
            diffs.append(
                f"warmup_steps (再開元 {self.scheduler._milestones[0]}, "
                f"補った値 {warmup})"
            )
        if diffs:
            raise ValueError(
                "auto で補った Scheduler の引数が再開元と違います: "
                f"{', '.join(diffs)}. エポック数などを変えて再開する場合は "
                "trainer.resume_scheduler: config にしてください"
            )

    def warn_scheduler_params(self, diffs: list[str]) -> None:
        """
        再開元の Scheduler の引数が config と違う場合に警告する関数

        Parameters
        ----------
        diffs: list[str]
            違う引数と, 再開元と config の値の説明

        Returns
        ----------
        None
        """
        # Scheduler の状態は引数ごと戻るので, config で変えた引数は無効
        if len(diffs) == 0:
            return

        # 警告とログで通知
        message = (
            "再開元の Scheduler の引数が config と違うので, "
            f"再開元の値を使います: {', '.join(diffs)}. config の値を使う"
            "場合は trainer.resume_scheduler: config にしてください"
        )
        warnings.warn(message)
        self.logger.warning(message)

    def replay_scheduler(self, loaded_state: dict, diffs: list[str]) -> None:
        """
        config の引数で Scheduler を作り直し, 再開元と同じステップ数まで
        進める関数

        通しで config の引数で学習した場合と同じ学習率になる. 指標の値で
        学習率を変える ReduceLROnPlateau は進め直せないので, 再開元の状態を
        使って警告

        Parameters
        ----------
        loaded_state: dict
            再開元のチェックポイントの Scheduler の状態
        diffs: list[str]
            違う引数と, 再開元と config の値の説明

        Returns
        ----------
        None
        """
        # 指標の値に依存する Scheduler は再開元の状態のまま
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            message = (
                "ReduceLROnPlateau は進め直せないので, 再開元の Scheduler の"
                f"状態を使います: {', '.join(diffs)}"
            )
            warnings.warn(message)
            self.logger.warning(message)
            return

        # 学習率を最初の値に戻してから, config の引数で作り直し
        # 再開元と同じ回数だけ進める (Optimizer の更新は伴わないので,
        # 順番についての PyTorch の警告は抑制)
        steps = int(loaded_state["last_epoch"])
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"]
        self.scheduler = build_scheduler(
            self.experiment.cfg.scheduler, self.optimizer,
            self.trainer_cfg.mode
        )

        # 総ステップ数の決まった Scheduler は, 進めた回数を超えられない
        total = getattr(self.scheduler, "total_steps", None)
        if total is not None and steps > total:
            raise ValueError(
                f"再開元は {steps} ステップ進めていて, config の "
                f"{type(self.scheduler).__name__} の総ステップ数 ({total}) "
                "を超えているので, config の引数では作り直せません"
            )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*lr_scheduler.step\\(\\)",
                category=UserWarning
            )
            for _ in range(steps):
                self.scheduler.step()

        # 作り直したことをログに記録
        self.logger.info(
            "再開元と config で Scheduler の引数が違うので, config の引数で"
            f"作り直して {steps} ステップ進めました: {', '.join(diffs)}"
        )

    def warn_persistent_workers(
        self, loaders: list[Optional[Iterable]]
    ) -> None:
        """
        ワーカーを使い回す DataLoader で再開する場合に警告する関数

        Parameters
        ----------
        loaders: list[Optional[Iterable]]
            学習と検証に使う DataLoader など, 検証しない場合は None を含む

        Returns
        ----------
        None
        """
        # persistent_workers の DataLoader はワーカーの乱数の種を最初の 1 回だけ引くので,
        # 再開すると通しで学習した場合と乱数の流れにずれが発生
        for loader in loaders:
            persistent = getattr(loader, "persistent_workers", False)
            if persistent and getattr(loader, "num_workers", 0) > 0:
                message = (
                    "persistent_workers=True の DataLoader で再開すると, "
                    "通しで学習した場合と結果が一致しません"
                )
                warnings.warn(message)
                self.logger.warning(message)
                return

    def finish(self, elapsed_time: float) -> None:
        """
        学習後にベストモデルを戻し, 結果を保存する関数

        Parameters
        ----------
        elapsed_time: float
            学習にかかった秒数

        Returns
        ----------
        None
        """
        # 設定に応じてベストモデルの重みに戻し,
        # 推論で使いやすいように重みだけを cpu に移して保存
        # ベストモデルの重みが無い場合 (再開元から戻せなかった場合) は最後のまま
        run_dir = self.experiment.run_dir
        self.final_weights = "partial" if self.partial_weights else "last"
        uses_best = (
            self.trainer_cfg.restore_best and self.trainer_cfg.select_best
            and self.checkpoint.best_model_state is not None
        )

        # fit を呼び直した場合に続けられるように, 置き換える前の重みを cpu に退避
        self.continue_weights = None
        if uses_best or self.eval_model() is not self.model:
            self.continue_weights = clone_to_cpu(self.model.state_dict())
        if uses_best:
            self.checkpoint.restore_best(self.model)
            self.final_weights = "best"
        else:
            # 評価に使うモデルが別 (EMA など) の場合は, その重みを保存
            eval_model = self.eval_model()
            if eval_model is not self.model:
                self.model.load_state_dict(eval_model.state_dict())
        weights = {
            "best": "ベストモデル", "last": "最後のエポック",
            "partial": "途中で止めたエポックの途中まで"
        }[self.final_weights]
        final_state = {
            key: (
                value.detach().cpu() if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in self.model.state_dict().items()
        }
        torch.save(final_state, run_dir / "final_model.pt")

        # 記録と学習曲線を保存
        self.history.save_csv(str(run_dir / "history.csv"))
        plot_history(self.history.records, str(run_dir))

        # step ごとに学習率を変えた場合は, 更新ごとの学習率も保存
        # 画像の名前は, 指標の学習曲線 (<指標>_curve.png) と重ならない形
        if self.scheduler_interval == "step" and len(self.lr_steps) > 0:
            self.lr_steps.save_csv(str(run_dir / "lr_steps.csv"))
            plot_lr_steps(
                self.lr_steps.records(), str(run_dir / "lr_steps.png")
            )

        # 結果をログに記録
        self.logger.info("学習完了")
        if self.trainer_cfg.select_best:
            self.logger.info(
                f"ベストモデルのエポック: {self.checkpoint.best_epoch}"
            )
            self.logger.info(f"ベストモデルの指標: {self.checkpoint.best_value}")
        else:
            self.logger.info("ベストモデルは選ばない設定 (select_best: false)")
        self.logger.info(f"final_model.pt の重み: {weights}")
        self.logger.info(f"学習時間: {elapsed_time:.2f} 秒")

        # エポックの途中で止めたときに戻すための状態は, 学習の後は不要なので解放
        self.start_state = None

    def fit(
        self, train_loader: Iterable, valid_loader: Optional[Iterable] = None
    ) -> History:
        """
        モデルを学習する関数

        Parameters
        ----------
        train_loader: Iterable
            学習用のバッチを返すもの, DataLoader や [data] のようなリスト
        valid_loader: Optional[Iterable] = None
            検証用のバッチを返すもの, None の場合は検証しない

        Returns
        ----------
        history: History
            エポックごとの記録
        """
        # 前の fit の後の処理の心拍は, この Trainer のものなら消し, 他の
        # Trainer のものなら終了
        if self.heartbeat is not None:
            self.heartbeat.cancel()
            self.heartbeat = None
        end_heartbeats()

        # 途中で止まった場合も原因が分かるように, エラーを train.log に記録して再送出
        # 外から読める状態にも, 完了, 中断, 失敗のどれかを記録
        try:
            history = self.train_epochs(train_loader, valid_loader)
        except KeyboardInterrupt:
            self.logger.warning("キーボードの割り込みで学習が止まりました")
            self.run_status.finish("stopped", error="KeyboardInterrupt")
            raise
        except Exception as error:
            self.logger.exception("エラーで学習が止まりました")
            self.run_status.finish(
                "failed", error=f"{type(error).__name__}: {error}"
            )
            raise

        # 結果のファイルを全て保存した後に完了の書き込み (読み手は完了を見てから読み込み可能)
        self.run_status.finish(
            "stopped" if self.stopped_by_request else "completed",
            completed_epochs=len(self.history.records),
            best_epoch=self.checkpoint.best_epoch or None,
            best_value=self.checkpoint.best_value,
            final_weights=self.final_weights
        )

        # スクリプトが続ける学習の後の処理 (評価, 推論の一式の書き出し
        # など) の間, 動いている印を書き続ける心拍の開始
        self.heartbeat = PostHeartbeat(self.run_status.run_dir, self.logger)
        self.heartbeat.start()

        return history

    def end_post(self) -> None:
        """
        学習の後の処理を終えたことを heartbeat.json に書く関数

        プロセスが普通に終われば自動で書くので, ノートブックのように
        プロセスが残り続ける場合に呼ぶ

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # fit を終えていない場合は心拍が無いので対象外
        if self.heartbeat is None:
            return
        self.heartbeat.end()

    def build_scheduler_from_data(self, train_loader: Iterable) -> None:
        """
        データの量で決まる Scheduler の引数を学習用データから補って作る関数

        名前付きの Optimizer の場合は, 作成を保留した Optimizer ごとに作る

        Parameters
        ----------
        train_loader: Iterable
            学習用のバッチを返すもの, len() でバッチ数を取得

        Returns
        ----------
        None
        """
        # 作成を保留した Scheduler の config と Optimizer と, 書く場所の名前
        cfg = self.experiment.cfg
        if self.named:
            targets = [
                (
                    name, cfg.optimizers[name].scheduler,
                    self.optimizer[name], f"optimizers.{name}.scheduler"
                )
                for name in self.optimizer if name in self.pending_named
            ]
        else:
            targets = [("", cfg.scheduler, self.optimizer, "scheduler")]

        # 1 エポックで Scheduler を進める回数 (step ごとならバッチ数)
        per_epoch = 1
        if self.scheduler_interval == "step":
            try:
                per_epoch = self.updates_per_epoch(len(train_loader))
                if per_epoch == 0:
                    raise ValueError("train の loader が空です")
            except TypeError as error:
                # バッチ数が要る引数を名前に出して, 書き方を案内
                _, scheduler_cfg, _, section = targets[0]
                missing = missing_data_size_params(scheduler_cfg)
                needed = [name for name in missing if name != "epochs"]
                name = needed[0] if needed else "warmup_ratio"
                hint = (
                    " total_steps か steps_per_epoch を"
                    if name == "steps_per_epoch" else "値を"
                )
                raise ValueError(
                    f"{scheduler_cfg.name or 'warmup'} の {name} を補えません. "
                    "train_loader が len() を持たないので, "
                    f"{section} の設定に{hint}書いてください"
                ) from error

        # Scheduler ごとに引数を補って作成
        built = {}
        messages = []
        for label, scheduler_cfg, optimizer, section in targets:
            with named_errors(label) if label else contextlib.nullcontext():
                built[label], filled = self.fill_scheduler(
                    scheduler_cfg, optimizer, per_epoch, label
                )
            messages.append(
                f"{scheduler_cfg.name or 'warmup'} の引数を学習用データから"
                f"補いました: {', '.join(filled)}" if not label else
                f"{section} ({scheduler_cfg.name or 'warmup'}): "
                f"{', '.join(filled)}"
            )
        if self.named:
            schedulers = dict(self.scheduler.schedulers)
            schedulers.update(built)
            self.scheduler = NamedSchedulers(schedulers)
        else:
            self.scheduler = built[""]
        self.pending_scheduler = False

        # 学習中に読まれても壊れた中身が見えないように, 一時ファイルから置き換え
        path = self.experiment.run_dir / "config.yaml"
        temp = path.with_name("config.yaml.tmp")
        OmegaConf.save(cfg, temp, resolve=True)
        temp.replace(path)
        if self.named:
            messages = [
                "Scheduler の引数を学習用データから補いました: "
                f"{'; '.join(messages)}"
            ]
        self.logger.info(messages[0])

    def fill_scheduler(
        self, scheduler_cfg: DictConfig, optimizer: optim.Optimizer,
        per_epoch: int, label: str
    ) -> tuple[Optional[optim.lr_scheduler.LRScheduler], list[str]]:
        """
        1 つの Scheduler の, データの量で決まる引数を補って作る関数

        Parameters
        ----------
        scheduler_cfg: DictConfig
            scheduler の config, 補った値を書き込む
        optimizer: optim.Optimizer
            Scheduler を付ける Optimizer
        per_epoch: int
            1 エポックで Scheduler を進める回数
        label: str
            名前付きの Optimizer の名前, 1 つの Optimizer の場合は空

        Returns
        ----------
        scheduler: Optional[optim.lr_scheduler.LRScheduler]
            作った Scheduler
        filled: list[str]
            補った引数と値 (ログに書く形)
        """
        # 学習全体のステップ数と, 割合で書いた warmup の回数
        auto = set(auto_params(scheduler_cfg))
        missing = missing_data_size_params(scheduler_cfg)
        total = self.trainer_cfg.epochs * per_epoch
        values: dict[str, int] = {}
        filled = []
        if scheduler_cfg.get("warmup_ratio") is not None:
            warmup = int(round(scheduler_cfg.warmup_ratio * total))
            scheduler_cfg.warmup_steps = warmup
            scheduler_cfg.warmup_ratio = None
            filled.append(f"warmup_steps={warmup}")
            self.warmup_from_ratio = True
            self.warmup_ratio_labels.add(label)
        warmup = int(scheduler_cfg.get("warmup_steps") or 0)

        # OneCycleLR の引数と, auto と書いた引数 (warmup の後の残りの回数)
        for name in missing:
            if name == "epochs":
                values[name] = self.trainer_cfg.epochs
            elif name in auto:
                values[name] = total - warmup
            else:
                values[name] = per_epoch
        if any(value < 1 for value in values.values()):
            raise ValueError(
                f"warmup ({warmup}) が学習全体のステップ数 ({total}) 以上なので, "
                f"{scheduler_cfg.name} の auto を補えません"
            )
        prefix = f"{label}." if label else ""
        self.auto_values.update(
            {f"{prefix}{name}": values[name] for name in auto}
        )

        # config に書き込んで作成
        for name, value in values.items():
            scheduler_cfg.params[name] = value
            filled.append(f"{name}={value}")
        scheduler = build_scheduler(
            scheduler_cfg, optimizer, self.trainer_cfg.mode
        )

        return scheduler, filled

    def training_state(self) -> dict:
        """
        エポックの途中で止めたときに戻すための, 学習の状態の写しを作る関数

        Parameters
        ----------
        None

        Returns
        ----------
        state: dict
            モデル, Optimizer, Scheduler, 更新回数, 乱数, loader (環境の
            状態と乱数), 継承したクラスが加えた状態 (テンソルは cpu の写し)
        """
        # チェックポイントと同じ名前で, 戻すのに必要なものだけを写す
        state = clone_to_cpu({
            "model": self.model.state_dict(),
            "optimizer": (
                None if self.optimizer is None
                else self.optimizer.state_dict()
            ),
            "scheduler": (
                None if self.scheduler is None
                else self.scheduler.state_dict()
            ),
            "global_step": self.global_step,
            "rng": get_rng_state(),
            "loaders": self.loader_states(),
            "ema": self.ema_state(),
            "extra": self.extra_state_dict()
        })

        return state

    def load_training_state(self, state: dict) -> None:
        """
        training_state か last.pt の状態に, 学習の状態を戻す関数

        Parameters
        ----------
        state: dict
            戻す状態 (model, optimizer, scheduler, global_step, rng, loaders,
            ema, extra)

        Returns
        ----------
        None
        """
        # モデル, Optimizer, Scheduler, 更新回数の復元
        self.model.load_state_dict(state["model"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = int(state.get("global_step", self.global_step))

        # 乱数, loader, EMA, 継承したクラスが加えた状態の復元
        for message in set_rng_state(state["rng"]):
            self.logger.warning(message)
        self.load_loader_states(state.get("loaders"), resuming=False)
        if self.ema_model is not None and state.get("ema") is not None:
            self.ema_model.load_state_dict(state["ema"]["state"])
        self.load_extra_state_dict(state.get("extra", {}))

    def discard_epoch(
        self, epoch: int, reason: str = "中断の要求で"
    ) -> None:
        """
        途中で止めたエポックを捨て, 最後に終えたエポックの状態に戻す関数

        記録 (history) にはまだ加えていないので, 更新ごとの学習率と,
        モデル, Optimizer, Scheduler, 更新回数, 乱数, 継承したクラスが加えた
        状態を戻す (同じ Trainer で fit を呼び直せるように). last.pt は
        書かないので, 再開は最後に保存した last.pt から

        Parameters
        ----------
        epoch: int
            途中で止めたエポック
        reason: str = "中断の要求で"
            ログに書く, 止めた理由

        Returns
        ----------
        None
        """
        self.stopped_by_request = True
        self.current_batch = None
        self.lr_steps.drop_from_epoch(epoch)
        message = f"{reason}エポック {epoch} の途中で学習を止めました"

        # 学習を始めたエポックなら始めた時点の状態, そうでなければ前のエポックの
        # last.pt の状態に戻す
        last_path = self.experiment.run_dir / "last.pt"
        source = None
        if epoch == self.start_epoch and self.start_state is not None:
            source = self.start_state
            message += ". 状態は学習を始めた時点のものに戻しました"
        elif last_path.exists():
            last = torch.load(
                last_path, map_location="cpu", weights_only=True, mmap=True
            )
            # この fit で書いた前のエポックのものか, 更新回数でも確認
            # (前の fit が残した同じエポックの last.pt を除外)
            if (
                last.get("epoch") == epoch - 1
                and last.get("global_step") == self.epoch_start_step
            ):
                source = last
                message += f". 状態はエポック {epoch - 1} のものに戻しました"
        if source is not None:
            # 評価に使うモデルが別の場合は, 戻したかを重みの変化で判定
            # (config の EMA は状態と一緒に戻すので判定不要)
            separate = self.eval_model() is not self.model and not (
                self.ema_model is not None
                and self.eval_model() is self.ema_model.module
            )
            before = (
                clone_to_cpu(self.eval_model().state_dict()) if separate
                else None
            )
            self.load_training_state(source)

            # 評価に使うモデルの状態を extra_state_dict で保存していない場合は,
            # そのモデルの途中の更新は戻せないので途中までの重み
            unchanged = before is not None and all(
                torch.equal(before[key], value.detach().cpu())
                for key, value in self.eval_model().state_dict().items()
                if isinstance(value, torch.Tensor)
            )
            if unchanged:
                self.partial_weights = True
                message += (
                    ". 評価に使うモデル (eval_model) の状態は extra_state_dict "
                    "に無いので戻せず, final_model.pt は途中までの重みです"
                )
                self.logger.warning(message)
                warnings.warn(message, stacklevel=2)
                return
            self.logger.info(message)
            return

        # 戻せない場合は, ベストモデルに戻す設定でなければ途中までの重み
        uses_best = (
            self.trainer_cfg.restore_best and self.trainer_cfg.select_best
            and self.checkpoint.best_model_state is not None
        )
        message += (
            f". エポック {epoch - 1} の last.pt が無い (every_n_epochs で保存し"
            "なかったなど) ので, 学習の状態は途中までのものです"
        )
        if not uses_best:
            self.partial_weights = True
            message += " (final_model.pt も途中までの重み)"
        self.logger.warning(message)
        warnings.warn(message, stacklevel=2)

    def updates_per_epoch(self, n_batches: int) -> int:
        """
        1 エポックのバッチ数から, パラメータを更新する回数を求める関数

        Parameters
        ----------
        n_batches: int
            1 エポックのバッチ数

        Returns
        ----------
        updates: int
            勾配を貯める数で割って切り上げた回数 (端数も 1 回更新)
        """
        # 貯める数ごとに 1 回, エポックの最後の端数も 1 回
        updates = math.ceil(n_batches / self.accumulate_grad_batches)

        return updates

    def plan_total_steps(
        self, train_loader: Iterable, continuing: bool = False
    ) -> None:
        """
        OneCycleLR を使う場合に, 予定の総ステップ数を求めて保存する関数

        Parameters
        ----------
        train_loader: Iterable
            学習用のバッチを返すもの, len() が無い場合は求めない
        continuing: bool = False
            同じ Trainer で fit を呼び直した続きの学習かどうか

        Returns
        ----------
        None
        """
        self.planned_total_steps = None
        if self.scheduler_interval == "epoch":
            per_epoch = 1
        else:
            try:
                per_epoch = self.updates_per_epoch(len(train_loader))
            except TypeError:
                return
        total = self.end_epoch * per_epoch

        # 続きの学習は, 前の fit までに進めた回数にこの fit の分を足した数
        # (前の fit とバッチ数が違っても合うように, 進めた回数から計算)
        if continuing:
            done = (
                self.global_step if self.scheduler_interval == "step"
                else self.start_epoch - 1
            )
            total = done + self.trainer_cfg.epochs * per_epoch

        # warmup が学習全体以上か, 割合が小さく 0 回になった場合は通知
        # (続きの学習では warmup を終えているので確認不要)
        for label, scheduler, scheduler_cfg in self.scheduler_entries():
            warmup = scheduler_cfg.get("warmup_steps")
            section = (
                f"optimizers.{label}.scheduler" if label else "scheduler"
            )
            subject = (
                f"{section}.warmup_steps" if label else "warmup"
            )
            message = None
            if not continuing and warmup and warmup >= total:
                message = (
                    f"{subject} ({warmup} 回) が学習全体のステップ数 "
                    f"({total}) 以上なので, 後ろの Scheduler に切り替わりません"
                )
            elif (
                not continuing and warmup == 0
                and label in self.warmup_ratio_labels
            ):
                message = (
                    f"{section}.warmup_ratio が小さく, warmup は 0 回に"
                    "なりました"
                )
            if message is not None:
                self.logger.warning(message)
                warnings.warn(message, stacklevel=2)

            # OneCycleLR の総ステップ数の確認に使う予定の値
            if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                self.planned_total_steps = total

    def check_total_steps(self) -> None:
        """
        OneCycleLR の総ステップ数が, 予定の総ステップ数と合うかを確かめる関数

        総ステップ数を超えて進めると学習の途中で PyTorch がエラーを出すので,
        学習を始める前にエラーにする. 少ない場合は学習率が下がりきらないので警告.
        再開した場合は再開元の総ステップ数が使われるので, trainer.epochs や
        バッチ数を変えて再開した場合も確認. fit を呼び直した場合は, 前の fit
        までを含めた予定の総ステップ数と比べる

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # Scheduler ごと (名前付きの Optimizer なら名前ごと) に確認
        planned = self.planned_total_steps
        for label, scheduler, _ in self.scheduler_entries():
            total = getattr(scheduler, "total_steps", None)
            if total is None or planned is None or total == planned:
                continue
            self.warn_total_steps(scheduler, label, total, planned)

    def warn_total_steps(
        self, scheduler: optim.lr_scheduler.LRScheduler, label: str,
        total: int, planned: int
    ) -> None:
        """
        OneCycleLR の総ステップ数が予定と違うことを, 多い場合は警告し,
        少ない場合はエラーにする関数

        Parameters
        ----------
        scheduler: optim.lr_scheduler.LRScheduler
            確かめた Scheduler
        label: str
            名前付きの Optimizer の名前, 1 つの Optimizer の場合は空
        total: int
            Scheduler の総ステップ数
        planned: int
            予定の総ステップ数

        Returns
        ----------
        None
        """
        prefix = f"optimizers.{label}: " if label else ""
        message = (
            f"{prefix}{type(scheduler).__name__} の総ステップ数 ({total}) と, "
            f"trainer.epochs と学習用データのバッチ数から求めた予定の総ステップ数 "
            f"({planned}) が違います. 再開した場合は再開元の総ステップ数が使われ"
            "ます"
        )
        if not self.named:
            message += (
                " (trainer.resume_scheduler: config で config の値で作り直せ"
                "ます)"
            )
        if self.loop_started:
            message += (
                ". 同じ Trainer で fit を呼び直した場合は, 前の fit の分も含めた"
                "予定の総ステップ数で確かめます"
            )
        if planned > total:
            raise ValueError(message)
        message += " (学習率は最後まで下がりきりません)"
        self.logger.warning(message)
        warnings.warn(message, stacklevel=2)

    def prepare_continue(self) -> None:
        """
        同じ Trainer で fit を呼び直した場合に, 前の fit の続きを準備する関数

        最後に終えたエポックの次から trainer.epochs の分だけ学習するように
        エポックの範囲を決める. 前の fit がエポックの途中でエラーか割り込みで
        止まった場合は, そのエポックを捨てて最後に終えたエポックの状態に戻す.
        モデルの重みと Early Stopping の状態は, 学習の前の確認を全て通って
        から apply_continue で変える

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 前の fit がエポックの途中で止まった場合は, そのエポックの捨て
        records = self.history.records
        last_epoch = (
            int(records[-1]["epoch"]) if records else self.start_epoch - 1
        )
        if not self.fit_finished and self.epoch > last_epoch:
            self.discard_epoch(
                self.epoch, "前の fit がエラーか割り込みで止まったので,"
            )
            self.stopped_by_request = False

        # 最後に終えたエポックの次から開始
        self.start_epoch = last_epoch + 1
        self.end_epoch = last_epoch + self.trainer_cfg.epochs
        self.epoch = last_epoch

        # 前の fit で置いた STOP が残っていると続きの学習もすぐに止まるので削除
        stop_path = self.experiment.run_dir / STOP_FILE
        if stop_path.exists():
            stop_path.unlink()
            self.logger.info("前の fit で置いた STOP のファイルを消しました")

        # auto で補った Scheduler の長さは最初の fit の分なので, 続きの学習では
        # 予定の外に出ることを通知
        if self.auto_values:
            filled = ", ".join(
                f"{name}={value}" for name, value in self.auto_values.items()
            )
            message = (
                f"auto で補った Scheduler の引数 ({filled}) は最初の fit の長さ"
                "です. 続きの学習では予定の長さを超えて Scheduler を進めます"
            )
            self.logger.warning(message)
            warnings.warn(message, stacklevel=4)
        self.logger.info(
            f"前の fit の続きとして, エポック {self.start_epoch} から "
            f"{self.end_epoch} まで学習します"
        )

    def apply_continue(self) -> None:
        """
        続きの学習を始める直前に, モデルの重みと Early Stopping の状態を
        続きの学習の始めの状態にする関数

        前の fit の終わりにベストモデルなどの重みに置き換えた場合は, 置き換える
        前 (最後のエポック) の重みに戻し, Early Stopping は止めた印と
        我慢の回数を戻す. Optimizer, Scheduler, 更新回数, 記録, ベストモデルは
        そのまま続ける

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # Optimizer の状態と揃うように, 最後のエポックの重みに復元
        if self.continue_weights is not None:
            self.model.load_state_dict(self.continue_weights)
            self.continue_weights = None

        # Early Stopping の止めた印と我慢の回数の初期化
        if self.early_stopping is not None:
            self.early_stopping.counter = 0
            self.early_stopping.stopped = False

    def train_epochs(
        self, train_loader: Iterable, valid_loader: Optional[Iterable]
    ) -> History:
        """
        エポックごとに学習と検証を繰り返す関数

        Parameters
        ----------
        train_loader: Iterable
            学習用のバッチを返すもの
        valid_loader: Optional[Iterable]
            検証用のバッチを返すもの, None の場合は検証しない

        Returns
        ----------
        history: History
            エポックごとの記録
        """
        # 監視する指標の決定
        monitor = self.get_monitor(valid_loader is not None)

        # 同じ Trainer で fit を繰り返す場合に備えて, 中断の印を初期化
        self.stopped_by_request = False

        # チェックポイントに状態を保存する loader (環境の状態と乱数など)
        self.loaders = {"train": train_loader, "valid": valid_loader}

        # 前の fit が学習を始めていれば, この fit は続き (最後に終えたエポックの
        # 次から, trainer.epochs の分だけ追加で学習)
        continuing = self.loop_started
        if continuing:
            # ベストモデルと Early Stopping の値を比べられるように, 監視する
            # 指標は前の fit と同じものに制限 (検証用データの有無など)
            if monitor != self.last_monitor:
                raise ValueError(
                    f"前の fit と監視する指標が違います: 前の fit は "
                    f"{self.last_monitor}, 今回は {monitor}. 続きの学習では"
                    "同じ指標になるように, 検証用データの有無を揃えてください"
                )
            self.prepare_continue()
        else:
            self.end_epoch = self.trainer_cfg.epochs
        self.fit_finished = False

        # 再開の前に失敗しても分かるように, 学習を始めたことを書き出し
        self.run_status.start(
            planned_epochs=self.end_epoch, start_epoch=self.start_epoch,
            monitor=monitor, mode=self.trainer_cfg.mode,
            seed=self.experiment.cfg.experiment.seed,
            device=str(self.device),
            select_best=self.trainer_cfg.select_best,
            continue_run=continuing, optimizers=self.optimizer_info()
        )

        # パラメータを持たないモデルは学習しないので, 検証用のデータが必要
        if self.no_params and valid_loader is None:
            raise ValueError(
                "パラメータを持たないモデルは学習のフェーズを回さないので, "
                "fit に検証用のデータ (valid_loader) を渡してください"
            )

        # 学習の指標 (train/) は記録しないので, 監視する指標は検証か評価し直し
        # の指標に制限
        prefixes = ["valid/"] + (
            ["train_eval/"] if self.trainer_cfg.eval_train else []
        )
        if self.no_params and not monitor.startswith(tuple(prefixes)):
            raise ValueError(
                f"パラメータを持たないモデルは学習のフェーズを回さず, 監視する"
                f"指標 {monitor} を記録しません. trainer.monitor には "
                f"{' か '.join(prefixes)} で始まる指標を書いてください"
            )

        # 学習用データの評価し直しは, ワーカーを使い回す DataLoader では
        # ワーカーの中の乱数を進めるので, 学習が変わることを通知
        persistent = getattr(train_loader, "persistent_workers", False)
        if (
            self.trainer_cfg.eval_train and persistent
            and getattr(train_loader, "num_workers", 0) > 0
        ):
            message = (
                "eval_train を persistent_workers=True の DataLoader と使うと, "
                "ワーカーの中の乱数が進むので, 学習の結果が eval_train: false "
                "の場合と一致しません"
            )
            warnings.warn(message, stacklevel=3)
            self.logger.warning(message)

        # 評価に使うモデルの確認 (失敗も run.json に残るように書き出しの後)
        self.check_eval_model()
        self.checkpoint.separate_best_model = (
            self.eval_model() is not self.model
        )

        # データの量で決まる Scheduler の引数を補って作成 (再開の復元の前)
        # 予定の総ステップ数を求め, 再開しない場合はここで確認
        if self.pending_scheduler:
            self.build_scheduler_from_data(train_loader)
            self.run_status.update(
                force=True, optimizers=self.optimizer_info()
            )
        self.plan_total_steps(train_loader, continuing)
        if self.trainer_cfg.resume_from is None or continuing:
            self.check_total_steps()

        # 再開する場合はチェックポイントから状態の復元 (最初の fit だけ)
        if self.trainer_cfg.resume_from is not None and not continuing:
            self.warn_persistent_workers([train_loader, valid_loader])
            self.resume(self.trainer_cfg.resume_from, monitor)
            self.run_status.update(
                force=True,
                start_epoch=self.start_epoch,
                completed_epochs=len(self.history.records),
                best_epoch=self.checkpoint.best_epoch or None,
                best_value=self.checkpoint.best_value,
                early_stopped_epoch=(
                    self.start_epoch - 1
                    if self.early_stopping is not None
                    and self.early_stopping.stopped else None
                )
            )

        # 続きの学習は, 確認を全て通ってからモデルの重みなどを変更
        if continuing:
            self.apply_continue()

        # Early Stopping で既に止めた学習を再開した場合は学習なし
        epochs = range(self.start_epoch, self.end_epoch + 1)
        if self.early_stopping is not None and self.early_stopping.stopped:
            self.logger.info("Early Stopping で止めた学習のため学習しません")
            epochs = range(0)

        # 学習の設定を記録
        self.log_settings(monitor)

        # エポックの途中で止めたときに戻せるように, 始めた時点の状態を cpu に
        # 退避 (以降のエポックでは last.pt から戻す)
        self.partial_weights = False
        self.start_state = self.training_state()
        self.epoch_start_step = self.global_step

        # 学習を始めた印 (この後に fit を呼び直すと続きの学習) と監視する指標
        self.loop_started = True
        self.last_monitor = monitor

        # エポックごとに学習
        # 止めたエポックも数えられるように, 進捗バーは手動で更新
        start_time = time.time()
        every = self.trainer_cfg.checkpoint.every_n_epochs
        disable = not self.trainer_cfg.progress_bar
        with tqdm(total=len(epochs), disable=disable) as pbar:
            for epoch in epochs:
                pbar.set_description(f"エポック {epoch}")
                self.epoch = epoch

                # このエポックを始めた時点の学習率の保持
                lrs = self.current_lrs()

                # 学習し, 判定するエポックでは評価し直しと検証
                # (判定しないエポックは学習だけ. パラメータを持たないモデルは
                # 学習なし)
                # エポックの途中で止める要求があれば, このエポックを捨てて終了
                judged = self.is_judged_epoch(epoch)
                record = {"epoch": epoch}
                self.epoch_extra = {}
                self.epoch_start_step = self.global_step
                try:
                    if not self.no_params:
                        record.update(
                            self.run_epoch(train_loader, train=True)
                        )
                    if judged and self.trainer_cfg.eval_train:
                        record.update(
                            self.run_valid_epoch(train_loader, "train_eval")
                        )
                    if judged and valid_loader is not None:
                        record.update(self.run_valid_epoch(valid_loader))
                except StopWithinEpoch:
                    self.discard_epoch(epoch)
                    pbar.write(
                        f"中断の要求でエポック {epoch} の途中で学習を止めました"
                    )
                    break
                record.update(lrs)

                # step の extra で返した値は, 平均せずにエポックの最後の値を記録
                # 指標と同じ名前は学習曲線とダッシュボードで混ざるので禁止
                for name in self.epoch_extra:
                    if name in lrs or any(
                        f"{phase}/{name}" in record
                        for phase in ("train", "train_eval", "valid")
                    ):
                        raise ValueError(
                            f"extra の {name} は指標と同じ名前です. "
                            "学習曲線で混ざるので, 別の名前にしてください"
                        )
                record.update(self.epoch_extra)

                # 判定するエポックでは, 監視する指標があることを確かめて
                # 同じ値の場合に比べる指標も取得してから記録
                value: Optional[float] = None
                tiebreak: Optional[float] = None
                if judged:
                    value = self.get_monitor_value(record, monitor)
                    tiebreak = self.get_tiebreak(record)
                self.history.append(record)

                # epoch ごとに学習率を更新する場合はここで更新
                # (判定しないエポックでは ReduceLROnPlateau だけ更新なし)
                if self.scheduler_interval == "epoch":
                    step_scheduler(self.scheduler, value, judged)

                # 判定するエポックだけ, ベストモデルと Early Stopping の状態を
                # 先に更新 (ベストモデルを選ばない設定では, 改善を判定しない)
                improved = False
                should_stop = False
                if judged:
                    improved = self.trainer_cfg.select_best and (
                        self.checkpoint.update(
                            value, epoch, self.eval_model(), tiebreak
                        )
                    )
                    should_stop = (
                        self.early_stopping is not None
                        and self.early_stopping.step(value, tiebreak)
                    )

                # 登録した関数にエポックの結果を渡し, 止める要求があるかの確認
                # 外から止める要求 (STOP のファイル) と同じく, このエポックで停止
                # (判定しないエポックで止めた場合, ベストモデルは前の判定のまま)
                callback_stop = self.run_callbacks(
                    epoch, record, improved, should_stop, judged
                )
                file_stop = self.run_status.stop_requested()
                stop_requested = file_stop or callback_stop

                # last.pt は決めた間隔と, 最後のエポックと止めるエポックに保存
                # 保存するものがある場合だけ, 更新した状態をまとめて保存
                write_last = (
                    epoch % every == 0 or epoch == self.end_epoch
                    or should_stop or stop_requested
                )
                if self.checkpoint.needs_save(improved, write_last):
                    self.checkpoint.save(
                        self.state_dict(epoch, monitor), improved, write_last
                    )

                # エポックの結果を記録して表示
                summary = ", ".join(
                    f"{key}={val:.6g}" for key, val in record.items()
                    if key != "epoch"
                )
                self.logger.info(f"エポック {epoch}: {summary}")
                pbar.set_postfix_str(summary)
                pbar.update(1)

                # 外から読める状態の更新
                self.run_status.update(
                    completed_epochs=len(self.history.records),
                    best_epoch=self.checkpoint.best_epoch or None,
                    best_value=self.checkpoint.best_value,
                    early_stopped_epoch=epoch if should_stop else None
                )

                # 改善しなくなった場合は学習の停止
                if should_stop:
                    self.logger.info(f"Early Stopping (エポック {epoch})")
                    pbar.write(
                        f"Early Stopping (エポック {epoch}), "
                        f"ベストモデルのエポック: {self.checkpoint.best_epoch}"
                    )
                    break

                # 外から止める要求があった場合は, このエポックで学習の停止
                # 予定の最後のエポックの場合は全て学習したので完了扱い
                if stop_requested and epoch < self.end_epoch:
                    self.stopped_by_request = True
                    source = (
                        "中断の要求" if file_stop else "登録した関数の要求"
                    )
                    message = f"{source}で学習を止めました (エポック {epoch})"
                    self.logger.info(message)
                    pbar.write(message)
                    break

        # 設定に応じてベストモデルを戻して結果を保存
        self.finish(time.time() - start_time)
        self.fit_finished = True
        history = self.history

        return history
