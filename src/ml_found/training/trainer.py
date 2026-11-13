import copy
import math
from pathlib import Path
import shutil
import time
import warnings
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

from omegaconf import OmegaConf
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
from ml_found.training.history import (
    History, LrSteps, MetricTracker, to_float
)
from ml_found.training.optim import (
    auto_params, build_optimizer, build_scheduler, check_scheduler,
    missing_data_size_params, requires_closure, step_scheduler, to_kwargs
)
from ml_found.training.status import RunStatus
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

# Scheduler が Optimizer のパラメータグループに書き込む引数の値
SCHEDULER_GROUP_KEYS = ("max_lr", "min_lr", "base_momentum", "max_momentum")


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
        cfg = experiment.cfg
        status = RunStatus(experiment.run_dir, experiment.logger)
        status.start(
            planned_epochs=cfg.trainer.epochs, start_epoch=1,
            monitor=cfg.trainer.monitor, mode=cfg.trainer.mode,
            seed=cfg.experiment.seed, device=str(experiment.device)
        )
        status.finish(
            "stopped" if interrupted else "failed",
            error=f"{type(error).__name__}: {error}"
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

        # 1 ステップの計算を行う関数を保存
        self.step_fn = step_fn

        # Optimizer が渡されていない場合は config から作成
        if optimizer is None:
            optimizer = build_optimizer(
                experiment.cfg.optimizer, self.model.parameters()
            )
        self.optimizer = optimizer

        # Scheduler が渡されていない場合は config から作成し, どちらかを保存
        # データの量で決まる引数 (OneCycleLR の steps_per_epoch など) が
        # 無い場合は, 学習用データが分かる fit の初めまで作成を保留
        self.scheduler_from_config = scheduler is None
        self.auto_values: dict[str, int] = {}
        self.pending_scheduler = scheduler is None and bool(
            missing_data_size_params(experiment.cfg.scheduler)
        )
        # 予定の総ステップ数 (OneCycleLR の総ステップ数との食い違いの確認に使用)
        self.planned_total_steps: Optional[int] = None
        if scheduler is None and not self.pending_scheduler:
            scheduler = build_scheduler(
                experiment.cfg.scheduler, self.optimizer,
                experiment.cfg.trainer.mode
            )
        elif (
            scheduler is not None and scheduler.optimizer is not self.optimizer
        ):
            raise ValueError(
                "scheduler は Trainer が使う optimizer で作ってください"
            )
        self.scheduler = scheduler

        # 学習率を更新する間隔を保存し, Scheduler との組み合わせの確認
        self.scheduler_interval = experiment.cfg.scheduler.interval
        check_scheduler(
            self.scheduler, self.scheduler_interval,
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

        # エポックごとの記録と, 最初のエポックを保存
        self.history = History()
        self.start_epoch = 1

        # エポックの終わりに呼ぶ関数 (add_callback で追加)
        self.callbacks: list[Callable[[dict], Optional[bool]]] = []

        # step ごとに学習率を変える場合の, 更新ごとの学習率の記録
        self.lr_steps = LrSteps()

        # step から参照できるように, 今のエポックとパラメータの更新回数を保存
        self.epoch = 0
        self.global_step = 0

        # LBFGS のように損失を計算し直す closure が必要な Optimizer かどうかを保存
        self.use_closure = requires_closure(self.optimizer)

        # closure で使う学習時のバッチと, 最後に求めた勾配のノルムの入れ物を用意
        self.current_batch: Any = None
        self.last_grad_norm: Optional[float] = None

        # closure の中で step を呼んでいるかどうかの印を用意
        # closure は step を何度も呼ぶので, 予測を貯める処理などを飛ばすのに使用
        self.in_closure = False

        # 外から読める実験の状態と進み具合を書く仕組みの作成
        self.run_status = RunStatus(experiment.run_dir, self.logger)

        # 外からの要求で学習を止めたかどうかの印を用意
        self.stopped_by_request = False

        # step の extra で返した, 平均せずに記録だけする値の入れ物を用意
        self.epoch_extra: dict[str, float] = {}

        # バッチのサンプル数を推定できなかったことを 1 回だけ警告するための印
        self.warned_batch_size = False

        # エポックの途中で止めたときに戻す, 学習を始めた時点の重みと,
        # 戻せずに途中までの重みが残ったかどうかの印を用意
        self.start_state: Optional[dict] = None
        self.partial_weights = False

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

    def add_callback(self, fn: Callable[[dict], Optional[bool]]) -> None:
        """
        エポックの終わりに呼ぶ関数を加える関数

        関数はエポックごとに, epoch (エポック), record (記録の写し),
        improved (ベストモデルを更新したか), best_epoch, best_value,
        early_stopping (Early Stopping で止めるか) の辞書を受け取る. 真の値
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
        self, epoch: int, record: dict, improved: bool, should_stop: bool
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
                "early_stopping": bool(should_stop)
            }
            # numpy や torch の真偽値も, 真の値なら止める要求として扱う
            if fn(info):
                stop = True

        return stop

    def eval_model(self) -> nn.Module:
        """
        検証, ベストモデルの判定, final_model.pt に使うモデルを返す関数

        EMA の重みで検証する場合などに継承して上書きする. 返すモデルは
        self.model と同じ名前の重み (state_dict のキー) を持つ必要がある.
        step を上書きする場合は, 学習以外 (phase が valid か train_eval) で
        このモデルを使う

        Parameters
        ----------
        None

        Returns
        ----------
        model: nn.Module
            評価に使うモデル, 既定は学習するモデル (self.model) そのもの
        """
        # 既定は学習するモデルをそのまま使用
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

    def clip_gradients(self) -> Optional[float]:
        """
        勾配のノルムを求め, 上限が設定されていればクリップする関数

        Parameters
        ----------
        None

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

        # Optimizer が更新するパラメータだけを対象に設定
        params = [
            p for group in self.optimizer.param_groups for p in group["params"]
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
                    outputs = self.step(self.current_batch, "train")
                    loss = self.check_outputs(outputs, "train")
                    loss.backward()
            finally:
                self.in_closure = False

            # 勾配のノルムを求めてクリップ
            self.last_grad_norm = self.clip_gradients()

            return loss

        return closure

    def update_parameters(self, loss: torch.Tensor) -> None:
        """
        損失から勾配を求めてパラメータを更新する関数

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

        # closure が必要な Optimizer は損失を計算し直す関数を渡して更新
        elif self.use_closure:
            self.optimizer.step(self.make_closure())

        # 勾配法では勾配を初期化して逆伝播し, クリップしてから更新
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.last_grad_norm = self.clip_gradients()
            self.optimizer.step()

        # パラメータの更新回数のカウント
        self.global_step += 1

        # step ごとに学習率を更新する場合は, この更新に使った学習率を記録して
        # から更新 (エポックの記録の lr はエポックを始めた時点の値だけ)
        if self.scheduler_interval == "step":
            if self.scheduler is not None:
                self.lr_steps.append(
                    self.global_step, self.epoch,
                    float(self.optimizer.param_groups[0]["lr"])
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

        # 学習時と, 検証時にも勾配を使う設定の場合だけ勾配を計算
        grad_enabled = train or self.trainer_cfg.valid_grad

        # 進み具合を外に知らせるため, 分かる場合は 1 エポックのバッチの数を取得
        total_batches: Optional[int] = None
        try:
            total_batches = len(loader)
        except (TypeError, NotImplementedError):
            pass
        # 1 エポックが短い学習の邪魔にならないように, フェーズの始めも間隔を確保
        self.run_status.progress(
            phase, self.epoch, self.trainer_cfg.epochs, 0, total_batches
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

                # 決めた範囲だけ勾配を計算
                with torch.set_grad_enabled(grad_enabled):
                    outputs = self.step(batch, phase)

                # 戻り値を確かめて, 学習時は closure 用にバッチを保存して更新
                loss = self.check_outputs(outputs, phase)
                if train:
                    self.current_batch = batch
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
                    phase, self.epoch, self.trainer_cfg.epochs, n_batches,
                    total_batches
                )

                # エポックの途中で止める要求 (STOP に now) の確認
                # 最後のバッチの後は, エポックの区切りで止める処理に任せる
                is_last = total_batches is not None and (
                    n_batches >= total_batches
                )
                if not is_last and self.run_status.stop_now_requested():
                    raise StopWithinEpoch()

        # フェーズの終わりの進み具合の書き出し (間隔の中なら次の機会に)
        self.run_status.progress(
            phase, self.epoch, self.trainer_cfg.epochs, n_batches,
            total_batches
        )

        # 学習時のバッチを検証の間まで残さないように解放
        self.current_batch = None

        # バッチが 1 つも無い場合は平均を求められないのでエラー
        if n_batches == 0:
            raise ValueError(f"{phase} の loader が空です")

        # バッチの平均とエポック全体から求めた指標を集約
        epoch_metrics = tracker.compute()
        for key, value in self.on_epoch_end(phase).items():
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
        if self.last_grad_norm is not None:
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
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                None if self.scheduler is None
                else self.scheduler.state_dict()
            ),
            "history": self.history.state_dict(),
            "lr_steps": self.lr_steps.state_dict(),
            "early_stopping": (
                None if self.early_stopping is None
                else self.early_stopping.state_dict()
            ),
            "checkpoint": self.checkpoint.state_dict(),
            "rng": get_rng_state(),
            "extra": self.extra_state_dict()
        }

        return state

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

        # モデル, Optimizer, Scheduler, 記録の復元
        # Scheduler が Optimizer に書き込む値 (OneCycleLR の max_lr など) も
        # config と比べるため, 読み込む前の値を保持
        self.model.load_state_dict(state["model"])
        fresh_groups = scheduler_group_values(self.optimizer)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state["scheduler"] is not None:
            fresh_state = self.scheduler.state_dict()
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
        self.history.load_state_dict(state["history"])
        self.lr_steps = LrSteps()
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

        # 継承したクラスが加えた状態の復元
        self.load_extra_state_dict(state.get("extra", {}))

        # 続きの乱数を使えるように乱数の状態の復元
        # 戻せない外部ライブラリの乱数は train.log にも記録
        for message in set_rng_state(state["rng"]):
            self.logger.warning(message)
            warnings.warn(message, stacklevel=2)

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
            save_checkpoint(state, run_dir / "last.pt")

        # 続きのエポックとパラメータの更新回数から開始
        self.start_epoch = state["epoch"] + 1
        self.epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.logger.info(f"再開元: {path} (エポック {state['epoch']})")

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
        # ベストモデルの重みを eval_model から取る場合は, 今のモデルと違うので
        # best.pt から
        separate = self.checkpoint.separate_best_model
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
            # 今のモデルの重み (eval_model が別の場合は best_model に保存)
            weights = None
            if best["epoch"] == best_epoch:
                stored = (best.get("checkpoint") or {}).get("best_model")
                if best.get("weights_only") or not separate:
                    weights = best["model"]
                elif stored is not None:
                    weights = stored
            if weights is not None:
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
                save_checkpoint({
                    "weights_only": True, "epoch": best["epoch"],
                    "monitor": best["monitor"],
                    "best_value": self.checkpoint.best_value,
                    "model": best["model"]
                }, target)
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
        self.logger.info(f"Optimizer:\n\n{self.optimizer}\n")
        scheduler_name = (
            "なし" if self.scheduler is None
            else type(self.scheduler).__name__
        )
        self.logger.info(f"Scheduler: {scheduler_name}")
        if self.scheduler is not None:
            raw_state = self.scheduler.state_dict()
            scheduler_state = {
                key: value for key, value in raw_state.items()
                if not key.startswith("_")
            }
            self.logger.info(f"Scheduler の状態: {scheduler_state}")
        self.logger.info(f"監視する指標: {monitor} ({self.trainer_cfg.mode})")
        if self.trainer_cfg.monitor_tiebreak is not None:
            self.logger.info(
                "同じ値の場合に比べる指標: "
                f"{self.trainer_cfg.monitor_tiebreak} "
                f"({self.trainer_cfg.monitor_tiebreak_mode})"
            )

        # 評価に使うモデルが別の場合は記録
        if self.eval_model() is not self.model:
            self.logger.info(
                "評価に使うモデル: eval_model が返すモデル "
                f"({type(self.eval_model()).__name__})"
            )

        # 更新の手順を記録
        if self.trainer_cfg.manual_optimization:
            self.logger.info("更新の手順: step の中で自分で更新")
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
        diffs = [
            f"{name} (再開元 {getattr(self.scheduler, name)}, 補った値 "
            f"{value})"
            for name, value in self.auto_values.items()
            if getattr(self.scheduler, name, value) != value
        ]
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
        if (
            self.trainer_cfg.restore_best and self.trainer_cfg.select_best
            and self.checkpoint.best_model_state is not None
        ):
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

        return history

    def build_scheduler_from_data(self, train_loader: Iterable) -> None:
        """
        データの量で決まる Scheduler の引数を学習用データから補って作る関数

        Parameters
        ----------
        train_loader: Iterable
            学習用のバッチを返すもの, len() でバッチ数を取得

        Returns
        ----------
        None
        """
        # 1 エポックで Scheduler を進める回数 (step ごとならバッチ数) と,
        # エポック数 (trainer.epochs) の取得
        # auto と書いた引数は学習全体のステップ数 (エポック数 × 1 エポックの回数)
        cfg = self.experiment.cfg
        auto = set(auto_params(cfg.scheduler))
        values = {}
        for name in missing_data_size_params(cfg.scheduler):
            if name == "epochs":
                values[name] = self.trainer_cfg.epochs
            elif self.scheduler_interval == "epoch":
                values[name] = 1
            else:
                try:
                    values[name] = len(train_loader)
                    if values[name] == 0:
                        raise ValueError("train の loader が空です")
                except TypeError as error:
                    hint = (
                        "値" if name in auto
                        else "total_steps か steps_per_epoch"
                    )
                    raise ValueError(
                        f"{cfg.scheduler.name} の {name} を補えません. "
                        "train_loader が len() を持たないので, "
                        f"scheduler.params に{hint}を書いてください"
                    ) from error

        for name in auto:
            values[name] = self.trainer_cfg.epochs * values[name]
        self.auto_values = {name: values[name] for name in auto}

        # config に書き込んで作成し, 補った値が残るように config.yaml も更新
        for name, value in values.items():
            cfg.scheduler.params[name] = value
        self.scheduler = build_scheduler(
            cfg.scheduler, self.optimizer, self.trainer_cfg.mode
        )
        self.pending_scheduler = False

        # 学習中に読まれても壊れた中身が見えないように, 一時ファイルから置き換え
        path = self.experiment.run_dir / "config.yaml"
        temp = path.with_name("config.yaml.tmp")
        OmegaConf.save(cfg, temp, resolve=True)
        temp.replace(path)
        filled = ", ".join(f"{name}={value}" for name, value in values.items())
        self.logger.info(
            f"{cfg.scheduler.name} の引数を学習用データから補いました: {filled}"
        )

    def training_state(self) -> dict:
        """
        エポックの途中で止めたときに戻すための, 学習の状態の写しを作る関数

        Parameters
        ----------
        None

        Returns
        ----------
        state: dict
            モデル, Optimizer, Scheduler, 更新回数, 乱数, 継承したクラスが
            加えた状態 (テンソルは cpu の写し)
        """
        # チェックポイントと同じ名前で, 戻すのに必要なものだけを写す
        state = clone_to_cpu({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                None if self.scheduler is None
                else self.scheduler.state_dict()
            ),
            "global_step": self.global_step,
            "rng": get_rng_state(),
            "extra": self.extra_state_dict()
        })

        return state

    def load_training_state(self, state: dict) -> None:
        """
        training_state か last.pt の状態に, 学習の状態を戻す関数

        Parameters
        ----------
        state: dict
            戻す状態 (model, optimizer, scheduler, global_step, rng, extra)

        Returns
        ----------
        None
        """
        # モデル, Optimizer, Scheduler, 更新回数の復元
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = int(state.get("global_step", self.global_step))

        # 乱数と, 継承したクラスが加えた状態 (EMA など) の復元
        for message in set_rng_state(state["rng"]):
            self.logger.warning(message)
        self.load_extra_state_dict(state.get("extra", {}))

    def discard_epoch(self, epoch: int) -> None:
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

        Returns
        ----------
        None
        """
        self.stopped_by_request = True
        self.current_batch = None
        self.lr_steps.drop_from_epoch(epoch)
        message = f"中断の要求でエポック {epoch} の途中で学習を止めました"

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
            if last.get("epoch") == epoch - 1:
                source = last
                message += f". 状態はエポック {epoch - 1} のものに戻しました"
        if source is not None:
            self.load_training_state(source)

            # 評価に使うモデルが別で, その状態を extra_state_dict で保存して
            # いない場合は, そのモデルの途中の更新は戻せない
            if self.eval_model() is not self.model and not source.get("extra"):
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

    def plan_total_steps(self, train_loader: Iterable) -> None:
        """
        OneCycleLR を使う場合に, 予定の総ステップ数を求めて保存する関数

        Parameters
        ----------
        train_loader: Iterable
            学習用のバッチを返すもの, len() が無い場合は求めない

        Returns
        ----------
        None
        """
        self.planned_total_steps = None
        if not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
            return
        if self.scheduler_interval == "epoch":
            per_epoch = 1
        else:
            try:
                per_epoch = len(train_loader)
            except TypeError:
                return
        self.planned_total_steps = self.trainer_cfg.epochs * per_epoch

    def check_total_steps(self) -> None:
        """
        OneCycleLR の総ステップ数が, 予定の総ステップ数と合うかを確かめる関数

        総ステップ数を超えて進めると学習の途中で PyTorch がエラーを出すので,
        学習を始める前にエラーにする. 少ない場合は学習率が下がりきらないので警告.
        再開した場合は再開元の総ステップ数が使われるので, trainer.epochs や
        バッチ数を変えて再開した場合も確認

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        total = getattr(self.scheduler, "total_steps", None)
        planned = self.planned_total_steps
        if total is None or planned is None or total == planned:
            return

        message = (
            f"{type(self.scheduler).__name__} の総ステップ数 ({total}) と, "
            f"trainer.epochs と学習用データのバッチ数から求めた予定の総ステップ数 "
            f"({planned}) が違います. 再開した場合は再開元の総ステップ数が使われ"
            "ます (trainer.resume_scheduler: config で config の値で作り直せ"
            "ます)"
        )
        if planned > total:
            raise ValueError(message)
        message += " (学習率は最後まで下がりきりません)"
        self.logger.warning(message)
        warnings.warn(message, stacklevel=2)

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

        # 再開の前に失敗しても分かるように, 学習を始めたことを書き出し
        self.run_status.start(
            planned_epochs=self.trainer_cfg.epochs, start_epoch=1,
            monitor=monitor, mode=self.trainer_cfg.mode,
            seed=self.experiment.cfg.experiment.seed,
            device=str(self.device),
            select_best=self.trainer_cfg.select_best
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
        self.plan_total_steps(train_loader)
        if self.trainer_cfg.resume_from is None:
            self.check_total_steps()

        # 再開する場合はチェックポイントから状態の復元
        if self.trainer_cfg.resume_from is not None:
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

        # Early Stopping で既に止めた学習を再開した場合は学習なし
        epochs = range(self.start_epoch, self.trainer_cfg.epochs + 1)
        if self.early_stopping is not None and self.early_stopping.stopped:
            self.logger.info("Early Stopping で止めた学習のため学習しません")
            epochs = range(0)

        # 学習の設定を記録
        self.log_settings(monitor)

        # エポックの途中で止めたときに戻せるように, 始めた時点の状態を cpu に
        # 退避 (以降のエポックでは last.pt から戻す)
        self.partial_weights = False
        self.start_state = self.training_state()

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
                lr = self.optimizer.param_groups[0]["lr"]

                # 学習し, 検証用のデータがある場合は検証
                # エポックの途中で止める要求があれば, このエポックを捨てて終了
                record = {"epoch": epoch}
                self.epoch_extra = {}
                try:
                    record.update(self.run_epoch(train_loader, train=True))
                    if self.trainer_cfg.eval_train:
                        record.update(
                            self.run_valid_epoch(train_loader, "train_eval")
                        )
                    if valid_loader is not None:
                        record.update(self.run_valid_epoch(valid_loader))
                except StopWithinEpoch:
                    self.discard_epoch(epoch)
                    pbar.write(
                        f"中断の要求でエポック {epoch} の途中で学習を止めました"
                    )
                    break
                record["lr"] = lr

                # step の extra で返した値は, 平均せずにエポックの最後の値を記録
                # 指標と同じ名前は学習曲線とダッシュボードで混ざるので禁止
                for name in self.epoch_extra:
                    if any(
                        f"{phase}/{name}" in record
                        for phase in ("train", "train_eval", "valid")
                    ):
                        raise ValueError(
                            f"extra の {name} は指標と同じ名前です. "
                            "学習曲線で混ざるので, 別の名前にしてください"
                        )
                record.update(self.epoch_extra)

                # 監視する指標があることを確かめてから記録
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

                # 監視する指標が同じ値の場合に比べる指標の取得
                tiebreak = self.get_tiebreak(record)
                self.history.append(record)

                # epoch ごとに学習率を更新する場合はここで更新
                if self.scheduler_interval == "epoch":
                    step_scheduler(self.scheduler, value)

                # ベストモデルと Early Stopping の状態を先に更新
                # (ベストモデルを選ばない設定では, 改善を判定しない)
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
                callback_stop = self.run_callbacks(
                    epoch, record, improved, should_stop
                )
                file_stop = self.run_status.stop_requested()
                stop_requested = file_stop or callback_stop

                # last.pt は決めた間隔と, 最後のエポックと止めるエポックに保存
                # 保存するものがある場合だけ, 更新した状態をまとめて保存
                write_last = (
                    epoch % every == 0 or epoch == self.trainer_cfg.epochs
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
                    best_epoch=self.checkpoint.best_epoch,
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
                if stop_requested and epoch < self.trainer_cfg.epochs:
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
        history = self.history

        return history
