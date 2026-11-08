import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Optional

import torch
from torch import nn, optim
from tqdm import tqdm

from ml_found.experiment.seed import get_rng_state, set_rng_state
from ml_found.experiment.setup import Experiment
from ml_found.training.callbacks import (
    Checkpoint, EarlyStopping, load_checkpoint, save_checkpoint
)
from ml_found.training.history import History, MetricTracker, to_float
from ml_found.training.optim import (
    build_optimizer, build_scheduler, step_scheduler
)
from ml_found.utils.batch import get_batch_size, move_to_device
from ml_found.utils.plotting import plot_history


class Trainer:
    def __init__(
        self, model: nn.Module, experiment: Experiment,
        step_fn: Optional[Callable[[nn.Module, Any], dict]] = None,
        optimizer: Optional[optim.Optimizer] = None
    ):
        # 実験, 学習の config, ロガー, デバイスを保存する
        self.experiment = experiment
        self.cfg = experiment.cfg.trainer
        self.logger = experiment.logger
        self.device = experiment.device

        # モデルをデバイスに移して保存する
        self.model = model.to(self.device)

        # 1 ステップの計算を行う関数を保存する
        self.step_fn = step_fn

        # Optimizer が渡されていない場合は config から作る
        if optimizer is None:
            optimizer = build_optimizer(
                experiment.cfg.optimizer, self.model.parameters()
            )
        self.optimizer = optimizer

        # Scheduler を config から作り, 更新する間隔を保存する
        self.scheduler = build_scheduler(
            experiment.cfg.scheduler, self.optimizer, self.cfg.mode
        )
        self.scheduler_interval = experiment.cfg.scheduler.interval

        # Early Stopping を使う場合は作る
        self.early_stopping = None
        if self.cfg.early_stopping.use:
            self.early_stopping = EarlyStopping(
                patience=self.cfg.early_stopping.patience,
                min_delta=self.cfg.early_stopping.min_delta,
                mode=self.cfg.mode
            )

        # ベストモデルと最後のモデルを保存する仕組みを作る
        self.checkpoint = Checkpoint(
            experiment.run_dir, mode=self.cfg.mode,
            save_best=self.cfg.checkpoint.save_best,
            save_last=self.cfg.checkpoint.save_last
        )

        # エポックごとの記録と, 最初のエポックを保存する
        self.history = History()
        self.start_epoch = 1

    def training_step(self, batch: Any) -> dict:
        """
        学習時に 1 バッチ分の損失と指標を求める関数

        Parameters
        ----------
        batch: Any
            デバイスに移したバッチ

        Returns
        -------
        outputs: dict
            loss は必須, その他の指標と batch_size は任意
        """
        # 関数が渡されていない場合は継承して上書きしてもらう
        if self.step_fn is None:
            raise NotImplementedError(
                "step_fn を渡すか training_step を上書きしてください"
            )

        # 渡された関数で計算する
        outputs = self.step_fn(self.model, batch)

        return outputs

    def validation_step(self, batch: Any) -> dict:
        """
        検証時に 1 バッチ分の損失と指標を求める関数

        Parameters
        ----------
        batch: Any
            デバイスに移したバッチ

        Returns
        -------
        outputs: dict
            loss は必須, その他の指標と batch_size は任意
        """
        # 学習時と同じ計算をする, 違う計算をする場合は継承して上書きする
        outputs = self.training_step(batch)

        return outputs

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
        -------
        loss: torch.Tensor
            1 バッチ分の損失
        """
        # 辞書で loss を含んでいることを確かめる
        if not isinstance(outputs, dict) or "loss" not in outputs:
            raise ValueError("step は loss を含む辞書を返してください")

        # 学習曲線の画像のファイル名に使うので / を含む名前を禁止し,
        # バッチごとの平均を求めるので指標は 1 つの値に限る
        for key, value in outputs.items():
            if "/" in key:
                raise ValueError(f"指標の名前に / は使えません: {key}")
            if isinstance(value, torch.Tensor) and value.numel() != 1:
                raise ValueError(
                    f"指標 {key} は 1 つの値にしてください: "
                    f"形状 {tuple(value.shape)}"
                )

        # 逆伝播できるように損失はテンソルで返してもらう
        loss = outputs["loss"]
        if not isinstance(loss, torch.Tensor):
            raise ValueError("loss は torch.Tensor で返してください")

        # 損失が発散していないことを確かめる
        if not math.isfinite(to_float(loss)):
            raise ValueError(f"{phase} の loss が {to_float(loss)} になりました")

        return loss

    def update_parameters(self, loss: torch.Tensor) -> None:
        """
        損失から勾配を求めてパラメータを更新する関数

        Parameters
        ----------
        loss: torch.Tensor
            1 バッチ分の損失

        Returns
        -------
        None
        """
        # 勾配を初期化して逆伝播を計算する
        self.optimizer.zero_grad()
        loss.backward()

        # 勾配が大きくなりすぎないようにノルムの上限で切る
        if self.cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip_norm
            )

        # パラメータを更新する
        self.optimizer.step()

        # step ごとに学習率を更新する場合はここで更新する
        if self.scheduler_interval == "step":
            step_scheduler(self.scheduler)

    def run_epoch(self, loader: Iterable, train: bool) -> dict[str, float]:
        """
        1 エポック分の学習または検証を行う関数

        Parameters
        ----------
        loader: Iterable
            バッチを順に返すもの, DataLoader や [data] のようなリスト
        train: bool
            学習する場合は True, 検証する場合は False

        Returns
        -------
        metrics: dict[str, float]
            train/ または valid/ を付けた指標ごとの平均
        """
        # 学習モードと推論モードを切り替える
        self.model.train(train)
        phase = "train" if train else "valid"
        tracker = MetricTracker()

        # バッチごとに計算する
        n_batches = 0
        for batch in loader:
            batch = move_to_device(batch, self.device)
            n_batches += 1

            # 学習時だけ勾配を計算する
            with torch.set_grad_enabled(train):
                if train:
                    outputs = self.training_step(batch)
                else:
                    outputs = self.validation_step(batch)

            # 戻り値を確かめて, 学習時はパラメータを更新する
            loss = self.check_outputs(outputs, phase)
            if train:
                self.update_parameters(loss)

            # サンプル数で重み付けして指標を足し込む
            metrics = dict(outputs)
            n = metrics.pop("batch_size", None)
            if n is None:
                n = get_batch_size(batch)
            tracker.update(metrics, int(n))

        # バッチが 1 つも無い場合は平均を求められないのでエラーにする
        if n_batches == 0:
            raise ValueError(f"{phase} の loader が空です")

        # 接頭辞を付けて平均を返す
        metrics = {
            f"{phase}/{key}": value for key, value in tracker.compute().items()
        }

        return metrics

    def get_monitor(self, has_valid: bool) -> str:
        """
        ベストモデルの判定に使う指標の名前を決める関数

        Parameters
        ----------
        has_valid: bool
            検証用のデータがあるかどうか

        Returns
        -------
        monitor: str
            指標の名前
        """
        # 指定されている場合はそのまま使い, 無い場合は検証用のデータの有無で決める
        if self.cfg.monitor is not None:
            monitor = self.cfg.monitor
        elif has_valid:
            monitor = "valid/loss"
        else:
            monitor = "train/loss"

        return monitor

    def state_dict(self, epoch: int) -> dict:
        """
        チェックポイントとして保存する内容をまとめる関数

        Parameters
        ----------
        epoch: int
            今回のエポック

        Returns
        -------
        state: dict
            モデル, Optimizer, Scheduler, 記録, 乱数などの状態
        """
        # 学習を再開するのに必要な状態を全てまとめる
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                None if self.scheduler is None
                else self.scheduler.state_dict()
            ),
            "history": self.history.state_dict(),
            "early_stopping": (
                None if self.early_stopping is None
                else self.early_stopping.state_dict()
            ),
            "checkpoint": self.checkpoint.state_dict(),
            "rng": get_rng_state()
        }

        return state

    def resume(self, path: str) -> None:
        """
        チェックポイントから学習を再開できる状態に戻す関数

        Parameters
        ----------
        path: str
            チェックポイントのパス

        Returns
        -------
        None
        """
        # チェックポイントを読み込む
        state = load_checkpoint(path, self.device)

        # モデル, Optimizer, Scheduler, 記録を戻す
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.history.load_state_dict(state["history"])

        # Early Stopping の状態を戻す
        if (
            self.early_stopping is not None
            and state["early_stopping"] is not None
        ):
            self.early_stopping.load_state_dict(state["early_stopping"])

        # 同じディレクトリに best.pt がある場合はベストモデルの重みも戻す
        best_path = Path(path).with_name("best.pt")
        best_state = None
        if best_path.exists():
            best_state = load_checkpoint(str(best_path), self.device)
        self.checkpoint.load_state_dict(
            state["checkpoint"],
            None if best_state is None else best_state["model"]
        )

        # 続きの乱数を使えるように乱数の状態を戻す
        set_rng_state(state["rng"])

        # 新しい保存先からも続けて再開できるように, 読み込んだ状態を保存する
        run_dir = self.experiment.run_dir
        if best_state is not None and self.cfg.checkpoint.save_best:
            save_checkpoint(best_state, run_dir / "best.pt")
        if self.cfg.checkpoint.save_last:
            save_checkpoint(state, run_dir / "last.pt")

        # 続きのエポックから始める
        self.start_epoch = state["epoch"] + 1
        self.logger.info(f"再開元: {path} (エポック {state['epoch']})")

    def log_settings(self, monitor: str) -> None:
        """
        学習の設定をログに記録する関数

        Parameters
        ----------
        monitor: str
            ベストモデルの判定に使う指標の名前

        Returns
        -------
        None
        """
        # モデルの構造とパラメータ数を記録する
        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"モデルの構造:\n\n{self.model}\n")
        self.logger.info(f"パラメータ数: {n_params}")

        # Optimizer, Scheduler, 監視する指標を記録する
        self.logger.info(f"Optimizer:\n\n{self.optimizer}\n")
        scheduler_name = (
            "なし" if self.scheduler is None
            else type(self.scheduler).__name__
        )
        self.logger.info(f"Scheduler: {scheduler_name}")
        self.logger.info(f"監視する指標: {monitor} ({self.cfg.mode})")

    def finish(self, elapsed_time: float) -> None:
        """
        学習後にベストモデルを戻し, 結果を保存する関数

        Parameters
        ----------
        elapsed_time: float
            学習にかかった秒数

        Returns
        -------
        None
        """
        # ベストモデルの重みに戻し, 推論で使いやすいように重みだけを保存する
        run_dir = self.experiment.run_dir
        self.checkpoint.restore_best(self.model)
        torch.save(self.model.state_dict(), run_dir / "final_model.pt")

        # 記録と学習曲線を保存する
        self.history.save_csv(str(run_dir / "history.csv"))
        plot_history(self.history.records, str(run_dir))

        # 結果をログに記録する
        self.logger.info("学習完了")
        self.logger.info(f"ベストモデルのエポック: {self.checkpoint.best_epoch}")
        self.logger.info(f"ベストモデルの指標: {self.checkpoint.best_value}")
        self.logger.info(f"学習時間: {elapsed_time:.2f}秒")

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
        -------
        history: History
            エポックごとの記録
        """
        # 監視する指標を決める
        monitor = self.get_monitor(valid_loader is not None)

        # 再開する場合はチェックポイントから状態を戻す
        if self.cfg.resume_from is not None:
            self.resume(self.cfg.resume_from)

        # Early Stopping で既に止めた学習を再開した場合は学習しない
        epochs = range(self.start_epoch, self.cfg.epochs + 1)
        if self.early_stopping is not None and self.early_stopping.stopped:
            self.logger.info("Early Stopping で止めた学習のため学習しません")
            epochs = range(0)

        # 学習の設定を記録する
        self.log_settings(monitor)

        # エポックごとに学習する
        start_time = time.time()
        with tqdm(epochs, disable=not self.cfg.progress_bar) as pbar:
            for epoch in pbar:
                pbar.set_description(f"Epoch {epoch}")

                # このエポックを始めた時点の学習率を控える
                lr = self.optimizer.param_groups[0]["lr"]

                # 学習し, 検証用のデータがある場合は検証する
                record = {"epoch": epoch}
                record.update(self.run_epoch(train_loader, train=True))
                if valid_loader is not None:
                    record.update(self.run_epoch(valid_loader, train=False))
                record["lr"] = lr

                # 監視する指標があることを確かめてから記録する
                if monitor not in record:
                    raise ValueError(
                        f"監視する指標 {monitor} がありません: "
                        f"{list(record.keys())}"
                    )
                value = record[monitor]
                self.history.append(record)

                # epoch ごとに学習率を更新する場合はここで更新する
                if self.scheduler_interval == "epoch":
                    step_scheduler(self.scheduler, value)

                # ベストモデルと Early Stopping の状態を先に更新する
                improved = self.checkpoint.update(value, epoch, self.model)
                should_stop = (
                    self.early_stopping is not None
                    and self.early_stopping.step(value)
                )

                # 更新した状態でチェックポイントを保存する
                self.checkpoint.save(self.state_dict(epoch), improved)

                # エポックの結果を記録して表示する
                summary = ", ".join(
                    f"{key}={val:.6g}" for key, val in record.items()
                    if key != "epoch"
                )
                self.logger.info(f"エポック {epoch}: {summary}")
                pbar.set_postfix_str(summary)

                # 改善しなくなった場合は学習を止める
                if should_stop:
                    self.logger.info(f"Early Stopping (エポック {epoch})")
                    pbar.write(
                        f"Early Stopping at epoch {epoch}. "
                        f"Best epoch: {self.checkpoint.best_epoch}"
                    )
                    break

        # ベストモデルを戻して結果を保存する
        self.finish(time.time() - start_time)

        return self.history
