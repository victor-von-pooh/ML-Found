from collections import UserDict
from pathlib import Path
from typing import Any, Callable
import warnings

import pytest
import torch
from torch import nn

from ml_found.experiment.setup import Experiment
from ml_found.training.callbacks import (
    is_improved_with_tiebreak, load_checkpoint
)
from ml_found.training.optim import requires_closure
from ml_found.training.registry import (
    describe_optimizers, register_optimizer, unregister_optimizer
)
from ml_found.training.trainer import Trainer
from ml_found.utils.batch import get_batch_size
from test_trainer import make_experiment, make_loader, regression_step


def test_restore_best_false(tmp_path: Path) -> None:
    """
    restore_best を false にすると最後の重みが残ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 損失が大きいほど良いとして, ベストを最初のエポックに設定
    experiment = make_experiment(
        tmp_path,
        "trainer:\n  epochs: 4\n  mode: max\n  restore_best: false\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader(), make_loader())
    assert trainer.checkpoint.best_epoch == 1

    # final_model.pt は last.pt (エポック 4) のモデルと一致し, best.pt とは不一致
    final = torch.load(experiment.run_dir / "final_model.pt")
    last = load_checkpoint(str(experiment.run_dir / "last.pt"), "cpu")
    best = load_checkpoint(str(experiment.run_dir / "best.pt"), "cpu")
    assert all(torch.equal(v, last["model"][k]) for k, v in final.items())
    assert not all(torch.equal(v, best["model"][k]) for k, v in final.items())


def test_every_n_epochs(tmp_path: Path) -> None:
    """
    last.pt の保存の間隔と, 最後のエポックで必ず保存されることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 3 エポックごとの保存でも, 最後のエポック 4 の状態を保存
    experiment = make_experiment(
        tmp_path,
        "trainer:\n  epochs: 4\n  checkpoint:\n    every_n_epochs: 3\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader(), make_loader())
    last = load_checkpoint(str(experiment.run_dir / "last.pt"), "cpu")
    assert last["epoch"] == 4

    # 0 以下の間隔は出力先を作る前にエラー
    with pytest.raises(ValueError, match="every_n_epochs"):
        make_experiment(
            tmp_path / "zero",
            "trainer:\n  checkpoint:\n    every_n_epochs: 0\n"
        )
    assert not (tmp_path / "zero" / "outputs").exists()


def input_grad_step(model: nn.Module, batch: tuple, phase: str) -> dict:
    """
    入力に対する微分を損失に使う step の関数

    Parameters
    ----------
    model: nn.Module
        学習するモデル
    batch: tuple
        x と y
    phase: str
        train または valid

    Returns
    ----------
    outputs: dict
        予測の誤差と, 入力に対する微分を 2 に近づける項の和
    """
    # 入力に対する微分を自動微分で計算
    x, y = batch
    x = x.clone().requires_grad_(True)
    pred = model(x)
    dx = torch.autograd.grad(pred.sum(), x, create_graph=True)[0]

    # 予測の誤差と, 微分を 2 に近づける項の和を損失に設定
    mse = nn.MSELoss()
    outputs = {"loss": mse(pred, y) + mse(dx, torch.full_like(dx, 2.0))}

    return outputs


def test_valid_grad(tmp_path: Path) -> None:
    """
    valid_grad で検証時にも入力に対する微分を使えることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 既定では検証時に勾配が止まるので失敗
    off = make_experiment(tmp_path / "off", "trainer:\n  epochs: 1\n")
    trainer = Trainer(nn.Linear(1, 1), off, input_grad_step)
    with pytest.raises(RuntimeError):
        trainer.fit(make_loader(), make_loader())

    # valid_grad を有効にすると検証まで実行可能
    on = make_experiment(
        tmp_path / "on", "trainer:\n  epochs: 1\n  valid_grad: true\n"
    )
    trainer = Trainer(nn.Linear(1, 1), on, input_grad_step)
    history = trainer.fit(make_loader(), make_loader())
    assert history.get("valid/loss")[0] is not None


def noisy_step(model: nn.Module, batch: tuple, phase: str) -> dict:
    """
    損失に乱数を含む step の関数

    Parameters
    ----------
    model: nn.Module
        学習するモデル
    batch: tuple
        x と y
    phase: str
        train または valid

    Returns
    ----------
    outputs: dict
        入力に雑音を加えた平均二乗誤差と, 加えた雑音の平均
    """
    # 入力に乱数の雑音を加えて誤差を計算
    x, y = batch
    noise = torch.randn_like(x)
    mse = nn.MSELoss()
    outputs = {
        "loss": mse(model(x + 0.1 * noise), y), "noise": noise.mean()
    }

    return outputs


def test_valid_seed(tmp_path: Path) -> None:
    """
    valid_seed で検証の乱数が固定され, 学習の乱数が変わらないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 検証の乱数を固定すると, 毎エポック同じ雑音
    fixed = make_experiment(
        tmp_path / "fixed", "trainer:\n  epochs: 3\n  valid_seed: 7\n"
    )
    trainer = Trainer(nn.Linear(1, 1), fixed, noisy_step)
    history = trainer.fit(make_loader(), make_loader())
    noises = history.get("valid/noise")
    assert noises[0] == noises[1] == noises[2]

    # 検証が学習の乱数を消費しないので, 検証しない学習と学習の記録が一致
    alone = make_experiment(tmp_path / "alone", "trainer:\n  epochs: 3\n")
    reference = Trainer(nn.Linear(1, 1), alone, noisy_step)
    reference_history = reference.fit(make_loader())
    assert history.get("train/loss") == reference_history.get("train/loss")


def test_log_grad_norm(tmp_path: Path) -> None:
    """
    log_grad_norm で勾配のノルムが記録されることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 学習の指標にだけ grad_norm を追加
    experiment = make_experiment(
        tmp_path, "trainer:\n  epochs: 2\n  log_grad_norm: true\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    history = trainer.fit(make_loader(), make_loader())
    assert all(v > 0 for v in history.get("train/grad_norm"))
    assert "valid/grad_norm" not in history.keys()


def test_lbfgs_closure(tmp_path: Path) -> None:
    """
    closure が必要な LBFGS を config だけで使えることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # LBFGS は closure が必要と判定され, 1 次関数をほぼ正確に学習
    experiment = make_experiment(
        tmp_path,
        "optimizer:\n  name: LBFGS\n  params:\n    lr: 1.0\n"
        "trainer:\n  epochs: 3\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    assert trainer.use_closure
    history = trainer.fit([next(iter(make_loader(batch_size=64)))])
    assert history.get("train/loss")[-1] < 1e-6
    adam = torch.optim.Adam([nn.Parameter(torch.ones(1))])
    assert not requires_closure(adam)

    # 自作のクラスは, closure に既定値があってもクラスの属性で宣言できる
    class DeclaredOptimizer(torch.optim.SGD):
        # closure が必要と宣言した Optimizer
        requires_closure = True

    declared = DeclaredOptimizer([nn.Parameter(torch.ones(1))], lr=0.1)
    assert requires_closure(declared)

    # 登録簿の仕様にも closure が必要かを持つ
    specs = {spec["name"]: spec for spec in describe_optimizers()}
    assert specs["LBFGS"]["requires_closure"] is True
    assert specs["Adam"]["requires_closure"] is False


class ManualTrainer(Trainer):
    def step(self, batch: Any, phase: str) -> dict:
        """
        学習時は step の中で自分でパラメータを更新する関数

        Parameters
        ----------
        batch: Any
            x と y
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            勾配の流れない平均二乗誤差
        """
        # 損失を求め, 学習時は自分で逆伝播して更新
        outputs = regression_step(self.model, batch, phase)
        loss = outputs["loss"]
        if phase == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # ML-Found には勾配の流れない損失を返す
        outputs["loss"] = loss.detach()

        return outputs


def test_manual_optimization(tmp_path: Path) -> None:
    """
    manual_optimization で ML-Found が更新を行わないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 勾配の流れない損失でもエラーにならず, 更新回数はカウント
    experiment = make_experiment(
        tmp_path, "trainer:\n  epochs: 2\n  manual_optimization: true\n"
    )
    trainer = ManualTrainer(nn.Linear(1, 1), experiment)
    history = trainer.fit(make_loader(n=64, batch_size=16))
    assert trainer.global_step == 8
    assert history.get("train/loss")[1] < history.get("train/loss")[0]


class ExtraStateTrainer(Trainer):
    def __init__(self, model: nn.Module, experiment: Experiment):
        # 親クラスのコンストラクタを呼び出す
        super(ExtraStateTrainer, self).__init__(
            model, experiment, regression_step
        )

        # チェックポイントに加える値を保存
        self.counter = torch.zeros(1)

    def update_parameters(self, loss: torch.Tensor) -> None:
        """
        更新のたびに値を 1 増やす関数

        Parameters
        ----------
        loss: torch.Tensor
            1 バッチ分の損失

        Returns
        ----------
        None
        """
        # 元の手順で更新してから数を増加
        super(ExtraStateTrainer, self).update_parameters(loss)
        self.counter += 1

    def extra_state_dict(self) -> dict:
        """
        チェックポイントに数を加える関数

        Parameters
        ----------
        None

        Returns
        ----------
        state: dict
            更新の数
        """
        # 数を複製して受け渡し
        state = {"counter": self.counter.clone()}

        return state

    def load_extra_state_dict(self, state: dict) -> None:
        """
        チェックポイントから数を戻す関数

        Parameters
        ----------
        state: dict
            extra_state_dict で取得した状態

        Returns
        ----------
        None
        """
        # 数の復元
        self.counter = state["counter"].clone()


def test_extra_state(tmp_path: Path) -> None:
    """
    extra_state_dict の状態が再開で戻ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 2 エポック学習して 8 回更新
    first = make_experiment(tmp_path / "first", "trainer:\n  epochs: 2\n")
    trainer = ExtraStateTrainer(nn.Linear(1, 1), first)
    trainer.fit(make_loader(n=64, batch_size=16))
    assert trainer.counter.item() == 8

    # 再開して 3 エポック目まで学習すると 12 回
    path = first.run_dir / "last.pt"
    second = make_experiment(
        tmp_path / "second", f"trainer:\n  epochs: 3\n  resume_from: {path}\n"
    )
    resumed = ExtraStateTrainer(nn.Linear(1, 1), second)
    resumed.fit(make_loader(n=64, batch_size=16))
    assert resumed.counter.item() == 12


def test_checks_before_run_dir(tmp_path: Path) -> None:
    """
    設定の誤りが出力先を作る前に見つかることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 存在しない再開元はエラーで, 出力先は未作成
    with pytest.raises(FileNotFoundError, match="resume_from"):
        make_experiment(
            tmp_path / "resume", "trainer:\n  resume_from: missing/last.pt\n"
        )
    assert not (tmp_path / "resume" / "outputs").exists()

    # ReduceLROnPlateau の向きの食い違いもエラーで, 出力先は未作成
    with pytest.raises(ValueError, match="mode"):
        make_experiment(
            tmp_path / "plateau",
            "scheduler:\n  name: ReduceLROnPlateau\n  params:\n"
            "    mode: max\n"
        )
    assert not (tmp_path / "plateau" / "outputs").exists()


def test_duplicate_metric(tmp_path: Path) -> None:
    """
    on_epoch_end と step の指標の名前が重なるとエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # step と同じ名前の mae を on_epoch_end でも返す Trainer
    class DuplicateTrainer(Trainer):
        def on_epoch_end(self, phase: str) -> dict:
            """
            step と同じ名前の指標を返す関数

            Parameters
            ----------
            phase: str
                train または valid

            Returns
            ----------
            metrics: dict
                step と同じ名前の指標
            """
            # 重なる名前を返却
            metrics = {"mae": 0.0}

            return metrics

    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 1\n")
    trainer = DuplicateTrainer(nn.Linear(1, 1), experiment, regression_step)
    with pytest.raises(ValueError, match="重複"):
        trainer.fit(make_loader())


def test_error_logged(tmp_path: Path) -> None:
    """
    学習中のエラーが train.log に記録されることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 損失を返さない step で学習を始めるとエラー
    def broken_step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        損失を返さない step の関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: tuple
            x と y
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            loss を含まない辞書
        """
        # loss の無い辞書を返却
        outputs = {"mae": torch.zeros(1)}

        return outputs

    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 1\n")
    trainer = Trainer(nn.Linear(1, 1), experiment, broken_step)
    with pytest.raises(ValueError):
        trainer.fit(make_loader())
    log = (experiment.run_dir / "train.log").read_text(encoding="utf-8")
    assert "エラーで学習が止まりました" in log


def test_mapping_batch_size() -> None:
    """
    dict の子クラスでない Mapping のバッチからサンプル数を推定できることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # BatchEncoding と同じく UserDict のバッチ
    batch = UserDict({"input_ids": torch.zeros(5, 3), "labels": torch.ones(5)})
    assert get_batch_size(batch) == 5


def test_scheduler_param_warning(tmp_path: Path) -> None:
    """
    再開元と config で Scheduler の引数が違う場合に警告することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # T_max を epochs に合わせた学習を 1 エポックで止める
    text = (
        "scheduler:\n  name: CosineAnnealingLR\n  params:\n"
        "    T_max: ${trainer.epochs}\n"
    )
    first = make_experiment(
        tmp_path / "first", text + "trainer:\n  epochs: 1\n"
    )
    Trainer(nn.Linear(1, 1), first, regression_step).fit(make_loader())

    # epochs を増やして再開すると, 再開元の T_max が使われるので警告
    path = first.run_dir / "last.pt"
    second = make_experiment(
        tmp_path / "second",
        text + f"trainer:\n  epochs: 2\n  resume_from: {path}\n"
    )
    trainer = Trainer(nn.Linear(1, 1), second, regression_step)
    with pytest.warns(UserWarning, match="T_max"):
        trainer.fit(make_loader())


def test_final_model_on_cpu(tmp_path: Path) -> None:
    """
    final_model.pt が cpu のテンソルで保存されることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # map_location を付けずに読み込んでも cpu のテンソル
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 1\n")
    Trainer(nn.Linear(1, 1), experiment, regression_step).fit(make_loader())
    final = torch.load(experiment.run_dir / "final_model.pt")
    assert all(value.device.type == "cpu" for value in final.values())


def test_scheduler_param_no_false_warning(tmp_path: Path) -> None:
    """
    引数を別の型で持つ Scheduler で, 再開時に誤って警告しないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # milestones を Counter で持つ MultiStepLR で 1 エポック学習
    text = (
        "scheduler:\n  name: MultiStepLR\n  params:\n"
        "    milestones: [2, 3]\n"
    )
    first = make_experiment(
        tmp_path / "first", text + "trainer:\n  epochs: 1\n"
    )
    Trainer(nn.Linear(1, 1), first, regression_step).fit(make_loader())

    # 同じ config で再開すると, 引数は変わらないので警告なし
    path = first.run_dir / "last.pt"
    second = make_experiment(
        tmp_path / "second",
        text + f"trainer:\n  epochs: 2\n  resume_from: {path}\n"
    )
    trainer = Trainer(nn.Linear(1, 1), second, regression_step)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trainer.fit(make_loader())


class CrashTrainer(Trainer):
    def on_epoch_start(self, phase: str) -> None:
        """
        エポック 5 の学習を始めるときに止まる関数

        Parameters
        ----------
        phase: str
            学習時は train, 検証時は valid

        Returns
        ----------
        None
        """
        # 学習の途中で止まった状況を再現
        if self.epoch == 5:
            raise RuntimeError("エポック 5 で停止")


def test_resume_skips_newer_best(tmp_path: Path) -> None:
    """
    last.pt より新しい best.pt を再開時に写さないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 3 エポックごとの保存で, エポック 5 で止まると last.pt はエポック 3,
    # best.pt は損失が下がり続けるのでエポック 4 の状態
    first = make_experiment(
        tmp_path / "first",
        "trainer:\n  epochs: 6\n  checkpoint:\n    every_n_epochs: 3\n"
    )
    trainer = CrashTrainer(nn.Linear(1, 1), first, regression_step)
    with pytest.raises(RuntimeError):
        trainer.fit(make_loader(), make_loader())
    last = load_checkpoint(str(first.run_dir / "last.pt"), "cpu")
    best = load_checkpoint(str(first.run_dir / "best.pt"), "cpu")
    assert (last["epoch"], best["epoch"]) == (3, 4)

    # エポック 3 から再開すると, 再開で捨てるエポック 4 の best.pt は未コピー
    path = first.run_dir / "last.pt"
    second = make_experiment(
        tmp_path / "second",
        f"trainer:\n  epochs: 3\n  resume_from: {path}\n"
    )
    Trainer(nn.Linear(1, 1), second, regression_step).fit(
        make_loader(), make_loader()
    )
    assert not (second.run_dir / "best.pt").exists()

    # ベストのエポックが一致する best.pt は従来どおりコピー
    third_first = make_experiment(
        tmp_path / "third_first", "trainer:\n  epochs: 2\n"
    )
    Trainer(nn.Linear(1, 1), third_first, regression_step).fit(
        make_loader(), make_loader()
    )
    path = third_first.run_dir / "last.pt"
    third = make_experiment(
        tmp_path / "third", f"trainer:\n  epochs: 2\n  resume_from: {path}\n"
    )
    Trainer(nn.Linear(1, 1), third, regression_step).fit(
        make_loader(), make_loader()
    )
    assert (third.run_dir / "best.pt").exists()


def test_manual_with_clip_error(tmp_path: Path) -> None:
    """
    manual_optimization と勾配のクリップは同時に使えないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 自分で更新する設定に勾配のクリップを加えると, 出力先を作る前にエラー
    with pytest.raises(ValueError, match="manual_optimization"):
        make_experiment(
            tmp_path,
            "trainer:\n  manual_optimization: true\n  grad_clip_norm: 1.0\n"
        )
    assert not (tmp_path / "outputs").exists()


def test_log_grad_norm_keeps_grads(tmp_path: Path) -> None:
    """
    クリップせずに勾配のノルムを記録しても, 勾配が変わらないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 1 つの勾配だけ inf にしても, 他の勾配は NaN にならず元の値のまま
    experiment = make_experiment(tmp_path, "trainer:\n  log_grad_norm: true\n")
    model = nn.Linear(1, 1)
    trainer = Trainer(model, experiment, regression_step)
    model.weight.grad = torch.full_like(model.weight, float("inf"))
    model.bias.grad = torch.ones_like(model.bias)
    grad_norm = trainer.clip_gradients()
    assert grad_norm == float("inf")
    assert torch.equal(model.bias.grad, torch.ones_like(model.bias))


def test_in_closure_flag(tmp_path: Path) -> None:
    """
    closure の中の step だけ in_closure が立つことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # step が呼ばれたときの in_closure を記録する Trainer
    class FlagTrainer(Trainer):
        def step(self, batch: Any, phase: str) -> dict:
            """
            in_closure を記録してから損失を求める関数

            Parameters
            ----------
            batch: Any
                x と y
            phase: str
                train または valid

            Returns
            ----------
            outputs: dict
                平均二乗誤差と平均絶対誤差
            """
            # 印を記録して元の step で計算
            self.flags.append(self.in_closure)
            outputs = super(FlagTrainer, self).step(batch, phase)

            return outputs

    experiment = make_experiment(
        tmp_path,
        "optimizer:\n  name: LBFGS\n  params:\n    lr: 1.0\n"
        "    max_iter: 3\n"
        "trainer:\n  epochs: 1\n"
    )
    trainer = FlagTrainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.flags = []
    trainer.fit([next(iter(make_loader(batch_size=64)))])

    # 最初の 1 回は closure の外, 残りは closure の中で, 終わると印を戻す
    assert trainer.flags[0] is False
    assert all(trainer.flags[1:]) and len(trainer.flags) > 1
    assert trainer.in_closure is False


class ScriptedMetricTrainer(Trainer):
    def on_epoch_end(self, phase: str) -> dict:
        """
        検証の後にエポックごとに決めておいた指標を返す関数

        Parameters
        ----------
        phase: str
            train または valid

        Returns
        ----------
        metrics: dict
            検証の場合は score と aux, 学習の場合は空の辞書
        """
        # 学習の場合は指標なし
        metrics = {}
        if phase != "valid":
            return metrics

        # 今のエポックの値を返却
        metrics = {
            "score": self.scores[self.epoch - 1],
            "aux": self.auxes[self.epoch - 1]
        }

        return metrics


def run_scripted(
    tmp_path: Path, text: str, scores: list[float], auxes: list[float]
) -> ScriptedMetricTrainer:
    """
    決めておいた指標で学習する関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    text: str
        default に加える YAML の内容
    scores: list[float]
        エポックごとの valid/score
    auxes: list[float]
        エポックごとの valid/aux

    Returns
    ----------
    trainer: ScriptedMetricTrainer
        学習を終えた Trainer
    """
    # score を大きい方が良い指標として監視して学習
    base = f"  epochs: {len(scores)}\n  monitor: valid/score\n  mode: max\n"
    experiment = make_experiment(tmp_path, "trainer:\n" + base + text)
    trainer = ScriptedMetricTrainer(
        nn.Linear(1, 1), experiment, regression_step
    )
    trainer.scores = scores
    trainer.auxes = auxes
    trainer.fit(make_loader(), make_loader())

    return trainer


def test_monitor_tiebreak(tmp_path: Path) -> None:
    """
    監視する指標が同じ値の場合に, 別の指標で比べることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 設定しない場合は, 最初に 0.9 に達したエポック 2 がベスト
    scores = [0.5, 0.9, 0.9, 0.9]
    auxes = [1.0, 0.8, 0.6, 0.7]
    plain = run_scripted(tmp_path / "plain", "", scores, auxes)
    assert plain.checkpoint.best_epoch == 2

    # aux が小さい方を良いとして比べると, 同じ 0.9 のうち aux が最小のエポック 3
    tie = "  monitor_tiebreak: valid/aux\n  monitor_tiebreak_mode: min\n"
    tied = run_scripted(tmp_path / "tied", tie, scores, auxes)
    assert tied.checkpoint.best_epoch == 3


def test_monitor_tiebreak_early_stopping(tmp_path: Path) -> None:
    """
    同じ値が続いても比べる指標が改善していれば Early Stopping しないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # score が同じ値のまま aux だけ下がり続ける学習
    scores = [0.9] * 5
    auxes = [1.0, 0.9, 0.8, 0.7, 0.6]
    stop = "  early_stopping:\n    use: true\n    patience: 2\n"

    # 設定しない場合は 2 エポック改善なしでエポック 3 で停止
    plain = run_scripted(tmp_path / "plain", stop, scores, auxes)
    assert len(plain.history.records) == 3

    # aux で比べると毎エポック改善とみなし, 最後まで学習
    tie = "  monitor_tiebreak: valid/aux\n"
    tied = run_scripted(tmp_path / "tied", stop + tie, scores, auxes)
    assert len(tied.history.records) == 5
    assert tied.checkpoint.best_epoch == 5


def test_progress_bar_batches(tmp_path: Path) -> None:
    """
    バッチ単位の進捗バーを出しても学習できることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 進捗バーを表示する設定で 1 エポック学習
    experiment = make_experiment(
        tmp_path, "trainer:\n  epochs: 1\n  progress_bar_batches: true\n"
    )
    experiment.cfg.trainer.progress_bar = True
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    history = trainer.fit(make_loader(), make_loader())
    assert len(history.records) == 1


def test_tiebreak_tolerance_and_min_delta() -> None:
    """
    丸め誤差の範囲の同じ値と, min_delta 未満の改善でも比べることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 丸め誤差でわずかに小さい値でも, 比べる指標が良ければ改善
    best = 0.9666666666666667
    near = 0.9666666547457378
    assert is_improved_with_tiebreak(near, best, "max", 0.0, 0.5, 0.6, "min")
    assert not is_improved_with_tiebreak(
        near, best, "max", 0.0, 0.7, 0.6, "min"
    )

    # 丸め誤差でわずかに大きい値でも, 比べる指標が悪ければ改善なし
    above = 0.9666666766666667
    assert not is_improved_with_tiebreak(
        above, best, "max", 0.0, 0.7, 0.6, "min"
    )
    assert is_improved_with_tiebreak(above, best, "max", 0.0, None, 0.6, "min")

    # min_delta 未満の改善は, 比べる指標が良ければ改善
    assert is_improved_with_tiebreak(0.905, 0.9, "max", 0.01, 0.5, 0.6, "min")
    assert not is_improved_with_tiebreak(
        0.905, 0.9, "max", 0.01, None, 0.6, "min"
    )

    # 明らかに悪い値は, 比べる指標が良くても改善なし
    assert not is_improved_with_tiebreak(
        0.8, 0.9, "max", 0.0, 0.1, 0.6, "min"
    )


def test_tiebreak_errors(tmp_path: Path) -> None:
    """
    比べる指標の名前の誤りと, 再開時の設定の違いがエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 存在しない指標の名前はエラーで, 記録は追加なし
    wrong = make_experiment(
        tmp_path / "wrong",
        "trainer:\n  epochs: 1\n  monitor_tiebreak: valid/missing\n"
    )
    trainer = Trainer(nn.Linear(1, 1), wrong, regression_step)
    with pytest.raises(ValueError, match="valid/missing"):
        trainer.fit(make_loader(), make_loader())
    assert len(trainer.history.records) == 0

    # 比べる指標を変えて再開するとエラー
    first = make_experiment(
        tmp_path / "first",
        "trainer:\n  epochs: 1\n  monitor_tiebreak: valid/mae\n"
    )
    Trainer(nn.Linear(1, 1), first, regression_step).fit(
        make_loader(), make_loader()
    )
    path = first.run_dir / "last.pt"
    second = make_experiment(
        tmp_path / "second",
        "trainer:\n  epochs: 2\n  monitor_tiebreak: train/mae\n"
        f"  resume_from: {path}\n"
    )
    resumed = Trainer(nn.Linear(1, 1), second, regression_step)
    with pytest.raises(ValueError, match="同じ値の場合に比べる指標"):
        resumed.fit(make_loader(), make_loader())

    # 使っていなかった比べる指標を再開時に加えてもエラー
    plain = make_experiment(tmp_path / "plain", "trainer:\n  epochs: 1\n")
    Trainer(nn.Linear(1, 1), plain, regression_step).fit(
        make_loader(), make_loader()
    )
    added = make_experiment(
        tmp_path / "added",
        "trainer:\n  epochs: 2\n  monitor_tiebreak: valid/mae\n"
        f"  resume_from: {plain.run_dir / 'last.pt'}\n"
    )
    resumed = Trainer(nn.Linear(1, 1), added, regression_step)
    with pytest.raises(ValueError, match="同じ値の場合に比べる指標"):
        resumed.fit(make_loader(), make_loader())

    # 比べる指標の向きの誤りは出力先を作る前にエラー
    with pytest.raises(ValueError, match="monitor_tiebreak_mode"):
        make_experiment(
            tmp_path / "mode", "trainer:\n  monitor_tiebreak_mode: low\n"
        )


class RandomSearch(torch.optim.Optimizer):
    # 勾配を使わない Optimizer (損失が下がる場合だけ乱数で動かした値を採用)
    uses_gradients = False

    def __init__(self, params: Any, lr: float = 0.1):
        # 親クラスのコンストラクタを呼び出す
        super(RandomSearch, self).__init__(params, {"lr": lr})

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """
        パラメータを乱数で動かし, 損失が下がった場合だけ残す関数

        Parameters
        ----------
        closure: Callable[[], torch.Tensor]
            今のパラメータの損失を求める関数

        Returns
        ----------
        loss: torch.Tensor
            更新した後の損失
        """
        # 今の損失を求めてから, 全てのパラメータを動かして比較
        loss = closure()
        saved = []
        for group in self.param_groups:
            for param in group["params"]:
                saved.append(param.detach().clone())
                param.add_(torch.randn_like(param) * group["lr"])
        new_loss = closure()

        # 損失が下がらなければ元に戻す
        if new_loss >= loss:
            params = [
                param for group in self.param_groups
                for param in group["params"]
            ]
            for param, value in zip(params, saved):
                param.copy_(value)
            new_loss = loss

        return new_loss


def test_gradient_free_optimizer(tmp_path: Path) -> None:
    """
    勾配を使わないと宣言した Optimizer では, 逆伝播をせず損失だけを求める
    closure を渡して学習することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 1 次関数を乱数の探索で学習
    experiment = make_experiment(
        tmp_path / "ok", "trainer:\n  epochs: 20\n"
    )
    model = nn.Linear(1, 1)
    trainer = Trainer(
        model, experiment, regression_step, RandomSearch(model.parameters())
    )
    assert not trainer.use_gradients
    history = trainer.fit(make_loader())
    losses = history.get("train/loss")
    assert losses[-1] < losses[0]
    assert all(param.grad is None for param in model.parameters())
    log = (experiment.run_dir / "train.log").read_text(encoding="utf-8")
    assert "勾配を使わず" in log

    # 検証で勾配を使う設定でも, 学習では勾配を記録しない
    grad_flags = []

    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        勾配を記録しているかを残して平均二乗誤差を求める関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: tuple
            x と y
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            平均二乗誤差
        """
        # フェーズと勾配の記録の有無を保存
        grad_flags.append((phase, torch.is_grad_enabled()))
        outputs = regression_step(model, batch, phase)

        return outputs

    experiment = make_experiment(
        tmp_path / "valid_grad",
        "trainer:\n  epochs: 1\n  valid_grad: true\n"
    )
    model = nn.Linear(1, 1)
    trainer = Trainer(
        model, experiment, step, RandomSearch(model.parameters())
    )
    trainer.fit(make_loader(), make_loader())
    assert any(flag for phase, flag in grad_flags if phase == "valid")
    assert not any(flag for phase, flag in grad_flags if phase == "train")

    # config で指定した場合は, 勾配を使う設定との組み合わせを出力先を作る前に
    # エラー
    register_optimizer(RandomSearch)
    try:
        with pytest.raises(ValueError, match="勾配を使わない"):
            make_experiment(
                tmp_path / "config",
                "optimizer:\n  name: RandomSearch\n"
                "trainer:\n  log_grad_norm: true\n"
            )
        assert not (tmp_path / "config" / "outputs").exists()
    finally:
        unregister_optimizer("RandomSearch")

    # 勾配を使う設定とは一緒に使えない
    experiment = make_experiment(
        tmp_path / "clip", "trainer:\n  grad_clip_norm: 1.0\n"
    )
    model = nn.Linear(1, 1)
    with pytest.raises(ValueError, match="勾配を使わない"):
        Trainer(
            model, experiment, regression_step,
            RandomSearch(model.parameters())
        )
