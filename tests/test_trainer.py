import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ml_found.config.loader import load_config
from ml_found.experiment import register_rng, unregister_rng
from ml_found.experiment.setup import Experiment, start_experiment
from ml_found.training.callbacks import load_checkpoint
from ml_found.training.history import History
from ml_found.training.trainer import Trainer


def make_experiment(tmp_path: Path, text: str = "") -> Experiment:
    """
    テスト用の実験を開始する関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    text: str = ""
        default に加える YAML の内容

    Returns
    ----------
    experiment: Experiment
        cpu で進捗バーを出さない実験
    """
    # cpu で進捗バーを出さない default と空の experiment を用意
    base = (
        "experiment:\n"
        "  name: test\n"
        "  device: cpu\n"
        "  seed: 0\n"
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    default_path = tmp_path / "default.yaml"
    default_path.write_text(base + text, encoding="utf-8")
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text("", encoding="utf-8")

    # 実験を開始
    cfg, diff = load_config(str(default_path), str(experiment_path))
    cfg.trainer.progress_bar = False
    experiment = start_experiment(cfg, diff, str(tmp_path))

    return experiment


def make_loader(
    n: int = 64, batch_size: int = 16, shuffle: bool = False
) -> DataLoader:
    """
    y = 2x + 1 を学習するための DataLoader を作る関数

    Parameters
    ----------
    n: int = 64
        サンプル数
    batch_size: int = 16
        バッチサイズ
    shuffle: bool = False
        エポックごとにデータを並べ替えるかどうか

    Returns
    ----------
    loader: DataLoader
        x と y を返す DataLoader
    """
    # 1 次関数のデータの作成
    x = torch.linspace(-1, 1, n).unsqueeze(1)
    y = 2 * x + 1
    loader = DataLoader(
        TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle
    )

    return loader


def regression_step(model: nn.Module, batch: tuple, phase: str) -> dict:
    """
    平均二乗誤差を求める関数

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
        平均二乗誤差と平均絶対誤差
    """
    # 予測して誤差の計算
    x, y = batch
    pred = model(x)
    mse = nn.MSELoss()
    outputs = {"loss": mse(pred, y), "mae": (pred - y).abs().mean()}

    return outputs


def scripted_step(losses: list) -> Callable[[nn.Module, tuple, str], dict]:
    """
    呼ばれるたびに決めておいた損失を返す step を作る関数

    Parameters
    ----------
    losses: list
        順に返す損失の値

    Returns
    ----------
    step: Callable[[nn.Module, tuple, str], dict]
        決めておいた損失を返す step
    """
    # 呼ばれた回数のカウント
    values = iter(losses)

    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        決めておいた損失を返す関数

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
            決めておいた損失
        """
        # 勾配が 0 になるようにモデルの出力に 0 を掛けて加算
        outputs = {"loss": model(batch[0]).sum() * 0 + next(values)}

        return outputs

    return step


def test_fit_minibatch(tmp_path: Path) -> None:
    """
    ミニバッチ学習で損失が下がり, 結果が保存されることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 線形回帰を学習
    experiment = make_experiment(
        tmp_path,
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
        "trainer:\n  epochs: 30\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    history = trainer.fit(make_loader(), make_loader())
    losses = history.get("train/loss")
    run_dir = experiment.run_dir

    assert len(history.records) == 30
    assert losses[-1] < losses[0] * 0.01
    assert history.keys() == [
        "epoch", "train/loss", "train/mae", "valid/loss", "valid/mae", "lr"
    ]
    for name in [
        "best.pt", "last.pt", "final_model.pt", "history.csv",
        "loss_curve.png", "train.log"
    ]:
        assert (run_dir / name).exists()


def test_fit_fullbatch(tmp_path: Path) -> None:
    """
    全データを 1 度に流すフルバッチ学習ができることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # データ全体を 1 つの要素として受け渡し
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 5\n")
    x = torch.randn(32, 1)
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    history = trainer.fit([(x, 2 * x + 1)])

    assert len(history.records) == 5
    assert trainer.get_monitor(has_valid=False) == "train/loss"


def test_final_model_is_best(tmp_path: Path) -> None:
    """
    学習後のモデルがベストモデルの重みになっていることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 学習率が大きすぎて途中から悪化する設定で学習
    experiment = make_experiment(
        tmp_path,
        "optimizer:\n  name: SGD\n  params:\n    lr: 1.5\n"
        "trainer:\n  epochs: 10\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    history = trainer.fit(make_loader(batch_size=64))
    best = load_checkpoint(
        str(experiment.run_dir / "best.pt"), torch.device("cpu")
    )
    final = torch.load(experiment.run_dir / "final_model.pt")
    losses = history.get("train/loss")

    assert best["epoch"] == losses.index(min(losses)) + 1
    assert torch.equal(final["weight"], best["model"]["weight"])
    assert torch.equal(trainer.model.weight, best["model"]["weight"])


def test_early_stopping(tmp_path: Path) -> None:
    """
    改善しなくなると途中で学習が止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 学習率 0 で全く改善しない設定
    experiment = make_experiment(
        tmp_path,
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.0\n"
        "trainer:\n  epochs: 50\n"
        "  early_stopping:\n    use: true\n    patience: 3\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    history = trainer.fit(make_loader())

    assert len(history.records) == 4


def test_resume_matches_uninterrupted(tmp_path: Path) -> None:
    """
    途中で止めて再開しても続けて学習した場合と同じ結果になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # データを並べ替えながら 5 エポック続けて学習
    text = (
        "optimizer:\n  name: Adam\n  params:\n    lr: 0.05\n"
        "scheduler:\n  name: StepLR\n  params:\n    step_size: 2\n"
    )
    full = make_experiment(
        tmp_path / "full", text + "trainer:\n  epochs: 5\n"
    )
    full_history = Trainer(nn.Linear(1, 1), full, regression_step).fit(
        make_loader(shuffle=True)
    )

    # 3 エポックで止めてから 5 エポックまで再開
    first = make_experiment(
        tmp_path / "first", text + "trainer:\n  epochs: 3\n"
    )
    Trainer(nn.Linear(1, 1), first, regression_step).fit(
        make_loader(shuffle=True)
    )
    second = make_experiment(
        tmp_path / "second",
        text + "trainer:\n  epochs: 5\n"
        f"  resume_from: {first.run_dir / 'last.pt'}\n"
    )
    resumed_history = Trainer(nn.Linear(1, 1), second, regression_step).fit(
        make_loader(shuffle=True)
    )

    assert resumed_history.get("epoch") == [1, 2, 3, 4, 5]
    assert resumed_history.get("train/loss") == pytest.approx(
        full_history.get("train/loss")
    )
    assert resumed_history.get("lr") == pytest.approx(full_history.get("lr"))


def test_resume_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    相対パスで指定したチェックポイントをプロジェクトのディレクトリから辿ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        実行する場所を変えるために pytest が用意するもの

    Returns
    ----------
    None
    """
    # プロジェクトのディレクトリで 2 エポック学習
    project_dir = tmp_path / "project"
    first = make_experiment(project_dir, "trainer:\n  epochs: 2\n")
    Trainer(nn.Linear(1, 1), first, regression_step).fit(make_loader())
    relative_path = (first.run_dir / "last.pt").relative_to(project_dir)

    # プロジェクトの外に移動し, 相対パスで指定して再開
    monkeypatch.chdir(tmp_path)
    text = f"trainer:\n  epochs: 3\n  resume_from: {relative_path}\n"
    second = make_experiment(project_dir, text)
    history = Trainer(nn.Linear(1, 1), second, regression_step).fit(
        make_loader()
    )

    assert history.get("epoch") == [1, 2, 3]


def test_resume_keeps_best(tmp_path: Path) -> None:
    """
    再開前のベストモデルより悪いエポックをベストとみなさないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 損失が 3, 2, 1 と下がる 3 エポックを学習
    first = make_experiment(tmp_path / "first", "trainer:\n  epochs: 3\n")
    Trainer(nn.Linear(1, 1), first, scripted_step([3.0, 2.0, 1.0])).fit(
        [(torch.ones(1, 1),)]
    )

    # 損失が 1.5 に悪化する 4 エポック目を再開して学習
    text = (
        "trainer:\n  epochs: 4\n"
        f"  resume_from: {first.run_dir / 'last.pt'}\n"
    )
    second = make_experiment(tmp_path / "second", text)
    trainer = Trainer(nn.Linear(1, 1), second, scripted_step([1.5]))
    trainer.fit([(torch.ones(1, 1),)])
    best = load_checkpoint(
        str(second.run_dir / "best.pt"), torch.device("cpu")
    )

    assert trainer.checkpoint.best_epoch == 3
    assert trainer.checkpoint.best_value == 1.0
    assert best["epoch"] == 3


def test_resume_early_stopping(tmp_path: Path) -> None:
    """
    再開しても続けて学習した場合と同じエポックで止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 損失が変わらず, 2 回待つと 3 エポック目で止まる設定
    text = "  early_stopping:\n    use: true\n    patience: 2\n"

    # 2 エポックで止めてから再開
    first = make_experiment(
        tmp_path / "first", "trainer:\n  epochs: 2\n" + text
    )
    Trainer(nn.Linear(1, 1), first, scripted_step([1.0] * 2)).fit(
        [(torch.ones(1, 1),)]
    )
    second = make_experiment(
        tmp_path / "second",
        "trainer:\n  epochs: 10\n"
        f"  resume_from: {first.run_dir / 'last.pt'}\n" + text
    )
    history = Trainer(
        nn.Linear(1, 1), second, scripted_step([1.0] * 8)
    ).fit([(torch.ones(1, 1),)])

    assert history.get("epoch") == [1, 2, 3]

    # 止めた学習を再開しても学習しないことの確認
    third = make_experiment(
        tmp_path / "third",
        "trainer:\n  epochs: 10\n"
        f"  resume_from: {second.run_dir / 'last.pt'}\n" + text
    )
    history = Trainer(nn.Linear(1, 1), third, scripted_step([])).fit(
        [(torch.ones(1, 1),)]
    )

    assert history.get("epoch") == [1, 2, 3]
    assert (third.run_dir / "best.pt").exists()
    assert (third.run_dir / "last.pt").exists()


def test_resume_without_best_file(tmp_path: Path) -> None:
    """
    best.pt を保存しない設定で再開してもベストモデルに戻ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 学習率が大きすぎて途中から悪化する設定を best.pt を保存せずに学習
    text = (
        "optimizer:\n  name: SGD\n  params:\n    lr: 1.5\n"
        "trainer:\n  checkpoint:\n    save_best: false\n"
    )
    first = make_experiment(tmp_path / "first", text + "  epochs: 3\n")
    Trainer(nn.Linear(1, 1), first, regression_step).fit(
        make_loader(batch_size=64)
    )
    second = make_experiment(
        tmp_path / "second",
        text + f"  epochs: 8\n  resume_from: {first.run_dir / 'last.pt'}\n"
    )
    trainer = Trainer(nn.Linear(1, 1), second, regression_step)
    history = trainer.fit(make_loader(batch_size=64))
    losses = history.get("train/loss")
    final = torch.load(second.run_dir / "final_model.pt")

    assert trainer.checkpoint.best_epoch == losses.index(min(losses)) + 1
    assert torch.equal(
        final["weight"], trainer.checkpoint.best_model_state["weight"]
    )


def test_resume_chain_keeps_best_weights(tmp_path: Path) -> None:
    """
    再開を 2 回つないでもベストモデルの重みを失わないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # best.pt を残さずに, 損失が 3, 1, 2 となる 3 エポックを学習
    first = make_experiment(
        tmp_path / "first",
        "trainer:\n  epochs: 3\n  checkpoint:\n    save_best: false\n"
    )
    Trainer(nn.Linear(1, 1), first, scripted_step([3.0, 1.0, 2.0])).fit(
        [(torch.ones(1, 1),)]
    )

    # best.pt を残す設定で, 改善しない 4 エポック目を再開して学習
    second = make_experiment(
        tmp_path / "second",
        "trainer:\n  epochs: 4\n"
        f"  resume_from: {first.run_dir / 'last.pt'}\n"
    )
    Trainer(nn.Linear(1, 1), second, scripted_step([2.5])).fit(
        [(torch.ones(1, 1),)]
    )

    # さらに 5 エポック目を再開して学習
    third = make_experiment(
        tmp_path / "third",
        "trainer:\n  epochs: 5\n"
        f"  resume_from: {second.run_dir / 'last.pt'}\n"
    )
    trainer = Trainer(nn.Linear(1, 1), third, scripted_step([2.6]))
    trainer.fit([(torch.ones(1, 1),)])

    assert trainer.checkpoint.best_epoch == 2
    assert trainer.checkpoint.best_model_state is not None


def test_resume_with_different_monitor(tmp_path: Path) -> None:
    """
    監視する指標の向きを変えて再開するとエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 小さい方が良い向きで学習
    first = make_experiment(tmp_path / "first", "trainer:\n  epochs: 1\n")
    Trainer(nn.Linear(1, 1), first, regression_step).fit(make_loader())

    # 大きい方が良い向きに変えて再開
    second = make_experiment(
        tmp_path / "second",
        "trainer:\n  epochs: 2\n  mode: max\n"
        f"  resume_from: {first.run_dir / 'last.pt'}\n"
    )
    trainer = Trainer(nn.Linear(1, 1), second, regression_step)

    with pytest.raises(ValueError, match="監視する指標が違います"):
        trainer.fit(make_loader())


def test_nan_monitor(tmp_path: Path) -> None:
    """
    監視する指標が nan になるとエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 監視する指標だけが nan になる step を用意
    def nan_auc_step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        損失と nan の指標を返す関数

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
            平均二乗誤差と nan の指標
        """
        # 平均二乗誤差に nan の指標の追加
        outputs = regression_step(model, batch, phase)
        outputs["auc"] = float("nan")

        return outputs

    experiment = make_experiment(
        tmp_path, "trainer:\n  epochs: 2\n  monitor: train/auc\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, nan_auc_step)

    with pytest.raises(ValueError, match="train/auc"):
        trainer.fit(make_loader())


def test_scheduler_and_clipping(tmp_path: Path) -> None:
    """
    Scheduler と勾配クリッピングを使って学習できることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # step ごとに学習率を下げ, 勾配を切る設定
    experiment = make_experiment(
        tmp_path,
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
        "scheduler:\n  name: StepLR\n  interval: step\n"
        "  params:\n    step_size: 4\n    gamma: 0.5\n"
        "trainer:\n  epochs: 2\n  grad_clip_norm: 1.0\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    history = trainer.fit(make_loader(n=64, batch_size=16))

    assert history.get("lr") == pytest.approx([0.1, 0.05])

    # 更新ごとの学習率 (4 回ごとに半分) とエポックを記録
    with open(experiment.run_dir / "lr_steps.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(row["step"]) for row in rows] == list(range(1, 9))
    assert [int(row["epoch"]) for row in rows] == [1] * 4 + [2] * 4
    assert [float(row["lr"]) for row in rows] == pytest.approx(
        [0.1] * 4 + [0.05] * 4
    )
    assert (experiment.run_dir / "lr_steps.png").exists()


def test_lr_steps_only_for_step_interval(tmp_path: Path) -> None:
    """
    エポックごとに学習率を変える場合と Scheduler が無い場合は,
    更新ごとの学習率を保存しないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    for name, text in (
        ("epoch", "scheduler:\n  name: StepLR\n  params:\n    step_size: 1\n"),
        ("none", "scheduler:\n  interval: step\n"),
    ):
        experiment = make_experiment(
            tmp_path / name, text + "trainer:\n  epochs: 2\n"
        )
        Trainer(nn.Linear(1, 1), experiment, regression_step).fit(
            make_loader()
        )
        assert not (experiment.run_dir / "lr_steps.csv").exists()


def test_resume_keeps_lr_steps(tmp_path: Path) -> None:
    """
    再開しても更新ごとの学習率が続けて学習した場合と同じになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    text = (
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
        "scheduler:\n  name: StepLR\n  interval: step\n"
        "  params:\n    step_size: 3\n    gamma: 0.5\n"
    )

    def read(run_dir: Path) -> list[dict]:
        """
        保存した更新ごとの学習率を読み込む関数

        Parameters
        ----------
        run_dir: Path
            実験のディレクトリ

        Returns
        ----------
        rows: list[dict]
            lr_steps.csv の行
        """
        with open(run_dir / "lr_steps.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        return rows

    full = make_experiment(tmp_path / "full", text + "trainer:\n  epochs: 4\n")
    Trainer(nn.Linear(1, 1), full, regression_step).fit(make_loader())

    first = make_experiment(
        tmp_path / "first", text + "trainer:\n  epochs: 2\n"
    )
    Trainer(nn.Linear(1, 1), first, regression_step).fit(make_loader())
    second = make_experiment(
        tmp_path / "second",
        text + "trainer:\n  epochs: 4\n"
        f"  resume_from: {first.run_dir / 'last.pt'}\n"
    )
    Trainer(nn.Linear(1, 1), second, regression_step).fit(make_loader())

    assert read(second.run_dir) == read(full.run_dir)

    # エポックごとに切り替えて再開した場合は, 前の記録を書き出さない
    switched = make_experiment(
        tmp_path / "switched",
        text.replace("interval: step", "interval: epoch")
        + "trainer:\n  epochs: 3\n"
        f"  resume_from: {first.run_dir / 'last.pt'}\n"
    )
    Trainer(nn.Linear(1, 1), switched, regression_step).fit(make_loader())
    assert not (switched.run_dir / "lr_steps.csv").exists()

    # チェックポイントには数の配列で保存
    state = torch.load(first.run_dir / "last.pt", weights_only=True)
    assert state["lr_steps"]["lr"].dtype == torch.float64
    assert state["lr_steps"]["step"].tolist() == list(
        range(1, len(state["lr_steps"]["step"]) + 1)
    )


def test_override_step(tmp_path: Path) -> None:
    """
    step_fn の代わりに step を上書きでき, エポック数を参照できることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # step を上書きし, 呼ばれた時のエポックとパラメータの更新回数を控える Trainer の作成
    class RegressionTrainer(Trainer):
        def step(self, batch: tuple, phase: str) -> dict:
            """
            平均二乗誤差を求める関数

            Parameters
            ----------
            batch: tuple
                x と y
            phase: str
                train または valid

            Returns
            ----------
            outputs: dict
                平均二乗誤差
            """
            # エポックと更新回数を控えてから誤差の計算
            self.calls.append((phase, self.epoch, self.global_step))
            outputs = regression_step(self.model, batch, phase)

            return outputs

    # 2 エポック学習して呼ばれた順を記録
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 2\n")
    trainer = RegressionTrainer(nn.Linear(1, 1), experiment)
    trainer.calls = []
    trainer.fit(make_loader(n=32, batch_size=16), make_loader(n=16))

    assert trainer.calls == [
        ("train", 1, 0), ("train", 1, 1), ("valid", 1, 2),
        ("train", 2, 2), ("train", 2, 3), ("valid", 2, 4)
    ]


def test_phase_with_masks(tmp_path: Path) -> None:
    """
    1 つのデータを phase に応じたマスクで学習と検証に使い分けられることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # GNN のように学習用と検証用のマスクを持つ 1 つのデータの作成
    x = torch.linspace(-1, 1, 10).unsqueeze(1)
    data = {
        "x": x, "y": 2 * x + 1,
        "train_mask": torch.arange(10) < 7,
        "valid_mask": torch.arange(10) >= 7
    }

    def masked_step(model: nn.Module, batch: dict, phase: str) -> dict:
        """
        phase に応じたマスクで平均二乗誤差を求める関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: dict
            特徴量, 正解, マスク
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            平均二乗誤差と使ったサンプル数
        """
        # phase に応じたマスクで選んだノードだけ誤差の計算
        mask = batch[f"{phase}_mask"]
        pred = model(batch["x"])
        mse = nn.MSELoss()
        outputs = {
            "loss": mse(pred[mask], batch["y"][mask]),
            "n_nodes": float(mask.sum())
        }

        return outputs

    # 同じデータを学習用と検証用の両方に受け渡し
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 1\n")
    history = Trainer(nn.Linear(1, 1), experiment, masked_step).fit(
        [data], [data]
    )

    assert history.get("train/n_nodes") == [7.0]
    assert history.get("valid/n_nodes") == [3.0]


def test_on_epoch_end(tmp_path: Path) -> None:
    """
    エポック全体から求めた指標を記録できることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # エポック全体の予測を集めて最大誤差を求める Trainer の作成
    class MaxErrorTrainer(Trainer):
        def on_epoch_start(self, phase: str) -> None:
            """
            誤差を集める入れ物を用意する関数

            Parameters
            ----------
            phase: str
                train または valid

            Returns
            ----------
            None
            """
            # 入れ物を空に初期化
            self.errors = []

        def step(self, batch: tuple, phase: str) -> dict:
            """
            平均二乗誤差を求めて誤差を集める関数

            Parameters
            ----------
            batch: tuple
                x と y
            phase: str
                train または valid

            Returns
            ----------
            outputs: dict
                平均二乗誤差
            """
            # 予測して誤差の収集
            x, y = batch
            pred = self.model(x)
            self.errors.append((pred - y).abs().detach())
            mse = nn.MSELoss()
            outputs = {"loss": mse(pred, y)}

            return outputs

        def on_epoch_end(self, phase: str) -> dict:
            """
            集めた誤差から最大誤差を求める関数

            Parameters
            ----------
            phase: str
                train または valid

            Returns
            ----------
            metrics: dict
                最大誤差
            """
            # 全バッチの誤差をつなげて最大値を取得
            metrics = {"max_error": torch.cat(self.errors).max()}

            return metrics

    # 学習用と検証用のデータで 1 エポック学習
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 1\n")
    history = MaxErrorTrainer(nn.Linear(1, 1), experiment).fit(
        make_loader(), make_loader()
    )

    assert history.get("valid/max_error")[0] > 0
    assert "train/max_error" in history.keys()


def test_custom_scheduler(tmp_path: Path) -> None:
    """
    自分で作った Optimizer と Scheduler を渡せることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # step ごとに学習率を変える OneCycleLR の作成
    experiment = make_experiment(
        tmp_path, "scheduler:\n  interval: step\ntrainer:\n  epochs: 2\n"
    )
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.1, total_steps=8
    )
    trainer = Trainer(
        model, experiment, regression_step, optimizer, scheduler
    )
    trainer.fit(make_loader(n=64, batch_size=16))

    assert trainer.scheduler is scheduler
    assert trainer.global_step == 8

    # 別の Optimizer で作った Scheduler はエラー
    other = torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.1)
    with pytest.raises(ValueError, match="optimizer"):
        Trainer(
            nn.Linear(1, 1), experiment, regression_step,
            scheduler=torch.optim.lr_scheduler.StepLR(other, step_size=1)
        )

    # 間隔の typo は出力先を作る前の start_experiment でエラー
    with pytest.raises(ValueError, match="interval"):
        make_experiment(
            tmp_path / "typo", "scheduler:\n  interval: Epoch\n"
        )
    assert not (tmp_path / "typo" / "outputs").exists()

    # 監視する指標と向きが違う ReduceLROnPlateau はエラー
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    plateau = make_experiment(tmp_path / "plateau", "trainer:\n  mode: max\n")
    with pytest.raises(ValueError, match="mode"):
        Trainer(
            model, plateau, regression_step, optimizer,
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        )


@pytest.mark.parametrize(
    "outputs, message",
    [
        ({"loss": 1.0}, "torch.Tensor"),
        ({"mae": torch.tensor(1.0)}, "loss"),
        ({"loss": torch.tensor(1.0), "loss/kl": 1.0}, "/"),
        ({"loss": torch.tensor(float("nan"))}, "nan"),
        ({"loss": torch.tensor(1.0), "pred": torch.ones(3)}, "1 つの値")
    ]
)
def test_invalid_outputs(tmp_path: Path, outputs: dict, message: str) -> None:
    """
    step の戻り値が正しくない場合に分かりやすいエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    outputs: dict
        正しくない step の戻り値
    message: str
        エラーメッセージに含まれるべき文字列

    Returns
    ----------
    None
    """
    # 正しくない戻り値を返す step で学習
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 1\n")
    trainer = Trainer(nn.Linear(1, 1), experiment, lambda m, b, p: outputs)

    with pytest.raises(ValueError, match=message):
        trainer.fit(make_loader())


def test_errors(tmp_path: Path) -> None:
    """
    使い方が正しくない場合にエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # step_fn も上書きも無い場合
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 1\n")
    with pytest.raises(NotImplementedError):
        Trainer(nn.Linear(1, 1), experiment).fit(make_loader())

    # 存在しない指標を監視する場合
    experiment = make_experiment(
        tmp_path, "trainer:\n  epochs: 1\n  monitor: valid/acc\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    with pytest.raises(ValueError, match="valid/acc"):
        trainer.fit(make_loader())
    assert trainer.history.records == []

    # loader が空の場合
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 1\n")
    with pytest.raises(ValueError, match="空"):
        Trainer(nn.Linear(1, 1), experiment, regression_step).fit([])


def test_one_cycle_steps_from_loader(tmp_path: Path) -> None:
    """
    OneCycleLR の総ステップ数を, 学習用データのバッチ数から補うことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    base = "trainer:\n  epochs: 3\nscheduler:\n  name: OneCycleLR\n"

    # step ごとなら, エポック数とバッチ数 (64 / 16 = 4) を補い全ステップを使用
    experiment = make_experiment(
        tmp_path / "step",
        base + "  interval: step\n  params:\n    max_lr: 0.1\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    assert trainer.scheduler is None
    trainer.fit(make_loader(n=64, batch_size=16))
    assert trainer.scheduler.total_steps == 12
    saved = OmegaConf.load(experiment.run_dir / "config.yaml")
    assert saved.scheduler.params.steps_per_epoch == 4
    assert saved.scheduler.params.epochs == 3
    log = (experiment.run_dir / "train.log").read_text(encoding="utf-8")
    assert "steps_per_epoch=4" in log
    with open(experiment.run_dir / "lr_steps.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 12

    # 書いたエポック数は残し, エポックごとなら 1 エポックに 1 回
    experiment = make_experiment(
        tmp_path / "epoch",
        base + "  interval: epoch\n  params:\n    max_lr: 0.1\n"
        "    epochs: 4\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader(n=64, batch_size=16))
    assert trainer.scheduler.total_steps == 4

    # len() を持たない loader では, 書くように通知
    experiment = make_experiment(
        tmp_path / "iterable",
        base + "  interval: step\n  params:\n    max_lr: 0.1\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    batches = list(make_loader(n=64, batch_size=16))
    with pytest.raises(ValueError, match="steps_per_epoch を補えません"):
        trainer.fit(batch for batch in batches)


def test_one_cycle_steps_resume(tmp_path: Path) -> None:
    """
    補った OneCycleLR でも, 再開して続けて学習した場合と同じ学習率に
    なることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    text = (
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
        "scheduler:\n  name: OneCycleLR\n  interval: step\n"
        "  params:\n    max_lr: 0.1\n    epochs: 4\n"
    )
    full = make_experiment(tmp_path / "full", text + "trainer:\n  epochs: 4\n")
    full_history = Trainer(nn.Linear(1, 1), full, regression_step).fit(
        make_loader()
    )
    first = make_experiment(
        tmp_path / "first", text + "trainer:\n  epochs: 2\n"
    )
    Trainer(nn.Linear(1, 1), first, regression_step).fit(make_loader())
    second = make_experiment(
        tmp_path / "second",
        text + "trainer:\n  epochs: 4\n"
        f"  resume_from: {first.run_dir / 'last.pt'}\n"
    )
    resumed = Trainer(nn.Linear(1, 1), second, regression_step).fit(
        make_loader()
    )
    assert resumed.get("lr") == pytest.approx(full_history.get("lr"))

    # 補う設定でエポック数を増やして再開すると, 学習を始める前にエラー
    auto = (
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
        "scheduler:\n  name: OneCycleLR\n  interval: step\n"
        "  params:\n    max_lr: 0.1\n"
    )
    short = make_experiment(
        tmp_path / "short", auto + "trainer:\n  epochs: 2\n"
    )
    Trainer(nn.Linear(1, 1), short, regression_step).fit(make_loader())
    longer = make_experiment(
        tmp_path / "longer",
        auto + "trainer:\n  epochs: 4\n"
        f"  resume_from: {short.run_dir / 'last.pt'}\n"
    )
    with pytest.raises(ValueError, match="変えられません"):
        Trainer(nn.Linear(1, 1), longer, regression_step).fit(make_loader())
    assert not (longer.run_dir / "last.pt").exists()

    # 書いた総ステップ数が予定より少ない場合も, 学習を始める前にエラー.
    # 多い場合は警告
    for total, error in ((2, True), (100, False)):
        explicit = make_experiment(
            tmp_path / f"explicit_{total}",
            auto + f"    total_steps: {total}\ntrainer:\n  epochs: 2\n"
        )
        trainer = Trainer(nn.Linear(1, 1), explicit, regression_step)
        if error:
            with pytest.raises(ValueError, match="予定の総ステップ数"):
                trainer.fit(make_loader())
        else:
            with pytest.warns(UserWarning, match="下がりきりません"):
                trainer.fit(make_loader())

    # 空の loader は分かりやすいエラー
    empty = make_experiment(
        tmp_path / "empty", auto + "trainer:\n  epochs: 2\n"
    )
    with pytest.raises(ValueError, match="train の loader が空です"):
        Trainer(nn.Linear(1, 1), empty, regression_step).fit([])


def moving_step(losses: list) -> Callable[[nn.Module, tuple, str], dict]:
    """
    決めておいた損失の値を返しつつ, 重みは実際の誤差で動く step を作る関数

    Parameters
    ----------
    losses: list
        順に返す損失の値

    Returns
    ----------
    step: Callable[[nn.Module, tuple, str], dict]
        値は決めておいたもので, 勾配は二乗誤差のもの
    """
    values = iter(losses)

    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        決めておいた値の損失を, 二乗誤差の勾配が流れる形で返す関数

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
            決めておいた値の損失
        """
        mse = ((model(batch[0]) - batch[1]) ** 2).mean()
        outputs = {"loss": mse + (next(values) - mse.detach())}

        return outputs

    return step


def test_light_checkpoints(tmp_path: Path) -> None:
    """
    best.pt を重みだけにする設定と, last.pt にベストモデルの重みを入れない設定を
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 損失が 2 エポック目で最小になり, その後は悪化する学習 (重みは毎回動く)
    losses = [3.0, 1.0, 2.0, 4.0]

    def run(name: str, text: str, epochs: int, resume: str = "") -> Trainer:
        """
        決まった損失の推移で学習する関数

        Parameters
        ----------
        name: str
            出力先の名前
        text: str
            trainer に加える YAML の内容 (2 文字の字下げ)
        epochs: int
            エポック数
        resume: str = ""
            再開元のチェックポイントのパス

        Returns
        ----------
        trainer: Trainer
            学習を終えた Trainer
        """
        resume_text = f"  resume_from: {resume}\n" if resume else ""
        experiment = make_experiment(
            tmp_path / name,
            "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
            f"trainer:\n  epochs: {epochs}\n" + resume_text + text
        )
        trainer = Trainer(nn.Linear(1, 1), experiment, moving_step(losses))
        trainer.fit(make_loader(n=8, batch_size=8))

        return trainer

    def load(path: Path) -> dict:
        """
        チェックポイントを cpu に読み込む関数

        Parameters
        ----------
        path: Path
            チェックポイントのパス

        Returns
        ----------
        state: dict
            チェックポイントの内容
        """
        state = torch.load(path, map_location="cpu", weights_only=True)

        return state

    def same(left: dict, right: dict) -> bool:
        """
        2 つの重みが同じかを判定する関数

        Parameters
        ----------
        left: dict
            重み
        right: dict
            重み

        Returns
        ----------
        equal: bool
            全てのテンソルが同じか
        """
        equal = all(torch.equal(left[key], right[key]) for key in right)

        return equal

    # 既定でも best.pt はベストモデルの重みを 2 つ持たず, そこから再開できる
    plain = run("plain", "", 3)
    best = load(plain.experiment.run_dir / "best.pt")
    last = load(plain.experiment.run_dir / "last.pt")
    assert best["epoch"] == 2
    assert best["checkpoint"]["best_model"] is None
    assert last["checkpoint"]["best_model"] is not None
    assert not same(best["model"], last["model"])
    from_best = run(
        "from_best", "", 2, str(plain.experiment.run_dir / "best.pt")
    )
    assert same(from_best.checkpoint.best_model_state, best["model"])

    # best.pt を重みだけに, last.pt からベストモデルの重みを外す
    light = (
        "  checkpoint:\n    best_weights_only: true\n"
        "    last_with_best: false\n"
    )
    first = run("first", light, 3)
    best = load(first.experiment.run_dir / "best.pt")
    assert set(best) == {
        "weights_only", "epoch", "monitor", "best_value", "model"
    }
    assert load(first.experiment.run_dir / "last.pt")["checkpoint"][
        "best_model"
    ] is None

    # 再開すると, ベストモデルの重みを再開元の best.pt から戻し,
    # 再開時の last.pt も設定どおりにベストモデルの重みを外す
    resumed = run(
        "resumed", light, 4, str(first.experiment.run_dir / "last.pt")
    )
    assert same(resumed.checkpoint.best_model_state, best["model"])
    final = load(resumed.experiment.run_dir / "final_model.pt")
    assert same(final, best["model"])

    # best.pt が無いと警告し, 最後の重みが残ることを run.json にも記録
    (first.experiment.run_dir / "best.pt").rename(
        first.experiment.run_dir / "best_moved.pt"
    )
    with pytest.warns(UserWarning, match="best.pt がありません"):
        missing = run(
            "missing", light, 4, str(first.experiment.run_dir / "last.pt")
        )
    status = json.loads(
        (missing.experiment.run_dir / "run.json").read_text(encoding="utf-8")
    )
    assert status["final_weights"] == "last"

    # 重みだけの best.pt と final_model.pt からは, 出力先を作る前にエラー
    for name in ("best_moved.pt", "final_model.pt"):
        with pytest.raises(ValueError, match="last.pt を指定"):
            run(name, light, 4, str(first.experiment.run_dir / name))
        assert not (tmp_path / name / "outputs").exists()


def test_stop_within_epoch(tmp_path: Path) -> None:
    """
    STOP に now と書くとエポックの途中で止め, 最後に終えたエポックの状態に
    戻すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    text = (
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
        "scheduler:\n  name: StepLR\n  interval: step\n"
        "  params:\n    step_size: 2\n"
        "trainer:\n  epochs: 5\n  restore_best: false\n"
    )
    experiment = make_experiment(tmp_path / "stop", text)
    calls = {"train": 0}

    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        2 エポック目の 2 バッチ目で, エポックの途中で止める要求を出す関数

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
        if phase == "train":
            calls["train"] += 1
            if calls["train"] == 6:
                (experiment.run_dir / "STOP").write_text("now\n")
        outputs = regression_step(model, batch, phase)

        return outputs

    # 1 エポック 4 バッチで, 毎バッチ要求を確かめる
    trainer = Trainer(nn.Linear(1, 1), experiment, step)
    trainer.run_status.progress_interval = 0.0
    history = trainer.fit(make_loader(n=64, batch_size=16))

    # 2 エポック目は捨て, 1 エポック目の記録と重みが残る
    assert history.get("epoch") == [1]
    run_dir = experiment.run_dir
    status = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert status["status"] == "stopped"
    assert status["completed_epochs"] == 1
    last = torch.load(
        run_dir / "last.pt", map_location="cpu", weights_only=True
    )
    assert last["epoch"] == 1
    final = torch.load(
        run_dir / "final_model.pt", map_location="cpu", weights_only=True
    )
    assert all(torch.equal(final[key], last["model"][key]) for key in final)
    with open(run_dir / "lr_steps.csv", encoding="utf-8") as f:
        assert [row["epoch"] for row in csv.DictReader(f)] == ["1"] * 4
    log = (run_dir / "train.log").read_text(encoding="utf-8")
    assert "エポック 2 の途中で学習を止めました" in log

    # 止めた last.pt から再開すると, 2 エポック目から学習
    resumed = make_experiment(
        tmp_path / "resumed",
        text + f"  resume_from: {run_dir / 'last.pt'}\n"
    )
    history = Trainer(nn.Linear(1, 1), resumed, regression_step).fit(
        make_loader(n=64, batch_size=16)
    )
    assert history.get("epoch") == [1, 2, 3, 4, 5]


def test_stop_file_without_now(tmp_path: Path) -> None:
    """
    STOP の中身が now でなければ, エポックの途中では止めないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 3\n")
    calls = {"train": 0}

    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        1 エポック目の途中で, 中身の無い STOP を置く関数

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
        if phase == "train":
            calls["train"] += 1
            if calls["train"] == 2:
                (experiment.run_dir / "STOP").touch()
        outputs = regression_step(model, batch, phase)

        return outputs

    trainer = Trainer(nn.Linear(1, 1), experiment, step)
    trainer.run_status.progress_interval = 0.0
    history = trainer.fit(make_loader(n=64, batch_size=16))

    # これまで通り, 1 エポック目を終えてから止める
    assert history.get("epoch") == [1]
    assert calls["train"] == 4


def stop_at(
    experiment: Experiment, phase: str, call: int, text: str = "now\n"
) -> Callable[[nn.Module, tuple, str], dict]:
    """
    指定した呼び出しでエポックの途中で止める要求を出す step を作る関数

    Parameters
    ----------
    experiment: Experiment
        STOP を置く実験
    phase: str
        数えるフェーズ (train か valid)
    call: int
        STOP を置く呼び出しの回数
    text: str = "now\n"
        STOP の中身

    Returns
    ----------
    step: Callable[[nn.Module, tuple, str], dict]
        平均二乗誤差を返す step
    """
    calls = {"count": 0}

    def step(model: nn.Module, batch: tuple, phase_now: str) -> dict:
        """
        決めた呼び出しで STOP を置いて平均二乗誤差を返す関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: tuple
            x と y
        phase_now: str
            train または valid

        Returns
        ----------
        outputs: dict
            平均二乗誤差
        """
        if phase_now == phase:
            calls["count"] += 1
            if calls["count"] == call:
                (experiment.run_dir / "STOP").write_text(
                    text, encoding="utf-8"
                )
        outputs = regression_step(model, batch, phase_now)

        return outputs

    return step


def test_stop_within_epoch_cases(tmp_path: Path) -> None:
    """
    1 エポック目, 検証中, last.pt の無いエポックで止めた場合を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    sgd = "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"

    def run(
        name: str, text: str, phase: str, call: int, stop: str = "now\n"
    ) -> tuple[Trainer, dict, dict]:
        """
        途中で止める学習を行う関数

        Parameters
        ----------
        name: str
            出力先の名前
        text: str
            default に加える YAML の内容
        phase: str
            STOP を置くフェーズ
        call: int
            STOP を置く呼び出しの回数
        stop: str = "now\n"
            STOP の中身

        Returns
        ----------
        trainer: Trainer
            学習を終えた Trainer
        start: dict
            学習を始めた時点の重み
        status: dict
            run.json の内容
        """
        experiment = make_experiment(tmp_path / name, sgd + text)
        model = nn.Linear(1, 1)
        start = {
            key: value.clone() for key, value in model.state_dict().items()
        }
        trainer = Trainer(
            model, experiment, stop_at(experiment, phase, call, stop)
        )
        trainer.run_status.progress_interval = 0.0
        loader = make_loader(n=64, batch_size=16)
        trainer.fit(loader, loader)
        status = json.loads(
            (experiment.run_dir / "run.json").read_text(encoding="utf-8")
        )

        return trainer, start, status

    def load(trainer: Trainer) -> dict:
        """
        保存した final_model.pt を読み込む関数

        Parameters
        ----------
        trainer: Trainer
            学習を終えた Trainer

        Returns
        ----------
        weights: dict
            final_model.pt の重み
        """
        weights = torch.load(
            trainer.experiment.run_dir / "final_model.pt",
            map_location="cpu", weights_only=True
        )

        return weights

    # 1 エポック目で止めると, 既定の設定でも始めた時点の重みに戻り,
    # 記録は見出しだけ (BOM 付きの now も効く)
    trainer, start, status = run(
        "first", "trainer:\n  epochs: 3\n", "train", 2, "\ufeffnow\n"
    )
    assert all(torch.equal(load(trainer)[key], start[key]) for key in start)
    assert status["best_epoch"] is None
    assert status["final_weights"] == "last"
    history = (trainer.experiment.run_dir / "history.csv").read_text()
    assert history.strip() == "epoch"

    # 検証中に止めても, そのエポックは記録に加えずベストモデルを残す
    trainer, _, status = run(
        "valid", "trainer:\n  epochs: 3\n", "valid", 6
    )
    assert trainer.history.get("epoch") == [1]
    assert status["final_weights"] == "best"
    assert status["status"] == "stopped"

    # 前のエポックの last.pt が無いと, 途中までの重みが残ることを記録
    with pytest.warns(UserWarning, match="途中までの重み"):
        trainer, _, status = run(
            "partial",
            "trainer:\n  epochs: 5\n  restore_best: false\n"
            "  checkpoint:\n    every_n_epochs: 3\n",
            "train", 10
        )
    assert trainer.history.get("epoch") == [1, 2]
    assert status["final_weights"] == "partial"

    # 検証の最後のバッチで気付いた場合は, エポックを終えてから止める
    trainer, _, status = run("last", "trainer:\n  epochs: 3\n", "valid", 4)
    assert trainer.history.get("epoch") == [1]
    assert (trainer.experiment.run_dir / "last.pt").exists()


def test_warn_without_best_weights(tmp_path: Path) -> None:
    """
    last.pt にも best.pt にもベストモデルの重みを残さない設定で, 出力先を作る前に
    警告が出ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    with pytest.warns(UserWarning, match="ベストモデルの重みに戻せません"):
        make_experiment(
            tmp_path,
            "trainer:\n  checkpoint:\n    last_with_best: false\n"
            "    save_best: false\n"
        )


def test_step_extra(tmp_path: Path) -> None:
    """
    step の extra で返した値を, 平均せずにエポックの最後の値で記録することを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    class ScheduleTrainer(Trainer):
        def step(self, batch: tuple, phase: str) -> dict:
            """
            損失と, エポックで決まる重みを extra で返す関数

            Parameters
            ----------
            batch: tuple
                x と y
            phase: str
                train または valid

            Returns
            ----------
            outputs: dict
                損失と, 記録だけする重み
            """
            outputs = regression_step(self.model, batch, phase)
            outputs["extra"] = {"weight": 0.1 * self.epoch}

            return outputs

    experiment = make_experiment(tmp_path / "ok", "trainer:\n  epochs: 3\n")
    trainer = ScheduleTrainer(nn.Linear(1, 1), experiment)
    history = trainer.fit(make_loader(), make_loader())

    # train/ と valid/ が付かず, 平均の誤差も無い
    assert history.get("weight") == [0.1, 0.2, 0.30000000000000004]
    assert "train/weight" not in history.keys()
    assert "valid/weight" not in history.keys()

    # 名前と値の形の誤りはエラー
    for name, extra in (
        ("slash", {"a/b": 1.0}), ("reserved", {"lr": 1.0}),
        ("shape", {"w": torch.ones(2)}), ("type", [1.0]),
        ("comma", {"a, b": 1.0}), ("empty", {"": 1.0}), ("key", {1: 1.0}),
        ("string", {"w": "abc"}), ("none", {"w": None}),
        ("metric", {"loss": 1.0})
    ):
        class BadTrainer(Trainer):
            def step(self, batch: tuple, phase: str) -> dict:
                """
                誤った extra を返す関数

                Parameters
                ----------
                batch: tuple
                    x と y
                phase: str
                    train または valid

                Returns
                ----------
                outputs: dict
                    損失と誤った extra
                """
                outputs = regression_step(self.model, batch, phase)
                outputs["extra"] = extra

                return outputs

        bad = make_experiment(tmp_path / name, "trainer:\n  epochs: 1\n")
        with pytest.raises(ValueError, match="extra"):
            BadTrainer(nn.Linear(1, 1), bad).fit(make_loader())


def test_warn_unknown_batch_size(tmp_path: Path) -> None:
    """
    バッチのサンプル数を推定できない場合に, 1 回だけ警告することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    def step(model: nn.Module, batch: dict, phase: str) -> dict:
        """
        テンソルを含まないバッチから損失を求める関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: dict
            数だけを持つバッチ
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            損失
        """
        x = torch.tensor([[float(batch["x"])]])
        outputs = {"loss": (model(x) ** 2).mean()}

        return outputs

    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 2\n")
    batches = [{"x": 1}, {"x": 2}]
    with pytest.warns(UserWarning, match="サンプル数を推定できない") as record:
        Trainer(nn.Linear(1, 1), experiment, step).fit(batches)
    assert sum("サンプル数" in str(item.message) for item in record) == 1
    log = (experiment.run_dir / "train.log").read_text(encoding="utf-8")
    assert log.count("サンプル数を推定できない") == 1


def test_select_best_false(tmp_path: Path) -> None:
    """
    ベストモデルを選ばない設定で, best.pt を作らず最後の重みを残すことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 損失が 2 エポック目で最小になり, その後は悪化する学習 (重みは毎回動く)
    experiment = make_experiment(
        tmp_path / "ok",
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
        "trainer:\n  epochs: 4\n  select_best: false\n"
    )
    trainer = Trainer(
        nn.Linear(1, 1), experiment, moving_step([3.0, 1.0, 2.0, 4.0])
    )
    trainer.fit(make_loader(n=8, batch_size=8))

    run_dir = experiment.run_dir
    assert not (run_dir / "best.pt").exists()
    final = torch.load(
        run_dir / "final_model.pt", map_location="cpu", weights_only=True
    )
    last = torch.load(
        run_dir / "last.pt", map_location="cpu", weights_only=True
    )
    assert all(torch.equal(final[key], last["model"][key]) for key in final)
    status = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert status["final_weights"] == "last"
    assert status["best_epoch"] is None
    log = (run_dir / "train.log").read_text(encoding="utf-8")
    assert "select_best: false" in log

    # Early Stopping とは一緒に使えない
    with pytest.raises(ValueError, match="select_best"):
        make_experiment(
            tmp_path / "early",
            "trainer:\n  select_best: false\n  early_stopping:\n"
            "    use: true\n"
        )


def test_select_best_resume(tmp_path: Path) -> None:
    """
    select_best を変えて再開した場合と, エポックの途中で止めた場合を
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    sgd = "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"

    # ベストモデルを選ぶ学習 (2 エポック目がベスト) を 3 エポック
    first = make_experiment(
        tmp_path / "first", sgd + "trainer:\n  epochs: 3\n"
    )
    Trainer(
        nn.Linear(1, 1), first, moving_step([3.0, 1.0, 2.0])
    ).fit(make_loader(n=8, batch_size=8))
    assert (first.run_dir / "best.pt").exists()

    # 選ばない設定で再開し, 5 エポック目の途中で止める
    second = make_experiment(
        tmp_path / "second",
        sgd + "trainer:\n  epochs: 6\n  select_best: false\n"
        f"  resume_from: {first.run_dir / 'last.pt'}\n"
    )
    calls = {"count": 0}

    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        再開した 2 エポック目 (エポック 5) の学習で, 途中で止める要求を出す関数

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
        if phase == "train":
            calls["count"] += 1
            if calls["count"] == 6:
                (second.run_dir / "STOP").write_text("now\n")
        outputs = regression_step(model, batch, phase)

        return outputs

    with pytest.warns(UserWarning, match="select_best: false"):
        trainer = Trainer(nn.Linear(1, 1), second, step)
        trainer.run_status.progress_interval = 0.0
        trainer.fit(make_loader(n=8, batch_size=2))

    # best.pt を写さず, 止めたエポックの前の last.pt の重みに戻る
    run_dir = second.run_dir
    assert not (run_dir / "best.pt").exists()
    status = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert status["best_epoch"] is None
    assert status["select_best"] is False
    assert status["final_weights"] == "last"
    last = torch.load(
        run_dir / "last.pt", map_location="cpu", weights_only=True
    )
    assert last["epoch"] == 4
    final = torch.load(
        run_dir / "final_model.pt", map_location="cpu", weights_only=True
    )
    assert all(torch.equal(final[key], last["model"][key]) for key in final)

    # 選ばない設定から選ぶ設定で再開すると警告
    third = make_experiment(
        tmp_path / "third",
        sgd + f"trainer:\n  epochs: 5\n  resume_from: {run_dir / 'last.pt'}\n"
    )
    with pytest.warns(UserWarning, match="再開した後のエポックから"):
        Trainer(nn.Linear(1, 1), third, regression_step).fit(
            make_loader(n=8, batch_size=8)
        )

    # 同じ値の場合に比べる指標とは一緒に使えない
    with pytest.raises(ValueError, match="monitor_tiebreak"):
        make_experiment(
            tmp_path / "tiebreak",
            "trainer:\n  select_best: false\n"
            "  monitor_tiebreak: valid/loss\n"
        )


class EMATrainer(Trainer):
    def __init__(self, model: nn.Module, experiment: Experiment):
        # 親クラスのコンストラクタを呼び出す
        super(EMATrainer, self).__init__(model, experiment, None)

        # 重みの指数移動平均を持つモデルと, 確認用の記録を保存
        self.ema = optim.swa_utils.AveragedModel(
            self.model,
            multi_avg_fn=optim.swa_utils.get_ema_multi_avg_fn(0.5)
        )
        self.valid_models = []
        self.ema_states = {}

    def step(self, batch: tuple, phase: str) -> dict:
        """
        検証に渡されたモデルを記録して平均二乗誤差を求める関数

        Parameters
        ----------
        batch: tuple
            x と y
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            平均二乗誤差と平均絶対誤差
        """
        # 検証では評価に使うモデルで予測し, 使ったモデルを記録
        model = self.model if phase == "train" else self.eval_model()
        if phase == "valid":
            self.valid_models.append(model)
        outputs = regression_step(model, batch, phase)

        return outputs

    def update_parameters(self, loss: torch.Tensor) -> None:
        """
        パラメータを更新した後に EMA の重みも更新する関数

        Parameters
        ----------
        loss: torch.Tensor
            1 バッチ分の損失

        Returns
        ----------
        None
        """
        # 元の手順で更新してから EMA を更新
        super(EMATrainer, self).update_parameters(loss)
        self.ema.update_parameters(self.model)

    def on_epoch_end(self, phase: str) -> dict:
        """
        検証の後の EMA の重みを記録する関数

        Parameters
        ----------
        phase: str
            train または valid

        Returns
        ----------
        metrics: dict
            空の辞書
        """
        # 検証の後の EMA の重みを複製して記録
        if phase == "valid":
            self.ema_states[self.epoch] = {
                key: value.detach().clone()
                for key, value in self.ema.module.state_dict().items()
            }
        metrics = {}

        return metrics

    def extra_state_dict(self) -> dict:
        """
        チェックポイントに EMA の状態を加える関数

        Parameters
        ----------
        None

        Returns
        ----------
        state: dict
            EMA の重みと更新回数
        """
        # EMA の状態を受け渡し
        state = {"ema": self.ema.state_dict()}

        return state

    def load_extra_state_dict(self, state: dict) -> None:
        """
        チェックポイントから EMA の状態を戻す関数

        Parameters
        ----------
        state: dict
            extra_state_dict で加えた状態

        Returns
        ----------
        None
        """
        # EMA の状態の復元
        self.ema.load_state_dict(state["ema"])

    def eval_model(self) -> nn.Module:
        """
        EMA の重みを持つモデルを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        model: nn.Module
            EMA の重みを持つモデル
        """
        # EMA の中のモデルを使用
        model = self.ema.module

        return model


class WrongTrainer(Trainer):
    def __init__(self, model: nn.Module, experiment: Experiment, out: int):
        # 親クラスのコンストラクタを呼び出す
        super(WrongTrainer, self).__init__(model, experiment, regression_step)

        # 評価に使う, 重みの名前か形が違うモデルを保存
        self.other = nn.Linear(1, out, bias=out == 1)

    def eval_model(self) -> nn.Module:
        """
        重みの名前か形が違うモデルを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        model: nn.Module
            self.model と重みの違うモデル
        """
        # 用意したモデルを使用
        model = self.other

        return model


def same_weights(left: dict, right: dict) -> bool:
    """
    2 つの重みが全て等しいかを確かめる関数

    Parameters
    ----------
    left: dict
        比べる重み
    right: dict
        比べる重み

    Returns
    ----------
    same: bool
        名前と値が全て等しいかどうか
    """
    # 名前と値を全て比較
    same = left.keys() == right.keys() and all(
        torch.equal(left[key], right[key]) for key in left
    )

    return same


def test_eval_model(tmp_path: Path) -> None:
    """
    eval_model を上書きすると, 検証, ベストモデル, best.pt, final_model.pt
    にそのモデルが使われることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # ベストモデルに戻す場合は, ベストのエポックの EMA の重みが残る
    text = "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
    experiment = make_experiment(
        tmp_path / "best", text + "trainer:\n  epochs: 3\n"
    )
    trainer = EMATrainer(nn.Linear(1, 1), experiment)
    trainer.fit(make_loader(), make_loader())
    assert all(model is trainer.ema.module for model in trainer.valid_models)
    assert trainer.ema.module.training is False
    expected = trainer.ema_states[trainer.checkpoint.best_epoch]
    run_dir = experiment.run_dir
    final = torch.load(
        run_dir / "final_model.pt", map_location="cpu", weights_only=True
    )
    assert same_weights(final, expected)
    log = (run_dir / "train.log").read_text(encoding="utf-8")
    assert "評価に使うモデル" in log

    # best.pt は学習中の重みと, EMA のベストモデルの重みの両方を持つ
    best = load_checkpoint(str(run_dir / "best.pt"), torch.device("cpu"))
    assert same_weights(best["checkpoint"]["best_model"], expected)
    assert not same_weights(best["model"], expected)

    # 重みだけの best.pt は EMA のベストモデルの重み
    experiment = make_experiment(
        tmp_path / "weights",
        text + "trainer:\n  epochs: 3\n  checkpoint:\n"
        "    best_weights_only: true\n"
    )
    trainer = EMATrainer(nn.Linear(1, 1), experiment)
    trainer.fit(make_loader(), make_loader())
    expected = trainer.ema_states[trainer.checkpoint.best_epoch]
    best = torch.load(
        experiment.run_dir / "best.pt", map_location="cpu", weights_only=True
    )
    assert same_weights(best["model"], expected)

    # ベストモデルを選ばない場合は, 最後の EMA の重みが残り self.model に入る
    experiment = make_experiment(
        tmp_path / "last",
        text + "trainer:\n  epochs: 3\n  select_best: false\n"
    )
    trainer = EMATrainer(nn.Linear(1, 1), experiment)
    trainer.fit(make_loader(), make_loader())
    expected = trainer.ema_states[3]
    final = torch.load(
        experiment.run_dir / "final_model.pt", map_location="cpu",
        weights_only=True
    )
    assert same_weights(final, expected)
    assert same_weights(trainer.model.state_dict(), expected)
    last = torch.load(
        experiment.run_dir / "last.pt", map_location="cpu", weights_only=True
    )
    assert not same_weights(last["model"], expected)

    # 重みの名前か形が違うモデルは学習を始める前にエラー
    for out, message in [(2, "名前"), (1, "形")]:
        experiment = make_experiment(tmp_path / f"wrong{out}", text)
        trainer = WrongTrainer(nn.Linear(1, 1), experiment, out)
        if out == 1:
            trainer.other.weight = nn.Parameter(torch.zeros(2, 1))
        with pytest.raises(ValueError, match=message):
            trainer.fit(make_loader())
        assert trainer.history.records == []
        status = json.loads(
            (experiment.run_dir / "run.json").read_text(encoding="utf-8")
        )
        assert status["status"] == "failed"


@pytest.mark.parametrize("weights_only", [False, True])
def test_eval_model_resume(tmp_path: Path, weights_only: bool) -> None:
    """
    eval_model を上書きし, last.pt にベストモデルの重みを入れずに再開しても,
    EMA のベストモデルの重みに戻ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    weights_only: bool
        best.pt を重みだけにするかどうか

    Returns
    ----------
    None
    """
    # 2 エポック学習して, ベストのエポックの EMA の重みを記録
    text = (
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
        "trainer:\n  epochs: 2\n  checkpoint:\n"
        "    last_with_best: false\n"
        f"    best_weights_only: {str(weights_only).lower()}\n"
    )
    experiment = make_experiment(tmp_path / "first", text)
    trainer = EMATrainer(nn.Linear(1, 1), experiment)
    trainer.fit(make_loader(), make_loader())
    expected = trainer.ema_states[trainer.checkpoint.best_epoch]

    # 同じエポック数で再開すると, 学習せずに EMA のベストモデルの重みが残る
    last_path = experiment.run_dir / "last.pt"
    experiment = make_experiment(
        tmp_path / "resume", text + f"  resume_from: {last_path}\n"
    )
    trainer = EMATrainer(nn.Linear(1, 1), experiment)
    trainer.fit(make_loader(), make_loader())
    final = torch.load(
        experiment.run_dir / "final_model.pt", map_location="cpu",
        weights_only=True
    )
    assert same_weights(final, expected)
    status = json.loads(
        (experiment.run_dir / "run.json").read_text(encoding="utf-8")
    )
    assert status["final_weights"] == "best"


def test_eval_model_stop_within_epoch(tmp_path: Path) -> None:
    """
    eval_model が別のモデルの場合に, エポックの途中で止めると途中までの
    重みになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 2 エポック目の学習の途中で止める (ベストモデルを選ばない設定)
    experiment = make_experiment(
        tmp_path,
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
        "trainer:\n  epochs: 3\n  select_best: false\n"
    )
    trainer = EMATrainer(nn.Linear(1, 1), experiment)
    stopper = stop_at(experiment, "train", 6)

    def step(batch: tuple, phase: str) -> dict:
        """
        決めた呼び出しで STOP を置いて平均二乗誤差を返す関数

        Parameters
        ----------
        batch: tuple
            x と y
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            平均二乗誤差
        """
        # 検証では評価に使うモデルで計算
        model = trainer.model if phase == "train" else trainer.eval_model()
        outputs = stopper(model, batch, phase)

        return outputs

    # 中断の要求を毎回確かめるようにして学習
    trainer.step = step
    trainer.run_status.progress_interval = 0.0
    with pytest.warns(UserWarning, match="eval_model"):
        trainer.fit(make_loader(), make_loader())
    status = json.loads(
        (experiment.run_dir / "run.json").read_text(encoding="utf-8")
    )
    assert status["final_weights"] == "partial"
    assert len(trainer.history.records) == 1


def test_register_rng(tmp_path: Path) -> None:
    """
    登録した外部ライブラリの乱数が, シードで固定され, 再開で戻ることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # Python, NumPy, torch の全体の乱数とは別の乱数を持つ外部ライブラリの代わり
    # MT19937 の状態は NumPy の配列を含む
    library = {"rng": np.random.Generator(np.random.MT19937())}

    def seed_fn(seed: int) -> None:
        """
        外部ライブラリの乱数を固定する関数

        Parameters
        ----------
        seed: int
            乱数シードの値

        Returns
        ----------
        None
        """
        # 新しい乱数生成器に置き換えて固定
        library["rng"] = np.random.Generator(np.random.MT19937(seed))

    def get_state() -> dict:
        """
        外部ライブラリの乱数の状態を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        state: dict
            乱数生成器の状態
        """
        # NumPy の配列を含む辞書 (ML-Found が保存できる形に変換)
        state = library["rng"].bit_generator.state

        return state

    def set_state(state: dict) -> None:
        """
        外部ライブラリの乱数の状態を戻す関数

        Parameters
        ----------
        state: dict
            get_state が返した状態

        Returns
        ----------
        None
        """
        # 乱数生成器の状態を上書き
        library["rng"].bit_generator.state = state

    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        外部ライブラリの乱数で揺らした損失を返す関数

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
            平均二乗誤差に外部ライブラリの乱数を足した損失
        """
        # 損失に乱数を足して, 乱数の違いが記録に出るように計算
        outputs = regression_step(model, batch, phase)
        outputs["loss"] = outputs["loss"] + float(library["rng"].random())

        return outputs

    def run(name: str, text: str) -> History:
        """
        外部ライブラリの乱数を登録して学習する関数

        Parameters
        ----------
        name: str
            出力先の名前
        text: str
            default に加える YAML の内容

        Returns
        ----------
        history: History
            エポックごとの記録
        """
        # 登録してから実験を始め, 学習の後に登録の解除
        register_rng("library", seed_fn, get_state, set_state)
        try:
            experiment = make_experiment(tmp_path / name, sgd + text)
            history = Trainer(nn.Linear(1, 1), experiment, step).fit(
                make_loader(), make_loader()
            )
        finally:
            unregister_rng("library")

        return history

    # 同じシードなら同じ結果になり, 再開しても続けて学習した場合と一致
    # 検証の前に固定する設定では, 検証の値は毎エポック同じ
    sgd = (
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
        "trainer:\n  valid_seed: 7\n"
    )
    full = run("full", "  epochs: 4\n")
    assert run("again", "  epochs: 4\n").records == full.records
    run("first", "  epochs: 2\n")
    last_path = next((tmp_path / "first").rglob("last.pt"))
    resumed = run("second", f"  epochs: 4\n  resume_from: {last_path}\n")
    assert resumed.get("train/loss") == pytest.approx(full.get("train/loss"))
    assert resumed.get("valid/loss") == pytest.approx(full.get("valid/loss"))
    assert load_checkpoint(str(last_path), torch.device("cpu"))
    log = next((tmp_path / "full").rglob("train.log")).read_text(
        encoding="utf-8"
    )
    assert "['library']" in log

    # 登録が片方にしか無い場合は警告し, train.log にも記録
    experiment = make_experiment(
        tmp_path / "unknown",
        sgd + f"  epochs: 4\n  resume_from: {last_path}\n"
    )
    with pytest.warns(UserWarning, match="登録されていない"):
        Trainer(nn.Linear(1, 1), experiment, regression_step).fit(
            make_loader(), make_loader()
        )
    log = (experiment.run_dir / "train.log").read_text(encoding="utf-8")
    assert "登録されていない" in log
    plain = make_experiment(tmp_path / "plain", sgd + "  epochs: 1\n")
    Trainer(nn.Linear(1, 1), plain, regression_step).fit(
        make_loader(), make_loader()
    )
    plain_last = plain.run_dir / "last.pt"
    with pytest.warns(UserWarning, match="保存されていない"):
        run("missing", f"  epochs: 2\n  resume_from: {plain_last}\n")

    # start_experiment の後に登録した場合は警告
    experiment = make_experiment(tmp_path / "late", sgd + "  epochs: 1\n")
    register_rng("library", seed_fn, get_state, set_state)
    try:
        with pytest.warns(UserWarning, match="start_experiment の後"):
            Trainer(nn.Linear(1, 1), experiment, step)
    finally:
        unregister_rng("library")

    # 名前と関数の誤りはエラー
    with pytest.raises(ValueError, match="名前"):
        register_rng("", seed_fn, get_state, set_state)
    with pytest.raises(TypeError, match="get_state"):
        register_rng("library", seed_fn, None, set_state)
    with pytest.raises(TypeError, match="保存できない型"):
        register_rng("library", seed_fn, lambda: {"rng": object()}, set_state)
