from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_found.config.loader import load_config
from ml_found.experiment.setup import Experiment, start_experiment
from ml_found.training.callbacks import load_checkpoint
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
    -------
    experiment: Experiment
        cpu で進捗バーを出さない実験
    """
    # cpu で進捗バーを出さない default を用意する
    base = (
        "experiment:\n"
        "  name: test\n"
        "  device: cpu\n"
        "  seed: 0\n"
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    default_path = tmp_path / "default.yaml"
    default_path.write_text(base + text, encoding="utf-8")

    # 実験を開始する
    cfg, diff = load_config(str(default_path))
    cfg.trainer.progress_bar = False
    experiment = start_experiment(cfg, diff, str(tmp_path))

    return experiment


def make_loader(n: int = 64, batch_size: int = 16) -> DataLoader:
    """
    y = 2x + 1 を学習するための DataLoader を作る関数

    Parameters
    ----------
    n: int = 64
        サンプル数
    batch_size: int = 16
        バッチサイズ

    Returns
    -------
    loader: DataLoader
        x と y を返す DataLoader
    """
    # 1 次関数のデータを作る
    x = torch.linspace(-1, 1, n).unsqueeze(1)
    y = 2 * x + 1
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)

    return loader


def regression_step(model: nn.Module, batch: tuple) -> dict:
    """
    平均二乗誤差を求める関数

    Parameters
    ----------
    model: nn.Module
        学習するモデル
    batch: tuple
        x と y

    Returns
    -------
    outputs: dict
        平均二乗誤差と平均絶対誤差
    """
    # 予測して誤差を求める
    x, y = batch
    pred = model(x)
    outputs = {
        "loss": nn.functional.mse_loss(pred, y),
        "mae": (pred - y).abs().mean()
    }

    return outputs


def test_fit_minibatch(tmp_path: Path):
    """
    ミニバッチ学習で損失が下がり, 結果が保存されることを確かめる関数
    """
    # 線形回帰を学習する
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


def test_fit_fullbatch(tmp_path: Path):
    """
    全データを 1 度に流すフルバッチ学習ができることを確かめる関数
    """
    # データ全体を 1 つの要素として渡す
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 5\n")
    x = torch.randn(32, 1)
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    history = trainer.fit([(x, 2 * x + 1)])

    assert len(history.records) == 5
    assert trainer.get_monitor(has_valid=False) == "train/loss"


def test_final_model_is_best(tmp_path: Path):
    """
    学習後のモデルがベストモデルの重みになっていることを確かめる関数
    """
    # 学習率が大きすぎて途中から悪化する設定で学習する
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


def test_early_stopping(tmp_path: Path):
    """
    改善しなくなると途中で学習が止まることを確かめる関数
    """
    # 学習率 0 で全く改善しない設定にする
    experiment = make_experiment(
        tmp_path,
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.0\n"
        "trainer:\n  epochs: 50\n"
        "  early_stopping:\n    use: true\n    patience: 3\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    history = trainer.fit(make_loader())

    assert len(history.records) == 4


def test_resume(tmp_path: Path):
    """
    途中のチェックポイントから学習を再開できることを確かめる関数
    """
    # 3 エポック学習する
    first = make_experiment(tmp_path / "first", "trainer:\n  epochs: 3\n")
    trainer = Trainer(nn.Linear(1, 1), first, regression_step)
    trainer.fit(make_loader())
    last_path = first.run_dir / "last.pt"

    # 5 エポックまで再開する
    text = f"trainer:\n  epochs: 5\n  resume_from: {last_path}\n"
    second = make_experiment(tmp_path / "second", text)
    trainer = Trainer(nn.Linear(1, 1), second, regression_step)
    history = trainer.fit(make_loader())

    assert history.get("epoch") == [1, 2, 3, 4, 5]
    assert trainer.checkpoint.best_model_state is not None


def test_scheduler_and_clipping(tmp_path: Path):
    """
    Scheduler と勾配クリッピングを使って学習できることを確かめる関数
    """
    # step ごとに学習率を下げ, 勾配を切る設定にする
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


def test_override_training_step(tmp_path: Path):
    """
    step_fn の代わりに training_step を上書きできることを確かめる関数
    """
    # training_step を上書きした Trainer を作る
    class RegressionTrainer(Trainer):
        def training_step(self, batch: tuple) -> dict:
            """
            平均二乗誤差を求める関数

            Parameters
            ----------
            batch: tuple
                x と y

            Returns
            -------
            outputs: dict
                平均二乗誤差
            """
            # 予測して誤差を求める
            outputs = regression_step(self.model, batch)

            return outputs

    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 2\n")
    history = RegressionTrainer(nn.Linear(1, 1), experiment).fit(
        make_loader()
    )

    assert len(history.records) == 2


def test_errors(tmp_path: Path):
    """
    使い方が正しくない場合にエラーになることを確かめる関数
    """
    # step_fn も上書きも無い場合
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 1\n")
    with pytest.raises(NotImplementedError):
        Trainer(nn.Linear(1, 1), experiment).fit(make_loader())

    # 存在しない指標を監視する場合
    experiment = make_experiment(
        tmp_path, "trainer:\n  epochs: 1\n  monitor: valid/acc\n"
    )
    with pytest.raises(ValueError, match="valid/acc"):
        Trainer(nn.Linear(1, 1), experiment, regression_step).fit(
            make_loader()
        )

    # 損失が発散した場合
    def nan_step(model: nn.Module, batch: tuple) -> dict:
        """
        常に nan の損失を返す関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: tuple
            x と y

        Returns
        -------
        outputs: dict
            nan の損失
        """
        # 損失を nan にする
        outputs = {"loss": model(batch[0]).sum() * float("nan")}

        return outputs

    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 1\n")
    with pytest.raises(ValueError, match="nan"):
        Trainer(nn.Linear(1, 1), experiment, nan_step).fit(make_loader())
