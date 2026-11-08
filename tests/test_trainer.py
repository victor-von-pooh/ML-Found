from pathlib import Path
from typing import Callable

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
    # cpu で進捗バーを出さない default と空の experiment を用意する
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

    # 実験を開始する
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
    -------
    loader: DataLoader
        x と y を返す DataLoader
    """
    # 1 次関数のデータを作る
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


def scripted_step(losses: list) -> Callable[[nn.Module, tuple, str], dict]:
    """
    呼ばれるたびに決めておいた損失を返す step を作る関数

    Parameters
    ----------
    losses: list
        順に返す損失の値

    Returns
    -------
    step: Callable[[nn.Module, tuple, str], dict]
        決めておいた損失を返す step
    """
    # 呼ばれた回数を数える
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
        -------
        outputs: dict
            決めておいた損失
        """
        # 勾配が 0 になるようにモデルの出力に 0 を掛けて足す
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
    -------
    None
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


def test_fit_fullbatch(tmp_path: Path) -> None:
    """
    全データを 1 度に流すフルバッチ学習ができることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    -------
    None
    """
    # データ全体を 1 つの要素として渡す
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
    -------
    None
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


def test_early_stopping(tmp_path: Path) -> None:
    """
    改善しなくなると途中で学習が止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    -------
    None
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


def test_resume_matches_uninterrupted(tmp_path: Path) -> None:
    """
    途中で止めて再開しても続けて学習した場合と同じ結果になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    -------
    None
    """
    # データを並べ替えながら 5 エポック続けて学習する
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

    # 3 エポックで止めてから 5 エポックまで再開する
    first = make_experiment(
        tmp_path / "first", text + "trainer:\n  epochs: 3\n"
    )
    Trainer(nn.Linear(1, 1), first, regression_step).fit(
        make_loader(shuffle=True)
    )
    second = make_experiment(
        tmp_path / "second",
        text + f"trainer:\n  epochs: 5\n"
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


def test_resume_keeps_best(tmp_path: Path) -> None:
    """
    再開前のベストモデルより悪いエポックをベストとみなさないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    -------
    None
    """
    # 損失が 3, 2, 1 と下がる 3 エポックを学習する
    first = make_experiment(tmp_path / "first", "trainer:\n  epochs: 3\n")
    Trainer(nn.Linear(1, 1), first, scripted_step([3.0, 2.0, 1.0])).fit(
        [(torch.ones(1, 1),)]
    )

    # 損失が 1.5 に悪化する 4 エポック目を再開して学習する
    text = (
        f"trainer:\n  epochs: 4\n"
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
    -------
    None
    """
    # 損失が変わらず, 2 回待つと 3 エポック目で止まる設定にする
    text = "  early_stopping:\n    use: true\n    patience: 2\n"

    # 2 エポックで止めてから再開する
    first = make_experiment(
        tmp_path / "first", "trainer:\n  epochs: 2\n" + text
    )
    Trainer(nn.Linear(1, 1), first, scripted_step([1.0] * 2)).fit(
        [(torch.ones(1, 1),)]
    )
    second = make_experiment(
        tmp_path / "second",
        f"trainer:\n  epochs: 10\n"
        f"  resume_from: {first.run_dir / 'last.pt'}\n" + text
    )
    history = Trainer(
        nn.Linear(1, 1), second, scripted_step([1.0] * 8)
    ).fit([(torch.ones(1, 1),)])

    assert history.get("epoch") == [1, 2, 3]

    # 止めた学習を再開しても学習しない
    third = make_experiment(
        tmp_path / "third",
        f"trainer:\n  epochs: 10\n"
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
    -------
    None
    """
    # 学習率が大きすぎて途中から悪化する設定を best.pt を保存せずに学習する
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


def test_scheduler_and_clipping(tmp_path: Path) -> None:
    """
    Scheduler と勾配クリッピングを使って学習できることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    -------
    None
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


def test_override_step(tmp_path: Path) -> None:
    """
    step_fn の代わりに step を上書きでき, エポック数を参照できることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    -------
    None
    """
    # step を上書きし, 呼ばれた時のエポックとパラメータの更新回数を控える Trainer を作る
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
            -------
            outputs: dict
                平均二乗誤差
            """
            # エポックと更新回数を控えてから誤差を求める
            self.calls.append((phase, self.epoch, self.global_step))
            outputs = regression_step(self.model, batch, phase)

            return outputs

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
    -------
    None
    """
    # GNN のように学習用と検証用のマスクを持つ 1 つのデータを作る
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
        -------
        outputs: dict
            平均二乗誤差と使ったサンプル数
        """
        # phase に応じたマスクで選んだノードだけ誤差を求める
        mask = batch[f"{phase}_mask"]
        pred = model(batch["x"])
        outputs = {
            "loss": nn.functional.mse_loss(pred[mask], batch["y"][mask]),
            "n_nodes": float(mask.sum())
        }

        return outputs

    # 同じデータを学習用と検証用の両方に渡す
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
    -------
    None
    """
    # エポック全体の予測を集めて最大誤差を求める Trainer を作る
    class MaxErrorTrainer(Trainer):
        def on_epoch_start(self, phase: str) -> None:
            """
            誤差を集める入れ物を用意する関数

            Parameters
            ----------
            phase: str
                train または valid

            Returns
            -------
            None
            """
            # 入れ物を空にする
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
            -------
            outputs: dict
                平均二乗誤差
            """
            # 予測して誤差を集める
            x, y = batch
            pred = self.model(x)
            self.errors.append((pred - y).abs().detach())
            outputs = {"loss": nn.functional.mse_loss(pred, y)}

            return outputs

        def on_epoch_end(self, phase: str) -> dict:
            """
            集めた誤差から最大誤差を求める関数

            Parameters
            ----------
            phase: str
                train または valid

            Returns
            -------
            metrics: dict
                最大誤差
            """
            # 全バッチの誤差をつなげて最大値を取る
            metrics = {"max_error": torch.cat(self.errors).max()}

            return metrics

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
    -------
    None
    """
    # step ごとに学習率を変える OneCycleLR を作る
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

    # 別の Optimizer で作った Scheduler はエラーになる
    other = torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.1)
    with pytest.raises(ValueError, match="optimizer"):
        Trainer(
            nn.Linear(1, 1), experiment, regression_step,
            scheduler=torch.optim.lr_scheduler.StepLR(other, step_size=1)
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
    -------
    None
    """
    # 正しくない戻り値を返す step で学習する
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
    -------
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
