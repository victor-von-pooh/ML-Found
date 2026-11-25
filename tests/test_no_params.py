import csv
import json
from pathlib import Path

import pytest
import torch
from torch import nn, optim

from ml_found.training.trainer import Trainer
from test_trainer import make_experiment, make_loader


class DistanceModel(nn.Module):
    def __init__(self, x: torch.Tensor):
        # 親クラスのコンストラクタを呼び出す
        super(DistanceModel, self).__init__()

        # 学習用の行の中心と精度行列をバッファに保存
        center = x.mean(dim=0)
        covariance = torch.atleast_2d(torch.cov(x.T)) + 0.1 * torch.eye(
            x.shape[1]
        )
        self.register_buffer("center", center)
        self.register_buffer("precision", torch.linalg.inv(covariance))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        行ごとに中心からのマハラノビス距離の 2 乗を求める関数

        Parameters
        ----------
        x: torch.Tensor
            行の並び

        Returns
        ----------
        scores: torch.Tensor
            行ごとの度合い
        """
        # 中心との差を精度行列で測った距離
        diff = x - self.center
        scores = ((diff @ self.precision) * diff).sum(dim=1)

        return scores


class DistanceTrainer(Trainer):
    def on_epoch_start(self, phase: str) -> None:
        """
        度合いを集める入れ物を用意する関数

        Parameters
        ----------
        phase: str
            valid または train_eval

        Returns
        ----------
        None
        """
        # 呼ばれたフェーズの記録と入れ物の初期化
        self.phases = getattr(self, "phases", []) + [phase]
        self.scores = []

    def step(self, batch: tuple, phase: str) -> dict:
        """
        度合いの平均を損失として返す関数

        Parameters
        ----------
        batch: tuple
            x と y
        phase: str
            valid または train_eval

        Returns
        ----------
        outputs: dict
            度合いの平均
        """
        # 度合いを求めて収集
        scores = self.model(batch[0])
        self.scores.append(scores)
        outputs = {"loss": scores.mean()}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        エポック全体の度合いの最大値を返す関数

        Parameters
        ----------
        phase: str
            valid または train_eval

        Returns
        ----------
        metrics: dict
            度合いの最大値
        """
        # 集めた度合いの最大値
        metrics = {"max_score": float(torch.cat(self.scores).max())}

        return metrics


def make_model() -> DistanceModel:
    """
    make_loader の学習用の行から距離のモデルを作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    model: DistanceModel
        中心と精度行列をバッファに持つモデル
    """
    # 学習用の行の全てから中心と精度行列を計算
    x = torch.cat([batch[0] for batch in make_loader()])
    model = DistanceModel(x)

    return model


def read_history(run_dir: Path) -> list[dict]:
    """
    history.csv の行を読む関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ

    Returns
    ----------
    rows: list[dict]
        列の名前ごとの値の辞書の並び
    """
    with open(run_dir / "history.csv", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    return rows


def test_fit_without_parameters(tmp_path: Path) -> None:
    """
    パラメータを持たないモデルの fit が, Optimizer と Scheduler を作らずに
    検証だけを数えて他の実験と同じ形で残すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # Optimizer と Scheduler を書いた config で 1 エポック
    experiment = make_experiment(
        tmp_path,
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
        "scheduler:\n  name: StepLR\n  params:\n    step_size: 1\n"
        "trainer:\n  epochs: 1\n"
    )
    model = make_model()
    expected = {
        key: value.clone() for key, value in model.state_dict().items()
    }
    trainer = DistanceTrainer(model, experiment)
    trainer.fit(make_loader(), make_loader())
    run_dir = experiment.run_dir

    # Optimizer と Scheduler は無く, 更新は 0 回, 検証のフェーズだけ
    assert trainer.optimizer is None
    assert trainer.scheduler is None
    assert trainer.global_step == 0
    assert trainer.phases == ["valid"]

    # run.json は完了で, Optimizer の並びは空
    status = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["optimizers"] == []
    assert status["completed_epochs"] == 1
    assert status["best_epoch"] == 1
    assert status["monitor"] == "valid/loss"

    # history.csv に検証の指標があり, 学習と学習率の列は無し
    rows = read_history(run_dir)
    assert list(rows[0]) == ["epoch", "valid/loss", "valid/max_score"]
    assert float(rows[0]["valid/loss"]) == pytest.approx(
        status["best_value"]
    )

    # final_model.pt を読み直すとバッファが同じ値
    loaded = torch.load(run_dir / "final_model.pt", weights_only=True)
    assert loaded.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(loaded[key], value)

    # 使わないことを train.log に 1 行
    log = (run_dir / "train.log").read_text(encoding="utf-8")
    assert log.count("Optimizer と Scheduler を作りません") == 1
    assert "trainer.epochs" not in log


def test_epochs_other_than_one(tmp_path: Path) -> None:
    """
    trainer.epochs が 1 でない場合も止まらずに毎エポック同じ検証をし,
    train.log に警告を書くことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 3 エポックで作ると警告
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 3\n")
    with pytest.warns(UserWarning, match="trainer.epochs"):
        trainer = DistanceTrainer(make_model(), experiment)
    trainer.fit(make_loader(), make_loader())

    # 毎エポック同じ検証の値
    rows = read_history(experiment.run_dir)
    assert [row["epoch"] for row in rows] == ["1", "2", "3"]
    assert len({row["valid/loss"] for row in rows}) == 1
    assert len({row["valid/max_score"] for row in rows}) == 1
    assert trainer.phases == ["valid"] * 3

    # 警告は train.log に 1 行
    log = (experiment.run_dir / "train.log").read_text(encoding="utf-8")
    assert log.count("WARNING - パラメータを持たないモデルは学習しない") == 1


def test_named_optimizers_and_eval_train(tmp_path: Path) -> None:
    """
    optimizers を書いた config でも止まらずに作らず, 評価し直しを
    数えることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 名前付きの Optimizer と評価し直しを書いた config
    experiment = make_experiment(
        tmp_path,
        "optimizers:\n  main:\n    name: Adam\n    modules: [body]\n"
        "trainer:\n  epochs: 1\n  eval_train: true\n"
    )
    trainer = DistanceTrainer(make_model(), experiment)
    trainer.fit(make_loader(), make_loader())

    # Optimizer は無く, 評価し直しと検証の指標
    assert trainer.optimizer is None
    assert trainer.phases == ["train_eval", "valid"]
    rows = read_history(experiment.run_dir)
    assert set(rows[0]) == {
        "epoch", "train_eval/loss", "train_eval/max_score", "valid/loss",
        "valid/max_score"
    }


def test_requires_valid_loader(tmp_path: Path) -> None:
    """
    検証用のデータを渡さない場合に, 止まって run.json に失敗を残すことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 学習用のデータだけを渡すとエラー
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 1\n")
    trainer = DistanceTrainer(make_model(), experiment)
    with pytest.raises(ValueError, match="valid_loader"):
        trainer.fit(make_loader())

    # run.json は失敗
    status = json.loads(
        (experiment.run_dir / "run.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"


def test_rejects_passed_optimizer(tmp_path: Path) -> None:
    """
    自分で作った Optimizer を渡すと, Trainer を作るときに止まることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 別のテンソルを更新する Optimizer を渡すとエラー
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 1\n")
    other = optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
    with pytest.raises(ValueError, match="optimizer と scheduler"):
        DistanceTrainer(make_model(), experiment, optimizer=other)


def test_resume_without_parameters(tmp_path: Path) -> None:
    """
    パラメータを持たないモデルの last.pt から同じモデルで再開でき,
    パラメータを持つモデルでは止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 1 エポックで last.pt を保存
    first = make_experiment(tmp_path / "first", "trainer:\n  epochs: 1\n")
    DistanceTrainer(make_model(), first).fit(make_loader(), make_loader())
    last = first.run_dir / "last.pt"

    # 同じモデルで 2 エポックまで再開 (エポック数の警告は除外)
    second = make_experiment(
        tmp_path / "second",
        f"trainer:\n  epochs: 2\n  resume_from: {last}\n"
    )
    with pytest.warns(UserWarning, match="trainer.epochs"):
        trainer = DistanceTrainer(make_model(), second)
    history = trainer.fit(make_loader(), make_loader())
    assert history.get("epoch") == [1, 2]
    assert history.get("valid/loss")[0] == pytest.approx(
        history.get("valid/loss")[1]
    )

    # パラメータを持つモデルでは再開できない
    third = make_experiment(
        tmp_path / "third",
        f"trainer:\n  epochs: 1\n  resume_from: {last}\n"
    )

    def linear_step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        出力の平均を損失として返す関数

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
            出力の平均
        """
        # 出力の平均
        outputs = {"loss": model(batch[0]).mean()}

        return outputs

    linear = Trainer(nn.Linear(1, 1), third, linear_step)
    with pytest.raises(ValueError, match="パラメータを持たないモデルかどうか"):
        linear.fit(make_loader(), make_loader())


def test_model_with_parameters_unchanged(tmp_path: Path) -> None:
    """
    パラメータを持つモデルは, 今までどおり Optimizer を作って学習することを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 凍結したパラメータだけのモデルも Optimizer を作り, 学習のフェーズを回す
    experiment = make_experiment(tmp_path, "trainer:\n  epochs: 1\n")
    model = nn.Linear(1, 1)
    model.requires_grad_(False)

    def frozen_step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        勾配の流れない損失を返す関数

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
            勾配を持つ 0 を足した出力の平均
        """
        # 逆伝播できるように勾配を持つ 0 を加算
        zero = torch.zeros((), requires_grad=True)
        outputs = {"loss": model(batch[0]).mean() + zero}

        return outputs

    trainer = Trainer(model, experiment, frozen_step)
    history = trainer.fit(make_loader(), make_loader())
    assert isinstance(trainer.optimizer, optim.Adam)
    assert "train/loss" in history.records[0]
    assert "lr" in history.records[0]
