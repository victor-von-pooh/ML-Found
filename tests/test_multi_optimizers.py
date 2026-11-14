import csv
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch
from torch import nn, optim

from ml_found.config.loader import load_config
from ml_found.experiment.setup import Experiment, start_experiment
from ml_found.training.multi import split_parameters
from ml_found.training.optim import named_views
from ml_found.training.trainer import Trainer
from test_trainer import make_loader

# 2 つの部分を名前付きの Optimizer で学習する config
NAMED = (
    "optimizers:\n"
    "  first:\n"
    "    name: SGD\n"
    "    params: {lr: 0.1}\n"
    "    modules: [first]\n"
    "  second:\n"
    "    name: Adam\n"
    "    params: {lr: 0.01}\n"
    "    modules: [second]\n"
)


# 2 つの線形層を順につなぐモデル
class TwoParts(nn.Module):
    def __init__(self):
        # 親クラスのコンストラクタを呼び出す
        super(TwoParts, self).__init__()

        # 名前付きの Optimizer がそれぞれ受け持つ 2 つの層を保存
        self.first = nn.Linear(1, 1)
        self.second = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        2 つの層を順に通す関数

        Parameters
        ----------
        x: torch.Tensor
            入力

        Returns
        ----------
        y: torch.Tensor
            出力
        """
        # 1 層目, 2 層目の順に通す
        y = self.second(self.first(x))

        return y


def named_step(model: nn.Module, batch: tuple, phase: str) -> dict:
    """
    名前ごとの損失を返す step の関数

    first は 1 層目の出力を y に近づけ, second は全体の出力を y に近づける

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
        名前ごとの損失と平均絶対誤差
    """
    # 1 層目と全体の出力から, 名前ごとの損失を計算
    x, y = batch
    hidden = model.first(x)
    pred = model.second(hidden)
    outputs = {
        "loss": {
            "first": ((hidden - y) ** 2).mean(),
            "second": ((pred - y) ** 2).mean()
        },
        "mae": (pred - y).abs().mean()
    }

    return outputs


def make_named_experiment(tmp_path: Path, text: str = "") -> Experiment:
    """
    名前付きの Optimizer を使うテスト用の実験を開始する関数

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
    # default と experiment を書いて読み込み
    tmp_path.mkdir(parents=True, exist_ok=True)
    default_path = tmp_path / "default.yaml"
    default_path.write_text(
        "experiment:\n  name: test\n  device: cpu\n  seed: 0\n" + NAMED + text,
        encoding="utf-8"
    )
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text("", encoding="utf-8")
    cfg, diff = load_config(str(default_path), str(experiment_path))
    cfg.trainer.progress_bar = False
    experiment = start_experiment(cfg, diff, str(tmp_path))

    return experiment


def read_rows(path: Path) -> list[dict]:
    """
    CSV を読む関数

    Parameters
    ----------
    path: Path
        CSV のパス

    Returns
    ----------
    rows: list[dict]
        行ごとの辞書
    """
    # 見出しを使って辞書の一覧に変換
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    return rows


def test_named_optimizers_update_own_parts(tmp_path: Path) -> None:
    """
    名前付きの Optimizer が, それぞれの損失の勾配で受け持つ部分だけを
    更新することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 1 バッチだけ学習
    experiment = make_named_experiment(tmp_path, "trainer:\n  epochs: 1\n")
    torch.manual_seed(0)
    model = TwoParts()
    expected = TwoParts()
    expected.load_state_dict(model.state_dict())
    trainer = Trainer(model, experiment, named_step)
    loader = make_loader(n=8, batch_size=8)
    trainer.fit(loader)

    # 期待値: first は first の損失だけ, second は second の損失だけの勾配
    x, y = next(iter(loader))
    hidden = expected.first(x)
    pred = expected.second(hidden)
    first_grads = torch.autograd.grad(
        ((hidden - y) ** 2).mean(), list(expected.first.parameters()),
        retain_graph=True
    )
    second_grads = torch.autograd.grad(
        ((pred - y) ** 2).mean(), list(expected.second.parameters())
    )
    for parameter, grad in zip(expected.first.parameters(), first_grads):
        assert torch.allclose(
            trainer.model.first.state_dict()["weight"]
            if parameter.dim() == 2 else
            trainer.model.first.state_dict()["bias"],
            parameter.detach() - 0.1 * grad
        )
    adam = optim.Adam(expected.second.parameters(), lr=0.01)
    for parameter, grad in zip(expected.second.parameters(), second_grads):
        parameter.grad = grad
    adam.step()
    for key, value in expected.second.state_dict().items():
        assert torch.allclose(trainer.model.second.state_dict()[key], value)


def test_named_records_losses_and_lrs(tmp_path: Path) -> None:
    """
    名前ごとの損失, 合計の損失, 名前ごとの学習率と勾配のノルムを記録する
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # second に step ごとの Scheduler を付けて学習
    experiment = make_named_experiment(
        tmp_path,
        "trainer:\n  epochs: 2\n  log_grad_norm: true\n"
        "  grad_clip_norm: 1.0\n"
    )
    experiment.cfg.optimizers.second.scheduler.name = "StepLR"
    experiment.cfg.optimizers.second.scheduler.interval = "step"
    experiment.cfg.optimizers.second.scheduler.params = {
        "step_size": 1, "gamma": 0.5
    }
    trainer = Trainer(TwoParts(), experiment, named_step)
    trainer.fit(make_loader(), make_loader())

    # 学習曲線の記録
    rows = read_rows(experiment.run_dir / "history.csv")
    for key in (
        "train/loss", "train/loss_first", "train/loss_second",
        "valid/loss_first", "train/grad_norm_first", "train/grad_norm_second",
        "lr_first", "lr_second"
    ):
        assert key in rows[0], key
    loss = float(rows[0]["train/loss"])
    parts = float(rows[0]["train/loss_first"]) + float(
        rows[0]["train/loss_second"]
    )
    assert loss == pytest.approx(parts)
    assert float(rows[0]["lr_first"]) == pytest.approx(0.1)

    # 更新ごとの学習率は名前ごとの列 (second の学習率だけが低下)
    steps = read_rows(experiment.run_dir / "lr_steps.csv")
    assert list(steps[0].keys()) == ["step", "epoch", "lr_first", "lr_second"]
    assert float(steps[-1]["lr_first"]) == pytest.approx(0.1)
    assert float(steps[1]["lr_second"]) == pytest.approx(0.005)


def test_named_resume_matches_uninterrupted(tmp_path: Path) -> None:
    """
    名前付きの Optimizer と Scheduler の状態を保存して再開すると, 通して
    学習した場合と同じ結果になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # first にだけ付ける Scheduler
    scheduler = (
        "    scheduler:\n      name: StepLR\n"
        "      params: {step_size: 1, gamma: 0.5}\n"
    )
    text = NAMED.replace(
        "    modules: [first]\n", "    modules: [first]\n" + scheduler
    )

    def run(path: Path, epochs: int, resume: str = "") -> Path:
        """
        決めたエポック数を学習して実験のディレクトリを返す関数

        Parameters
        ----------
        path: Path
            一時ディレクトリ
        epochs: int
            エポック数
        resume: str = ""
            再開元の last.pt, 空の場合は再開しない

        Returns
        ----------
        run_dir: Path
            実験のディレクトリ
        """
        # default を書いて実験を開始し, 同じ初期値のモデルで学習
        path.mkdir(parents=True, exist_ok=True)
        default_path = path / "default.yaml"
        default_path.write_text(
            "experiment:\n  name: test\n  device: cpu\n  seed: 0\n" + text
            + f"trainer:\n  epochs: {epochs}\n"
            + (f"  resume_from: {resume}\n" if resume else ""),
            encoding="utf-8"
        )
        (path / "experiment.yaml").write_text("", encoding="utf-8")
        cfg, diff = load_config(
            str(default_path), str(path / "experiment.yaml")
        )
        cfg.trainer.progress_bar = False
        experiment = start_experiment(cfg, diff, str(path))
        torch.manual_seed(0)
        Trainer(TwoParts(), experiment, named_step).fit(
            make_loader(), make_loader()
        )

        return experiment.run_dir

    once = run(tmp_path / "once", 4)
    first = run(tmp_path / "first", 2)
    resumed = run(tmp_path / "resumed", 4, str(first / "last.pt"))

    # 再開した学習の記録が, 通した学習と一致
    expected = read_rows(once / "history.csv")
    rows = read_rows(resumed / "history.csv")
    assert len(rows) == 4
    for row, other in zip(rows, expected):
        for key in ("valid/loss", "lr_first", "lr_second"):
            assert float(row[key]) == pytest.approx(float(other[key])), key


def test_named_config_errors(tmp_path: Path) -> None:
    """
    名前付きの Optimizer の設定の誤りがエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 一番上の scheduler との併用
    with pytest.raises(ValueError, match="optimizers.<名前>.scheduler"):
        make_named_experiment(
            tmp_path / "top",
            "scheduler:\n  name: StepLR\n  params: {step_size: 1}\n"
        )

    # 受け持つ部分の無いモデル, 受け持たないパラメータ
    experiment = make_named_experiment(tmp_path / "parts")
    model = TwoParts()
    model.third = nn.Linear(1, 1)
    with pytest.raises(ValueError, match="受け持たない"):
        Trainer(model, experiment, named_step)
    with pytest.raises(ValueError, match="ありません"):
        split_parameters(TwoParts(), {"first": ["missing"]})
    with pytest.raises(ValueError, match="同じパラメータ"):
        split_parameters(
            TwoParts(), {"a": ["first"], "b": ["second", "first"]}
        )

    # 名前の違う損失
    def wrong_step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        名前を書き間違えた損失を返す step の関数

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
            名前を書き間違えた損失
        """
        # second の損失を除いた辞書に置き換え
        outputs = named_step(model, batch, phase)
        outputs["loss"] = {"first": outputs["loss"]["first"]}

        return outputs

    trainer = Trainer(TwoParts(), experiment, wrong_step)
    with pytest.raises(ValueError, match="optimizers の名前"):
        trainer.fit(make_loader())


def test_named_both_styles_error(tmp_path: Path) -> None:
    """
    optimizer と optimizers を両方書くとエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # default に両方
    default_path = tmp_path / "default.yaml"
    default_path.write_text(
        NAMED + "optimizer:\n  name: SGD\n", encoding="utf-8"
    )
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="両方"):
        load_config(str(default_path), str(experiment_path))

    # default は optimizers, experiment は optimizer
    default_path.write_text(NAMED, encoding="utf-8")
    experiment_path.write_text("optimizer:\n  name: SGD\n", encoding="utf-8")
    with pytest.raises(ValueError, match="optimizers.<名前>"):
        load_config(str(default_path), str(experiment_path))


def test_named_freeze_and_unfreeze(tmp_path: Path) -> None:
    """
    学習の途中で凍結したパラメータは更新せず, 最初に凍結していて後で凍結を
    解いたパラメータは更新することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # second の重みは最初は凍結 (学習後の重みを比べるのでベストに戻さない)
    experiment = make_named_experiment(
        tmp_path, "trainer:\n  epochs: 2\n  select_best: false\n"
    )
    model = TwoParts()
    model.second.weight.requires_grad_(False)
    trainer = Trainer(model, experiment, named_step)
    assert trainer.scheduler is not None
    assert trainer.scheduler["first"] is None

    # エポック 1 の後に, first の重みを凍結し second の重みの凍結を解く
    def swap(result: dict) -> None:
        """
        エポック 1 の後に凍結するパラメータを入れ替える関数

        Parameters
        ----------
        result: dict
            エポックの結果

        Returns
        ----------
        None
        """
        if result["epoch"] == 1:
            model.first.weight.requires_grad_(False)
            model.second.weight.requires_grad_(True)
            frozen.update(
                first=model.first.weight.detach().clone(),
                second=model.second.weight.detach().clone()
            )

    frozen: dict = {}
    trainer.add_callback(swap)
    trainer.fit(make_loader())

    # 凍結した first の重みは変わらず, 凍結を解いた second の重みは変化
    assert torch.equal(model.first.weight, frozen["first"])
    assert not torch.equal(model.second.weight, frozen["second"])


def test_named_resume_auto_mismatch(tmp_path: Path) -> None:
    """
    auto で補った Scheduler の引数が再開元と違う場合は, 名前付きの
    Optimizer でもエラーにすることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # first に T_max: auto の Scheduler を付けて 2 エポック学習
    scheduler = (
        "    scheduler:\n      name: CosineAnnealingLR\n"
        "      params: {T_max: auto}\n"
    )
    text = NAMED.replace(
        "    modules: [first]\n", "    modules: [first]\n" + scheduler
    )

    def start(path: Path, extra: str) -> Experiment:
        """
        名前付きの Optimizer と Scheduler の実験を開始する関数

        Parameters
        ----------
        path: Path
            一時ディレクトリ
        extra: str
            default に加える YAML の内容

        Returns
        ----------
        experiment: Experiment
            開始した実験
        """
        # default を書いて実験を開始
        path.mkdir(parents=True, exist_ok=True)
        (path / "default.yaml").write_text(
            "experiment:\n  name: test\n  device: cpu\n" + text + extra,
            encoding="utf-8"
        )
        (path / "experiment.yaml").write_text("", encoding="utf-8")
        cfg, diff = load_config(
            str(path / "default.yaml"), str(path / "experiment.yaml")
        )
        cfg.trainer.progress_bar = False
        experiment = start_experiment(cfg, diff, str(path))

        return experiment

    first = start(tmp_path / "first", "trainer:\n  epochs: 2\n")
    Trainer(TwoParts(), first, named_step).fit(make_loader())

    # エポック数を変えて再開すると, 補った T_max が再開元と違うのでエラー
    resumed = start(
        tmp_path / "resumed",
        f"trainer:\n  epochs: 5\n  resume_from: {first.run_dir / 'last.pt'}\n"
    )
    with pytest.raises(ValueError, match="T_max"):
        Trainer(TwoParts(), resumed, named_step).fit(make_loader())


def test_named_views_resolve_interpolations() -> None:
    """
    名前ごとの config で, 他の項目を参照する値 (${...}) を元の config で
    解決することを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # Optimizer の引数と trainer.epochs が, 他の項目を参照する config
    cfg = OmegaConf.create({
        "model": {"lr": 0.01, "epochs": 3},
        "trainer": {"epochs": "${model.epochs}", "mode": "min"},
        "optimizers": {
            "first": {
                "name": "SGD", "params": {"lr": "${model.lr}"},
                "modules": ["first"],
                "scheduler": {"name": None, "params": {}}
            }
        }
    })
    (name, view), = named_views(cfg)
    assert name == "first"
    assert view.optimizer.params.lr == 0.01
    assert view.trainer.epochs == 3
