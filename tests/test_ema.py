import json
from pathlib import Path

import pytest
import torch
from torch import nn

from ml_found.training.callbacks import load_checkpoint
from ml_found.training.trainer import Trainer

from test_trainer import (
    EMATrainer, make_experiment, make_loader, regression_step, same_weights,
    stop_at
)

# 学習の設定 (SGD と, 減衰率 0.5 の EMA)
SGD = "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
EMA = "  ema:\n    decay: 0.5\n"


def read_weights(path: Path) -> dict:
    """
    cpu に重みを読み込む関数

    Parameters
    ----------
    path: Path
        重みのファイル

    Returns
    ----------
    weights: dict
        読み込んだ重み
    """
    # 重みだけを cpu に読み込み
    weights = torch.load(path, map_location="cpu", weights_only=True)

    return weights


def read_status(run_dir: Path) -> dict:
    """
    run.json を読む関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ

    Returns
    ----------
    status: dict
        実験の状態
    """
    # JSON として読み込み
    status = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    return status


class RawEvalTrainer(Trainer):
    def eval_model(self) -> nn.Module:
        """
        EMA を持ったまま学習しているモデルを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        model: nn.Module
            学習しているモデル
        """
        # 上書きで学習しているモデルを使用
        model = self.model

        return model


def test_ema_matches_subclass(tmp_path: Path) -> None:
    """
    config の EMA が, 継承して書いた EMA と同じ検証, ベストモデル,
    final_model.pt になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 継承して書いた EMA (減衰率 0.5) で学習
    text = SGD + "trainer:\n  epochs: 3\n"
    experiment = make_experiment(tmp_path / "subclass", text)
    subclass = EMATrainer(nn.Linear(1, 1), experiment)
    subclass.fit(make_loader(), make_loader())
    expected = read_weights(experiment.run_dir / "final_model.pt")

    # config の EMA で同じ条件で学習
    experiment = make_experiment(tmp_path / "config", text + EMA)
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    assert trainer.eval_model() is trainer.ema_model.module
    trainer.fit(make_loader(), make_loader())
    run_dir = experiment.run_dir

    # 検証の値, ベストモデル, final_model.pt が一致
    assert trainer.history.records == subclass.history.records
    assert trainer.checkpoint.best_epoch == subclass.checkpoint.best_epoch
    final = read_weights(run_dir / "final_model.pt")
    assert same_weights(final, expected)
    assert same_weights(trainer.model.state_dict(), expected)
    log = (run_dir / "train.log").read_text(encoding="utf-8")
    assert "評価に使うモデル: EMA (decay=0.5, use_buffers=False)" in log

    # last.pt の model は学習しているモデルの重みで, EMA の状態は ema に入る
    last = load_checkpoint(str(run_dir / "last.pt"), torch.device("cpu"))
    assert last["ema"]["decay"] == 0.5
    assert last["ema"]["use_buffers"] is False
    assert int(last["ema"]["state"]["n_averaged"]) == 12
    assert last["extra"] == {}
    assert not same_weights(last["model"], expected)

    # EMA を使わない場合はチェックポイントの ema は None
    experiment = make_experiment(tmp_path / "off", text)
    plain = Trainer(nn.Linear(1, 1), experiment, regression_step)
    assert plain.ema_model is None
    assert plain.eval_model() is plain.model
    plain.fit(make_loader(), make_loader())
    last = load_checkpoint(
        str(experiment.run_dir / "last.pt"), torch.device("cpu")
    )
    assert last["ema"] is None


def test_ema_last_weights(tmp_path: Path) -> None:
    """
    ベストモデルを選ばない場合と重みだけの best.pt でも, EMA の重みが
    残ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # ベストモデルを選ばない場合は最後の EMA の重み
    experiment = make_experiment(
        tmp_path / "last",
        SGD + "trainer:\n  epochs: 3\n  select_best: false\n" + EMA
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader(), make_loader())
    ema_weights = {
        key: value.detach().clone()
        for key, value in trainer.ema_model.module.state_dict().items()
    }
    final = read_weights(experiment.run_dir / "final_model.pt")
    assert same_weights(final, ema_weights)
    assert read_status(experiment.run_dir)["final_weights"] == "last"

    # 重みだけの best.pt の model は EMA のベストモデルの重み
    experiment = make_experiment(
        tmp_path / "weights",
        SGD + "trainer:\n  epochs: 3\n  checkpoint:\n"
        "    best_weights_only: true\n" + EMA
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader(), make_loader())
    best = read_weights(experiment.run_dir / "best.pt")
    final = read_weights(experiment.run_dir / "final_model.pt")
    assert same_weights(best["model"], final)


def test_ema_resume_matches(tmp_path: Path) -> None:
    """
    EMA を使って再開した学習が, 通しの学習と一致することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 通しで 4 エポック
    text = SGD + "trainer:\n  epochs: 4\n" + EMA
    whole = make_experiment(tmp_path / "whole", text)
    trainer = Trainer(nn.Linear(1, 1), whole, regression_step)
    trainer.fit(make_loader(shuffle=True), make_loader())

    # 2 エポックで止めてから, last.pt から 4 エポックまで再開
    first = make_experiment(
        tmp_path / "first", SGD + "trainer:\n  epochs: 2\n" + EMA
    )
    Trainer(nn.Linear(1, 1), first, regression_step).fit(
        make_loader(shuffle=True), make_loader()
    )
    resumed = make_experiment(
        tmp_path / "resumed",
        SGD + "trainer:\n  epochs: 4\n"
        f"  resume_from: {first.run_dir / 'last.pt'}\n" + EMA
    )
    again = Trainer(nn.Linear(1, 1), resumed, regression_step)
    again.fit(make_loader(shuffle=True), make_loader())

    # 記録, final_model.pt, EMA の状態が一致
    assert again.history.records == trainer.history.records
    assert same_weights(
        read_weights(resumed.run_dir / "final_model.pt"),
        read_weights(whole.run_dir / "final_model.pt")
    )
    assert same_weights(
        again.ema_model.state_dict(), trainer.ema_model.state_dict()
    )


def test_ema_resume_mismatch(tmp_path: Path) -> None:
    """
    再開元と EMA の設定が違う場合に, 警告して続けることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # EMA を使わない実験と使う実験を 1 エポックずつ
    plain = make_experiment(tmp_path / "plain", SGD + "trainer:\n  epochs: 1\n")
    plain_trainer = Trainer(nn.Linear(1, 1), plain, regression_step)
    plain_trainer.fit(make_loader(), make_loader())
    with_ema = make_experiment(
        tmp_path / "ema", SGD + "trainer:\n  epochs: 1\n" + EMA
    )
    Trainer(nn.Linear(1, 1), with_ema, regression_step).fit(
        make_loader(), make_loader()
    )

    # EMA の無い再開元からは, 再開した重みから EMA を始める
    # (学習しないので, final_model.pt は再開元の最後の重み)
    resumed = make_experiment(
        tmp_path / "start",
        SGD + "trainer:\n  epochs: 1\n  select_best: false\n"
        f"  resume_from: {plain.run_dir / 'last.pt'}\n" + EMA
    )
    trainer = Trainer(nn.Linear(1, 1), resumed, regression_step)
    with pytest.warns(UserWarning, match="再開した重みから EMA を始めます"):
        trainer.fit(make_loader(), make_loader())
    last = load_checkpoint(
        str(plain.run_dir / "last.pt"), torch.device("cpu")
    )
    final = read_weights(resumed.run_dir / "final_model.pt")
    assert same_weights(final, last["model"])
    assert int(trainer.ema_model.n_averaged) == 0
    log = (resumed.run_dir / "train.log").read_text(encoding="utf-8")
    assert "再開した重みから EMA を始めます" in log

    # EMA を使った再開元から使わずに再開すると, EMA の状態は使わない
    resumed = make_experiment(
        tmp_path / "drop",
        SGD + "trainer:\n  epochs: 2\n"
        f"  resume_from: {with_ema.run_dir / 'last.pt'}\n"
    )
    trainer = Trainer(nn.Linear(1, 1), resumed, regression_step)
    with pytest.warns(UserWarning, match="EMA の状態を使いません"):
        trainer.fit(make_loader(), make_loader())

    # 減衰率を変えた場合は, 戻した EMA の重みから今回の減衰率で続ける
    resumed = make_experiment(
        tmp_path / "decay",
        SGD + "trainer:\n  epochs: 2\n"
        f"  resume_from: {with_ema.run_dir / 'last.pt'}\n"
        "  ema:\n    decay: 0.9\n"
    )
    trainer = Trainer(nn.Linear(1, 1), resumed, regression_step)
    with pytest.warns(UserWarning, match="EMA の設定が違います"):
        trainer.fit(make_loader(), make_loader())
    assert int(trainer.ema_model.n_averaged) == 8


def test_ema_stop_within_epoch(tmp_path: Path) -> None:
    """
    エポックの途中で止めると, EMA も最後に終えたエポックの状態に戻り,
    final_model.pt が途中までの重みにならないことを確かめる関数

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
        SGD + "trainer:\n  epochs: 3\n  select_best: false\n" + EMA
    )
    trainer = Trainer(
        nn.Linear(1, 1), experiment, stop_at(experiment, "train", 6)
    )
    trainer.run_status.progress_interval = 0.0
    trainer.fit(make_loader(), make_loader())
    assert len(trainer.history.records) == 1

    # EMA は last.pt (エポック 1 の後) の状態で, final_model.pt はその重み
    run_dir = experiment.run_dir
    last = load_checkpoint(str(run_dir / "last.pt"), torch.device("cpu"))
    assert same_weights(trainer.ema_model.state_dict(), last["ema"]["state"])
    final = read_weights(run_dir / "final_model.pt")
    expected = {
        key[len("module."):]: value
        for key, value in last["ema"]["state"].items()
        if key.startswith("module.")
    }
    assert same_weights(final, expected)
    assert read_status(run_dir)["final_weights"] == "last"


def test_ema_fit_again(tmp_path: Path) -> None:
    """
    同じ Trainer で fit を呼び直すと, EMA も続きから更新され, 通しの学習と
    一致することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 2 エポックを 2 回と, 4 エポックを 1 回
    twice = make_experiment(
        tmp_path / "twice", SGD + "trainer:\n  epochs: 2\n" + EMA
    )
    trainer = Trainer(nn.Linear(1, 1), twice, regression_step)
    trainer.fit(make_loader(), make_loader())
    trainer.fit(make_loader(), make_loader())
    once = make_experiment(
        tmp_path / "once", SGD + "trainer:\n  epochs: 4\n" + EMA
    )
    whole = Trainer(nn.Linear(1, 1), once, regression_step)
    whole.fit(make_loader(), make_loader())

    # 記録, EMA の状態, final_model.pt が一致
    assert trainer.history.records == whole.history.records
    assert same_weights(
        trainer.ema_model.state_dict(), whole.ema_model.state_dict()
    )
    assert same_weights(
        read_weights(twice.run_dir / "final_model.pt"),
        read_weights(once.run_dir / "final_model.pt")
    )


@pytest.mark.parametrize("use_buffers", [False, True])
def test_ema_buffers(tmp_path: Path, use_buffers: bool) -> None:
    """
    use_buffers に応じて, バッファを写すか平均するかが変わることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    use_buffers: bool
        バッファも平均するかどうか

    Returns
    ----------
    None
    """
    # BatchNorm を持つモデルで学習
    experiment = make_experiment(
        tmp_path,
        SGD + "trainer:\n  epochs: 2\n" + EMA
        + f"    use_buffers: {str(use_buffers).lower()}\n"
    )
    model = nn.Sequential(nn.Linear(1, 4), nn.BatchNorm1d(4), nn.Linear(4, 1))
    trainer = Trainer(model, experiment, regression_step)
    trainer.fit(make_loader(shuffle=True))

    # false では学習しているモデルの統計量と同じ, true では平均した値
    ema_mean = trainer.ema_model.module[1].running_mean
    last = load_checkpoint(
        str(experiment.run_dir / "last.pt"), torch.device("cpu")
    )
    model_mean = last["model"]["1.running_mean"]
    assert torch.equal(ema_mean, model_mean) is not use_buffers


def test_ema_eval_model_override(tmp_path: Path) -> None:
    """
    eval_model を上書きした場合は上書きした方で検証し, EMA は
    チェックポイントにだけ残ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 学習しているモデルで検証する上書きと, EMA を使わない学習を比べる
    text = SGD + "trainer:\n  epochs: 2\n"
    experiment = make_experiment(tmp_path / "override", text + EMA)
    trainer = RawEvalTrainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader(), make_loader())
    plain = make_experiment(tmp_path / "plain", text)
    Trainer(nn.Linear(1, 1), plain, regression_step).fit(
        make_loader(), make_loader()
    )
    assert same_weights(
        read_weights(experiment.run_dir / "final_model.pt"),
        read_weights(plain.run_dir / "final_model.pt")
    )
    last = load_checkpoint(
        str(experiment.run_dir / "last.pt"), torch.device("cpu")
    )
    assert int(last["ema"]["state"]["n_averaged"]) == 8


@pytest.mark.parametrize("decay", ["0", "1", "1.5", "-0.1", "nan"])
def test_ema_decay_check(tmp_path: Path, decay: str) -> None:
    """
    範囲の外の減衰率が, 出力先を作る前にエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    decay: str
        config に書く減衰率

    Returns
    ----------
    None
    """
    # 範囲の外の減衰率で実験を始めるとエラー
    with pytest.raises(ValueError, match="trainer.ema.decay"):
        make_experiment(
            tmp_path, f"trainer:\n  ema:\n    decay: {decay}\n"
        )
    assert not (tmp_path / "outputs").exists()
