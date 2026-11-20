import csv
import json
from pathlib import Path
from typing import Optional

import pytest
import torch
from torch import nn, optim

from ml_found.training.multi import NamedSchedulers
from ml_found.training.optim import step_scheduler
from ml_found.training.trainer import Trainer
from test_trainer import make_experiment, make_loader, regression_step


class ScoreTrainer(Trainer):
    def on_epoch_end(self, phase: str) -> dict:
        """
        呼ばれたフェーズを残し, 検証ではエポックごとに決めた指標を返す関数

        Parameters
        ----------
        phase: str
            train, train_eval, valid のどれか

        Returns
        ----------
        metrics: dict
            検証の場合は score, それ以外は空の辞書
        """
        # 呼ばれたエポックとフェーズの記録
        self.calls.append((self.epoch, phase))

        # 検証の場合だけ今のエポックの値を返却
        metrics = {}
        if phase == "valid":
            metrics = {"score": self.scores[self.epoch - 1]}

        return metrics


class Recorder:
    def __init__(self, stop_epoch: Optional[int] = None):
        # 受け取った辞書と, 止める要求を返すエポックを保存
        self.infos: list[dict] = []
        self.stop_epoch = stop_epoch

    def __call__(self, info: dict) -> bool:
        """
        エポックの結果を残し, 決めたエポックで止める要求を返す関数

        Parameters
        ----------
        info: dict
            エポックの結果

        Returns
        ----------
        stop: bool
            決めたエポックなら True
        """
        # 結果の記録と止める要求の判定
        self.infos.append(info)
        stop = info["epoch"] == self.stop_epoch

        return stop


def constant_step(model: nn.Module, batch: tuple, phase: str) -> dict:
    """
    勾配が 0 で常に同じ損失を返す関数

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
        値が 1 の損失
    """
    # モデルの出力に 0 を掛けて, 勾配が 0 の一定の損失
    outputs = {"loss": model(batch[0]).sum() * 0 + 1.0}

    return outputs


def epochs_with(history_records: list[dict], key: str) -> list[int]:
    """
    指標が記録されたエポックを集める関数

    Parameters
    ----------
    history_records: list[dict]
        エポックごとの記録
    key: str
        指標の名前

    Returns
    ----------
    epochs: list[int]
        指標のあるエポック
    """
    # 値のあるエポックだけを抽出
    epochs = [
        record["epoch"] for record in history_records
        if record.get(key) is not None
    ]

    return epochs


def run_scores(
    tmp_path: Path, text: str, scores: list[float]
) -> ScoreTrainer:
    """
    決めておいた検証の指標を小さい方を良いとして学習する関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    text: str
        trainer に加える YAML の内容
    scores: list[float]
        エポックごとの valid/score

    Returns
    ----------
    trainer: ScoreTrainer
        学習を終えた Trainer
    """
    # valid/score を監視して学習
    experiment = make_experiment(
        tmp_path,
        f"trainer:\n  epochs: {len(scores)}\n  monitor: valid/score\n" + text
    )
    trainer = ScoreTrainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.scores = scores
    trainer.calls = []
    trainer.fit(make_loader(), make_loader())

    return trainer


def test_judged_epochs(tmp_path: Path) -> None:
    """
    間隔で割り切れるエポックと最後のエポックだけ検証することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 7 エポックで間隔 3 (比べる指標も検証の指標)
    experiment = make_experiment(
        tmp_path,
        "trainer:\n  epochs: 7\n  valid_every_n_epochs: 3\n"
        "  monitor_tiebreak: valid/mae\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    recorder = Recorder()
    trainer.add_callback(recorder)
    history = trainer.fit(make_loader(), make_loader())

    # 学習は毎エポック, 検証はエポック 3, 6, 7 だけ
    assert history.get("epoch") == [1, 2, 3, 4, 5, 6, 7]
    assert epochs_with(history.records, "train/loss") == [1, 2, 3, 4, 5, 6, 7]
    assert epochs_with(history.records, "valid/loss") == [3, 6, 7]
    assert epochs_with(history.records, "lr") == [1, 2, 3, 4, 5, 6, 7]
    assert history.get("valid/loss")[0] is None

    # 登録した関数には判定したかが渡り, 判定しないエポックは改善なし
    judged = [info["epoch"] for info in recorder.infos if info["judged"]]
    assert judged == [3, 6, 7]
    assert not any(
        info["improved"] for info in recorder.infos if not info["judged"]
    )
    assert trainer.checkpoint.best_epoch in (3, 6, 7)

    # history.csv の判定しないエポックの検証の値は空
    with open(experiment.run_dir / "history.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["valid/loss"] == "" for row in rows] == [
        True, True, False, True, True, False, False
    ]


def test_best_only_among_judged(tmp_path: Path) -> None:
    """
    ベストモデルを判定したエポックの中だけから選ぶことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 毎エポック検証するとエポック 2 が最良
    scores = [9.0, 0.1, 5.0, 4.0, 3.0, 6.0, 7.0]
    every = run_scores(tmp_path / "every", "", scores)
    assert every.checkpoint.best_epoch == 2

    # 間隔 3 ではエポック 2 を検証しないので, 3, 6, 7 のうちのエポック 3
    sparse = run_scores(
        tmp_path / "sparse", "  valid_every_n_epochs: 3\n", scores
    )
    assert sparse.checkpoint.best_epoch == 3
    assert sparse.checkpoint.best_value == 5.0

    # on_epoch_end の valid は検証するエポックだけ, train は毎エポック
    valid_calls = [epoch for epoch, phase in sparse.calls if phase == "valid"]
    train_calls = [epoch for epoch, phase in sparse.calls if phase == "train"]
    assert valid_calls == [3, 6, 7]
    assert train_calls == [1, 2, 3, 4, 5, 6, 7]

    # run.json のベストも判定したエポック
    status = json.loads(
        (sparse.experiment.run_dir / "run.json").read_text(encoding="utf-8")
    )
    assert status["best_epoch"] == 3


def test_early_stopping_counts_validations(tmp_path: Path) -> None:
    """
    検証用のデータが無い場合も, Early Stopping が判定の回数で数えることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 学習率 0 で全く改善しない設定, 間隔 2 で patience 2
    experiment = make_experiment(
        tmp_path,
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.0\n"
        "trainer:\n  epochs: 50\n  valid_every_n_epochs: 2\n"
        "  early_stopping:\n    use: true\n    patience: 2\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    history = trainer.fit(make_loader())

    # エポック 2 が最良, 4 と 6 で改善しないのでエポック 6 で停止
    assert len(history.records) == 6
    assert trainer.checkpoint.best_epoch == 2
    status = json.loads(
        (experiment.run_dir / "run.json").read_text(encoding="utf-8")
    )
    assert status["early_stopped_epoch"] == 6


def test_plateau_skipped_but_step_lr_runs(tmp_path: Path) -> None:
    """
    判定しないエポックで ReduceLROnPlateau は進めず, StepLR は進めることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 損失が一定で, patience 0 の ReduceLROnPlateau (判定のたびに下げる)
    experiment = make_experiment(
        tmp_path / "plateau",
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
        "scheduler:\n  name: ReduceLROnPlateau\n"
        "  params:\n    patience: 0\n    factor: 0.5\n"
        "trainer:\n  epochs: 7\n  valid_every_n_epochs: 3\n"
    )
    history = Trainer(nn.Linear(1, 1), experiment, constant_step).fit(
        make_loader(), make_loader()
    )

    # エポック 3 で最良, 6 で下げるので, 学習率が変わるのはエポック 7 から
    assert history.get("lr") == pytest.approx([0.1] * 6 + [0.05])

    # StepLR は判定に関係なく毎エポック更新
    experiment = make_experiment(
        tmp_path / "step",
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
        "scheduler:\n  name: StepLR\n"
        "  params:\n    step_size: 1\n    gamma: 0.5\n"
        "trainer:\n  epochs: 4\n  valid_every_n_epochs: 3\n"
    )
    history = Trainer(nn.Linear(1, 1), experiment, constant_step).fit(
        make_loader(), make_loader()
    )
    assert history.get("lr") == pytest.approx([0.1, 0.05, 0.025, 0.0125])


def test_step_scheduler_judged_flag() -> None:
    """
    判定しない場合に ReduceLROnPlateau だけを進めず, 指標の無い判定は
    エラーにすることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 2 つの Optimizer に ReduceLROnPlateau と StepLR を付けた入れ物
    first = optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
    second = optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
    plateau = optim.lr_scheduler.ReduceLROnPlateau(first, patience=0)
    step_lr = optim.lr_scheduler.StepLR(second, step_size=1, gamma=0.5)
    schedulers = NamedSchedulers({"a": plateau, "b": step_lr})
    # 順番についての PyTorch の警告を避けるため, 先に更新
    first.step()
    second.step()

    # 判定しない場合は StepLR だけ更新
    step_scheduler(schedulers, None, judged=False)
    assert plateau.last_epoch == 0
    assert step_lr.last_epoch == 1

    # 判定する場合に指標が無ければエラー
    with pytest.raises(ValueError, match="指標の値が必要"):
        step_scheduler(plateau, None)
    with pytest.raises(ValueError, match="指標の値が必要"):
        schedulers.step(None)


def test_resume_across_boundary(tmp_path: Path) -> None:
    """
    間隔の区切りで止めて再開すると通しの学習と同じになり, 途中で区切ると
    その最後のエポックも判定することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # Early Stopping と StepLR を使い, 並べ替えながら通しで 7 エポック
    text = (
        "optimizer:\n  name: Adam\n  params:\n    lr: 0.05\n"
        "scheduler:\n  name: StepLR\n  params:\n    step_size: 2\n"
        "trainer:\n  valid_every_n_epochs: 3\n"
        "  early_stopping:\n    use: true\n    patience: 5\n"
    )
    full = make_experiment(tmp_path / "full", text + "  epochs: 7\n")
    full_trainer = Trainer(nn.Linear(1, 1), full, regression_step)
    full_history = full_trainer.fit(
        make_loader(shuffle=True), make_loader()
    )

    # 3 エポック (区切り) で止めてから 7 エポックまで再開
    first = make_experiment(tmp_path / "first", text + "  epochs: 3\n")
    Trainer(nn.Linear(1, 1), first, regression_step).fit(
        make_loader(shuffle=True), make_loader()
    )
    second = make_experiment(
        tmp_path / "second",
        text + "  epochs: 7\n"
        f"  resume_from: {first.run_dir / 'last.pt'}\n"
    )
    resumed_trainer = Trainer(nn.Linear(1, 1), second, regression_step)
    resumed_history = resumed_trainer.fit(
        make_loader(shuffle=True), make_loader()
    )

    # 記録とベストモデル, Early Stopping の数えた回数が通しと同じ
    assert epochs_with(resumed_history.records, "valid/loss") == [3, 6, 7]
    assert resumed_history.get("train/loss") == pytest.approx(
        full_history.get("train/loss")
    )
    for epoch in (3, 6, 7):
        assert resumed_history.records[epoch - 1]["valid/loss"] == (
            pytest.approx(full_history.records[epoch - 1]["valid/loss"])
        )
    assert resumed_history.get("lr") == pytest.approx(full_history.get("lr"))
    assert (
        resumed_trainer.checkpoint.best_epoch
        == full_trainer.checkpoint.best_epoch
    )
    assert (
        resumed_trainer.early_stopping.counter
        == full_trainer.early_stopping.counter
    )

    # 4 エポックで区切ると, その最後のエポック 4 も判定
    part = make_experiment(tmp_path / "part", text + "  epochs: 4\n")
    Trainer(nn.Linear(1, 1), part, regression_step).fit(
        make_loader(shuffle=True), make_loader()
    )
    rest = make_experiment(
        tmp_path / "rest",
        text + "  epochs: 7\n"
        f"  resume_from: {part.run_dir / 'last.pt'}\n"
    )
    rest_history = Trainer(nn.Linear(1, 1), rest, regression_step).fit(
        make_loader(shuffle=True), make_loader()
    )
    assert epochs_with(rest_history.records, "valid/loss") == [3, 4, 6, 7]


def test_fit_again_uses_absolute_epochs(tmp_path: Path) -> None:
    """
    fit を呼び直した続きの学習でも, 通しのエポックの番号で判定することを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 4 エポックずつ 2 回, 間隔 3
    experiment = make_experiment(
        tmp_path, "trainer:\n  epochs: 4\n  valid_every_n_epochs: 3\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader(), make_loader())
    history = trainer.fit(make_loader(), make_loader())

    # 1 回目は 3 と最後の 4, 2 回目は 6 と最後の 8
    assert epochs_with(history.records, "valid/loss") == [3, 4, 6, 8]


def test_stop_on_unjudged_epoch(tmp_path: Path) -> None:
    """
    判定しないエポックで止めると, 検証せずに止まり, ベストモデルは前の判定の
    ままであることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 間隔 3 でエポック 4 に止める要求
    experiment = make_experiment(
        tmp_path, "trainer:\n  epochs: 7\n  valid_every_n_epochs: 3\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    recorder = Recorder(stop_epoch=4)
    trainer.add_callback(recorder)
    history = trainer.fit(make_loader(), make_loader())

    # エポック 4 で検証せずに止まり, ベストモデルはエポック 3
    assert history.get("epoch") == [1, 2, 3, 4]
    assert "valid/loss" not in history.records[-1]
    assert [info["judged"] for info in recorder.infos] == [
        False, False, True, False
    ]
    assert trainer.checkpoint.best_epoch == 3
    assert trainer.stopped_by_request
    assert (experiment.run_dir / "last.pt").is_file()

    # 1 回も判定しないうちに止めるとベストモデルは無く, 最後の重みを保存
    early = make_experiment(
        tmp_path / "early",
        "trainer:\n  epochs: 7\n  valid_every_n_epochs: 3\n"
    )
    early_trainer = Trainer(nn.Linear(1, 1), early, regression_step)
    early_trainer.add_callback(Recorder(stop_epoch=1))
    early_trainer.fit(make_loader(), make_loader())
    assert early_trainer.final_weights == "last"
    assert not early_trainer.checkpoint.best_epoch


def test_eval_train_follows_interval(tmp_path: Path) -> None:
    """
    学習用データの評価し直しも検証するエポックだけ行うことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 3 エポックで間隔 2
    experiment = make_experiment(
        tmp_path,
        "trainer:\n  epochs: 3\n  valid_every_n_epochs: 2\n"
        "  eval_train: true\n"
    )
    history = Trainer(nn.Linear(1, 1), experiment, regression_step).fit(
        make_loader(), make_loader()
    )

    # 評価し直しと検証はエポック 2 と最後の 3 だけ
    assert epochs_with(history.records, "train_eval/loss") == [2, 3]
    assert epochs_with(history.records, "valid/loss") == [2, 3]


def test_interval_one_matches_default(tmp_path: Path) -> None:
    """
    間隔 1 では, 項目を書かない場合と同じ記録になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # ReduceLROnPlateau, Early Stopping, 評価し直しを使う設定
    text = (
        "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"
        "scheduler:\n  name: ReduceLROnPlateau\n"
        "  params:\n    patience: 0\n    factor: 0.5\n"
        "trainer:\n  epochs: 6\n  eval_train: true\n"
        "  early_stopping:\n    use: true\n    patience: 2\n"
    )
    plain = make_experiment(tmp_path / "plain", text)
    plain_trainer = Trainer(nn.Linear(1, 1), plain, regression_step)
    plain_history = plain_trainer.fit(make_loader(), make_loader())
    one = make_experiment(
        tmp_path / "one", text + "  valid_every_n_epochs: 1\n"
    )
    one_trainer = Trainer(nn.Linear(1, 1), one, regression_step)
    one_history = one_trainer.fit(make_loader(), make_loader())

    # 毎エポック検証し, 記録とベストモデルが一致
    assert epochs_with(plain_history.records, "valid/loss") == list(
        range(1, len(plain_history.records) + 1)
    )
    assert one_history.records == plain_history.records
    assert one_trainer.checkpoint.best_epoch == (
        plain_trainer.checkpoint.best_epoch
    )


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_interval(tmp_path: Path, value: int) -> None:
    """
    1 未満の間隔が出力先を作る前にエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    value: int
        誤った間隔

    Returns
    ----------
    None
    """
    # start_experiment でエラーになり, 出力先は未作成
    with pytest.raises(ValueError, match="valid_every_n_epochs"):
        make_experiment(
            tmp_path, f"trainer:\n  valid_every_n_epochs: {value}\n"
        )
    assert not (tmp_path / "outputs").exists()


def test_non_integer_interval(tmp_path: Path) -> None:
    """
    整数でない間隔が config の読み込みでエラーになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 整数として読めない値は config の読み込みでエラー
    with pytest.raises(ValueError):
        make_experiment(
            tmp_path / "text", "trainer:\n  valid_every_n_epochs: abc\n"
        )
    assert not (tmp_path / "text" / "outputs").exists()


def test_resume_warns_interval_change(tmp_path: Path) -> None:
    """
    再開で検証の間隔を変えると警告が出ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 間隔 2 で 2 エポック学習
    first = make_experiment(
        tmp_path / "first",
        "trainer:\n  epochs: 2\n  valid_every_n_epochs: 2\n"
    )
    Trainer(nn.Linear(1, 1), first, regression_step).fit(
        make_loader(), make_loader()
    )

    # 間隔 3 で 4 エポックまで再開すると警告
    second = make_experiment(
        tmp_path / "second",
        "trainer:\n  epochs: 4\n  valid_every_n_epochs: 3\n"
        f"  resume_from: {first.run_dir / 'last.pt'}\n"
    )
    trainer = Trainer(nn.Linear(1, 1), second, regression_step)
    with pytest.warns(UserWarning, match="検証の間隔が違います"):
        history = trainer.fit(make_loader(), make_loader())
    assert epochs_with(history.records, "valid/loss") == [2, 3, 4]
