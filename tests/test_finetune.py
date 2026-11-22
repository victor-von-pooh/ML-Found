from pathlib import Path
from typing import Iterator

import pytest
import torch
from torch import nn

from ml_found.experiment.setup import check_precision
from ml_found.training.trainer import Trainer

from test_trainer import (
    make_experiment, make_loader, regression_step, same_weights
)

# 学習の設定 (SGD)
SGD = "optimizer:\n  name: SGD\n  params:\n    lr: 0.1\n"


def read_weights(path: Path) -> dict:
    """
    cpu にチェックポイントか重みを読み込む関数

    Parameters
    ----------
    path: Path
        読むファイル

    Returns
    ----------
    state: dict
        読み込んだ内容
    """
    # テンソルと基本的な型だけを cpu に読み込み
    state = torch.load(path, map_location="cpu", weights_only=True)

    return state


def close_weights(left: dict, right: dict) -> bool:
    """
    2 つの重みが丸めの誤差の範囲で等しいかを確かめる関数

    Parameters
    ----------
    left: dict
        比べる重み
    right: dict
        比べる重み

    Returns
    ----------
    close: bool
        名前が同じで, 値が全て近いかどうか
    """
    # 名前と値を全て比較
    close = left.keys() == right.keys() and all(
        torch.allclose(left[key], right[key], atol=1e-6) for key in left
    )

    return close


class NoLenLoader:
    def __init__(self, batch_size: int):
        # バッチ数を持たない loader の中身として DataLoader を保存
        self.loader = make_loader(n=48, batch_size=batch_size)

    def __iter__(self) -> Iterator:
        """
        バッチを順に返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        batches: Iterator
            DataLoader のバッチ
        """
        # 中の DataLoader のバッチをそのまま返却
        batches = iter(self.loader)

        return batches


def make_backbone_model() -> nn.Module:
    """
    凍結した骨格と学習する頭を持つモデルを作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    model: nn.Module
        backbone (凍結, BatchNorm を含む) と head (学習) を持つモデル
    """
    # 骨格の全てのパラメータを凍結
    model = nn.Sequential()
    model.add_module(
        "backbone", nn.Sequential(nn.Linear(1, 4), nn.BatchNorm1d(4))
    )
    model.add_module("head", nn.Linear(4, 1))
    model.backbone.requires_grad_(False)

    return model


def save_pretrained(tmp_path: Path) -> Path:
    """
    学習済みの骨格の重みを, 別の名前の前置きで保存する関数

    Parameters
    ----------
    tmp_path: Path
        保存先のディレクトリ

    Returns
    ----------
    path: Path
        encoder. で始まるキーと, 使わない classifier. のキーを持つファイル
    """
    # 決まった値の骨格の重みを作り, 前置きを付けて保存
    torch.manual_seed(123)
    backbone = nn.Sequential(nn.Linear(1, 4), nn.BatchNorm1d(4))
    weights = {
        f"encoder.{key}": value for key, value in backbone.state_dict().items()
    }
    weights["classifier.weight"] = torch.zeros(3, 4)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "pretrained.pt"
    torch.save(weights, path)

    return path


def init_text(path: Path) -> str:
    """
    骨格に学習済みの重みを読む trainer.init_weights の YAML を作る関数

    Parameters
    ----------
    path: Path
        学習済みの重みのファイル

    Returns
    ----------
    text: str
        trainer の下に置く YAML
    """
    # encoder. を backbone. に付け替えて読む設定
    text = (
        "  init_weights:\n"
        f"    path: {path}\n"
        "    source_prefix: encoder.\n"
        "    target_prefix: backbone.\n"
    )

    return text


def test_accumulation_matches_large_batch(tmp_path: Path) -> None:
    """
    勾配の蓄積が, 貯めた数を合わせた大きなバッチの学習と一致することを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 16 のバッチで毎回更新
    large = make_experiment(
        tmp_path / "large", SGD + "trainer:\n  epochs: 2\n"
    )
    torch.manual_seed(0)
    plain = Trainer(nn.Linear(1, 1), large, regression_step)
    plain.fit(make_loader(batch_size=16))

    # 8 のバッチで 2 回ずつ貯めて更新
    small = make_experiment(
        tmp_path / "small",
        SGD + "trainer:\n  epochs: 2\n  accumulate_grad_batches: 2\n"
    )
    torch.manual_seed(0)
    accumulated = Trainer(nn.Linear(1, 1), small, regression_step)
    accumulated.fit(make_loader(batch_size=8))

    # 重みと更新回数が同じ
    assert close_weights(
        accumulated.model.state_dict(), plain.model.state_dict()
    )
    assert accumulated.global_step == plain.global_step == 8


@pytest.mark.parametrize("with_len", [True, False])
def test_accumulation_remainder(tmp_path: Path, with_len: bool) -> None:
    """
    エポックの最後の端数のバッチも, 貯めた数の平均の勾配で更新することを
    確かめる関数 (バッチ数を持たない loader ではエポックの終わりに更新)

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    with_len: bool
        loader がバッチ数を持つかどうか

    Returns
    ----------
    None
    """
    # 48 のサンプルを 1 つのバッチで更新
    whole = make_experiment(
        tmp_path / "whole", SGD + "trainer:\n  epochs: 2\n"
    )
    torch.manual_seed(0)
    plain = Trainer(nn.Linear(1, 1), whole, regression_step)
    plain.fit(make_loader(n=48, batch_size=48))

    # 16 のバッチ 3 つを, 4 つ貯める設定で更新 (毎エポック端数の 1 回)
    text = SGD + (
        "trainer:\n  epochs: 2\n  accumulate_grad_batches: 4\n"
        "  log_grad_norm: true\n"
    )
    experiment = make_experiment(tmp_path / "accumulated", text)
    torch.manual_seed(0)
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    loader = make_loader(n=48, batch_size=16) if with_len else NoLenLoader(16)
    trainer.fit(loader)

    # 重みと更新回数が同じで, 勾配のノルムも記録
    assert close_weights(trainer.model.state_dict(), plain.model.state_dict())
    assert trainer.global_step == 2
    assert all(
        "train/grad_norm" in record for record in trainer.history.records
    )


def test_accumulation_scheduler_steps(tmp_path: Path) -> None:
    """
    step ごとの Scheduler の総ステップ数を, 更新の回数で補うことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 4 つのバッチを 2 つずつ貯めると 1 エポックに 2 回の更新
    text = SGD + (
        "scheduler:\n  name: CosineAnnealingLR\n  interval: step\n"
        "  params:\n    T_max: auto\n"
        "trainer:\n  epochs: 3\n  accumulate_grad_batches: 2\n"
    )
    experiment = make_experiment(tmp_path, text)
    trainer = Trainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fit(make_loader(batch_size=16))

    # 補った T_max は全体の更新の回数
    assert experiment.cfg.scheduler.params.T_max == 6
    assert trainer.global_step == 6


@pytest.mark.parametrize(
    "text",
    [
        "trainer:\n  accumulate_grad_batches: 0\n",
        "trainer:\n  accumulate_grad_batches: true\n",
        "trainer:\n  accumulate_grad_batches: 2\n"
        "  manual_optimization: true\n",
        "optimizer:\n  name: LBFGS\n  params:\n    lr: 0.1\n"
        "trainer:\n  accumulate_grad_batches: 2\n",
        "trainer:\n  precision: fp16\n",
        "trainer:\n  init_weights:\n    path: \"\"\n"
    ]
)
def test_finetune_config_check(tmp_path: Path, text: str) -> None:
    """
    勾配の蓄積, 計算の精度, 始める前に読む重みの誤りが, 出力先を作る前に
    見つかることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    text: str
        誤りを含む config

    Returns
    ----------
    None
    """
    # 誤りで止まり, 出力先は作られない
    with pytest.raises(ValueError):
        make_experiment(tmp_path, text)
    assert not (tmp_path / "outputs").exists()


def test_bf16_autocast(tmp_path: Path) -> None:
    """
    bf16 では学習と検証の順伝播が bfloat16 で計算され, 重みは float32 の
    ままであることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # フェーズごとの予測の型を記録する step
    dtypes: dict[str, set] = {}

    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        予測の型を記録して平均二乗誤差を求める関数

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
        x, y = batch
        pred = model(x)
        dtypes.setdefault(phase, set()).add(pred.dtype)
        outputs = {"loss": nn.MSELoss()(pred.float(), y)}

        return outputs

    experiment = make_experiment(
        tmp_path, SGD + "trainer:\n  epochs: 1\n  precision: bf16\n"
    )
    trainer = Trainer(nn.Linear(1, 1), experiment, step)
    trainer.fit(make_loader(), make_loader())

    # 学習と検証は bfloat16, 重みと final_model.pt は float32
    assert dtypes == {"train": {torch.bfloat16}, "valid": {torch.bfloat16}}
    final = read_weights(experiment.run_dir / "final_model.pt")
    assert all(value.dtype == torch.float32 for value in final.values())


def test_bf16_unsupported_device() -> None:
    """
    bfloat16 で計算しないデバイスでは, 精度の確認が止めることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # cpu は使え, 自動の混合精度を持たないデバイスの種類は止まる
    cfg = type("Cfg", (), {})()
    cfg.trainer = type("TrainerCfg", (), {"precision": "bf16"})()
    check_precision(cfg, torch.device("cpu"))
    with pytest.raises(ValueError, match="bf16"):
        check_precision(cfg, torch.device("meta"))


def test_resume_matches_with_accumulation_and_bf16(tmp_path: Path) -> None:
    """
    勾配の蓄積と bf16 で再開した学習が, 通しの学習と一致することを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 通しで 4 エポック (端数の出る 5 つのバッチを 2 つずつ貯める)
    options = "  accumulate_grad_batches: 2\n  precision: bf16\n"
    whole = make_experiment(
        tmp_path / "whole", SGD + "trainer:\n  epochs: 4\n" + options
    )
    trainer = Trainer(nn.Linear(1, 1), whole, regression_step)
    trainer.fit(make_loader(n=80, shuffle=True), make_loader())

    # 2 エポックで止めてから, last.pt から 4 エポックまで再開
    first = make_experiment(
        tmp_path / "first", SGD + "trainer:\n  epochs: 2\n" + options
    )
    Trainer(nn.Linear(1, 1), first, regression_step).fit(
        make_loader(n=80, shuffle=True), make_loader()
    )
    resumed = make_experiment(
        tmp_path / "resumed",
        SGD + "trainer:\n  epochs: 4\n"
        f"  resume_from: {first.run_dir / 'last.pt'}\n" + options
    )
    again = Trainer(nn.Linear(1, 1), resumed, regression_step)
    again.fit(make_loader(n=80, shuffle=True), make_loader())

    # 記録, 更新回数, final_model.pt が一致
    assert again.history.records == trainer.history.records
    assert again.global_step == trainer.global_step == 12
    assert same_weights(
        read_weights(resumed.run_dir / "final_model.pt"),
        read_weights(whole.run_dir / "final_model.pt")
    )


def test_init_weights_partial(tmp_path: Path) -> None:
    """
    trainer.init_weights で, ファイルの重みの一部を前置きを付け替えて読む
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 骨格だけを学習済みの重みから読む
    path = save_pretrained(tmp_path / "weights")
    experiment = make_experiment(
        tmp_path / "run", SGD + "trainer:\n  epochs: 1\n" + init_text(path)
    )
    trainer = Trainer(make_backbone_model(), experiment, regression_step)

    # 骨格は学習済みの値, 頭と使わないキーは報告に残る
    weights = read_weights(path)
    state = trainer.model.state_dict()
    for key, value in weights.items():
        if key.startswith("encoder."):
            target = "backbone." + key[len("encoder."):]
            assert torch.equal(state[target], value)
    report = trainer.init_weights_report
    assert sorted(report["missing"]) == ["head.bias", "head.weight"]
    assert report["unused"] == []
    log = (experiment.run_dir / "train.log").read_text(encoding="utf-8")
    assert "trainer.init_weights.path" in log


@pytest.mark.parametrize(
    "options, message",
    [
        ("    strict: true\n", "そろいません"),
        ("    source_prefix: nothing.\n", "始まる重みがありません"),
        ("    target_prefix: head.\n", "形がモデルと違います")
    ]
)
def test_init_weights_errors(
    tmp_path: Path, options: str, message: str
) -> None:
    """
    trainer.init_weights のキーがそろわない場合, 前置きに当たらない場合,
    形が違う場合に止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    options: str
        trainer.init_weights に足す YAML
    message: str
        誤りの文に含まれる文

    Returns
    ----------
    None
    """
    # 骨格の前置きを外さずに頭に読むと形が違い, strict では頭が足りない
    path = save_pretrained(tmp_path / "weights")
    text = f"  init_weights:\n    path: {path}\n" + options
    if "prefix" not in options:
        text += "    source_prefix: encoder.\n    target_prefix: backbone.\n"
    elif "target_prefix" in options:
        text += "    source_prefix: encoder.0.\n"
    experiment = make_experiment(
        tmp_path / "run", SGD + "trainer:\n  epochs: 1\n" + text
    )
    with pytest.raises(ValueError, match=message):
        Trainer(make_backbone_model(), experiment, regression_step)


def test_init_weights_missing_file(tmp_path: Path) -> None:
    """
    trainer.init_weights.path のファイルが無い場合は, 出力先を作る前に止まる
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 無いファイルで止まり, 出力先は作られない
    with pytest.raises(FileNotFoundError):
        make_experiment(
            tmp_path, "trainer:\n  init_weights:\n    path: nosuch.pt\n"
        )
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("ema", [False, True])
def test_trainable_only_resume(tmp_path: Path, ema: bool) -> None:
    """
    学習するパラメータだけのチェックポイントが軽くなり, そこから再開した
    学習が通しの学習と一致することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    ema: bool
        EMA も使うかどうか

    Returns
    ----------
    None
    """
    # 骨格を学習済みの重みにし, 学習するパラメータだけを保存する設定
    path = save_pretrained(tmp_path / "weights")
    options = init_text(path) + "  checkpoint:\n    trainable_only: true\n"
    if ema:
        options += "  ema:\n    decay: 0.5\n"

    # 通しで 4 エポック
    whole = make_experiment(
        tmp_path / "whole", SGD + "trainer:\n  epochs: 4\n" + options
    )
    trainer = Trainer(make_backbone_model(), whole, regression_step)
    trainer.fit(make_loader(shuffle=True), make_loader())

    # 2 エポックの last.pt は凍結した重みを持たず, 除いた名前を持つ
    first = make_experiment(
        tmp_path / "first", SGD + "trainer:\n  epochs: 2\n" + options
    )
    Trainer(make_backbone_model(), first, regression_step).fit(
        make_loader(shuffle=True), make_loader()
    )
    last = read_weights(first.run_dir / "last.pt")
    frozen = {
        "backbone.0.weight", "backbone.0.bias", "backbone.1.weight",
        "backbone.1.bias"
    }
    assert set(last["trainable_only"]["frozen"]) == frozen
    assert not frozen & set(last["model"])
    assert "backbone.1.running_mean" in last["model"]
    assert "head.weight" in last["model"]
    if ema:
        assert not {f"module.{key}" for key in frozen} & set(
            last["ema"]["state"]
        )

    # last.pt から 4 エポックまで再開すると, 記録と final_model.pt が一致
    resumed = make_experiment(
        tmp_path / "resumed",
        SGD + "trainer:\n  epochs: 4\n"
        f"  resume_from: {first.run_dir / 'last.pt'}\n" + options
    )
    again = Trainer(make_backbone_model(), resumed, regression_step)
    again.fit(make_loader(shuffle=True), make_loader())
    assert again.history.records == trainer.history.records
    final = read_weights(resumed.run_dir / "final_model.pt")
    assert same_weights(
        final, read_weights(whole.run_dir / "final_model.pt")
    )
    assert frozen <= set(final)


def test_trainable_only_resume_needs_frozen(tmp_path: Path) -> None:
    """
    学習するパラメータだけのチェックポイントは, 除いたパラメータを持たない
    モデルでは再開できないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 骨格を凍結して 1 エポック学習
    options = "  checkpoint:\n    trainable_only: true\n"
    first = make_experiment(
        tmp_path / "first", SGD + "trainer:\n  epochs: 1\n" + options
    )
    Trainer(make_backbone_model(), first, regression_step).fit(
        make_loader(), make_loader()
    )

    # 骨格の形が違うモデルでは補えずに止まる
    resumed = make_experiment(
        tmp_path / "resumed",
        SGD + "trainer:\n  epochs: 2\n"
        f"  resume_from: {first.run_dir / 'last.pt'}\n"
    )
    model = make_backbone_model()
    model.backbone[0] = nn.Linear(1, 4, bias=False)
    with pytest.raises(ValueError, match="補えません"):
        Trainer(model, resumed, regression_step).fit(
            make_loader(), make_loader()
        )


def test_trainable_only_best_from_best_pt(tmp_path: Path) -> None:
    """
    last.pt にベストモデルの重みを入れない設定でも, 学習するパラメータだけの
    best.pt からベストモデルの重みを戻せることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 学習率を大きくして, 後のエポックで悪くなる学習
    text = (
        "optimizer:\n  name: SGD\n  params:\n    lr: 1.5\n"
        "trainer:\n  epochs: 3\n  checkpoint:\n    trainable_only: true\n"
        "    last_with_best: false\n"
    )
    first = make_experiment(tmp_path / "first", text)
    Trainer(make_backbone_model(), first, regression_step).fit(
        make_loader(), make_loader()
    )
    best = read_weights(first.run_dir / "best.pt")
    assert "trainable_only" in best
    assert "backbone.0.weight" not in best["model"]

    # 再開して, ベストモデルの重みが全ての重みで戻る
    resumed = make_experiment(
        tmp_path / "resumed",
        text.replace("epochs: 3", "epochs: 3\n  resume_from: "
                     f"{first.run_dir / 'last.pt'}")
    )
    again = Trainer(make_backbone_model(), resumed, regression_step)
    again.fit(make_loader(), make_loader())
    log = (resumed.run_dir / "train.log").read_text(encoding="utf-8")
    assert "best.pt から戻しました" in log
    restored = again.checkpoint.best_model_state
    assert restored is not None
    assert set(restored) == set(again.model.state_dict())


def test_init_from_trainable_only_checkpoint(tmp_path: Path) -> None:
    """
    学習するパラメータだけのチェックポイントを, 新しい学習の始める前に読む
    重みにできることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 学習するパラメータだけの last.pt を作る
    first = make_experiment(
        tmp_path / "first",
        SGD + "trainer:\n  epochs: 1\n"
        "  checkpoint:\n    trainable_only: true\n"
    )
    trainer = Trainer(make_backbone_model(), first, regression_step)
    trainer.fit(make_loader())

    # その頭の重みを新しい学習に読む
    second = make_experiment(
        tmp_path / "second",
        SGD + "trainer:\n  epochs: 1\n  init_weights:\n"
        f"    path: {first.run_dir / 'last.pt'}\n"
    )
    loaded = Trainer(make_backbone_model(), second, regression_step)
    assert torch.equal(
        loaded.model.head.weight.detach().cpu(),
        read_weights(first.run_dir / "last.pt")["model"]["head.weight"]
    )
    assert "backbone.0.weight" in loaded.init_weights_report["missing"]


def test_trainable_only_weights_only_best(tmp_path: Path) -> None:
    """
    重みだけの best.pt も, EMA のベストモデルの重みから学習しないパラメータを
    省くことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # EMA と重みだけの best.pt で, 学習するパラメータだけを保存
    text = SGD + (
        "trainer:\n  epochs: 2\n  ema:\n    decay: 0.5\n"
        "  checkpoint:\n    trainable_only: true\n"
        "    best_weights_only: true\n"
    )
    experiment = make_experiment(tmp_path, text)
    Trainer(make_backbone_model(), experiment, regression_step).fit(
        make_loader(), make_loader()
    )

    # best.pt は頭とバッファだけ, final_model.pt は全ての重み
    best = read_weights(experiment.run_dir / "best.pt")
    assert best["weights_only"]
    assert "backbone.0.weight" in best["trainable_only"]["frozen"]
    assert "backbone.0.weight" not in best["model"]
    assert "head.weight" in best["model"]
    final = read_weights(experiment.run_dir / "final_model.pt")
    assert "backbone.0.weight" in final
