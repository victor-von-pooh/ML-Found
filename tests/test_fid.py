import importlib.machinery
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from ml_found import (
    FrechetInceptionDistance, MissingRequirementError, Trainer,
    describe_requirements, environment_packages, frechet_distance
)
from ml_found.training import fid as fid_module
from test_requirements import hide_modules
from test_trainer import make_experiment, make_loader, regression_step


def pooled_model() -> nn.Module:
    """
    重みを取らずに使える小さな特徴のモデルを作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    model: nn.Module
        (N, 3, H, W) を 2 x 2 に平均して (N, 12) にするモデル
    """
    # 平均のプーリングと平らにする層
    model = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Flatten())

    return model


def make_images(n: int, shift: float = 0.0, seed: int = 0) -> torch.Tensor:
    """
    [0, 1] の乱数の画像を作る関数

    Parameters
    ----------
    n: int
        枚数
    shift: float = 0.0
        明るさのずらし
    seed: int = 0
        乱数の種

    Returns
    ----------
    images: torch.Tensor
        (n, 3, 8, 8) の画像
    """
    # 種を決めた一様乱数を縮めてずらす
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand(n, 3, 8, 8, generator=generator) * 0.5 + shift

    return images


def test_frechet_distance_diagonal() -> None:
    """
    対角の共分散で, 式から求めた値と一致することを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 対角の場合は |mean の差|^2 + sum (sqrt(a) - sqrt(b))^2
    mean1, mean2 = torch.tensor([0.0, 1.0]), torch.tensor([1.0, 3.0])
    var1, var2 = torch.tensor([1.0, 4.0]), torch.tensor([9.0, 1.0])
    expected = 1.0 + 4.0 + (1.0 - 3.0) ** 2 + (2.0 - 1.0) ** 2
    distance = frechet_distance(
        mean1, torch.diag(var1), mean2, torch.diag(var2)
    )
    assert distance == pytest.approx(expected)

    # 同じ分布では 0
    assert frechet_distance(
        mean1, torch.diag(var1), mean1, torch.diag(var1)
    ) == pytest.approx(0.0, abs=1e-9)


def test_frechet_distance_rotation() -> None:
    """
    回して対角でなくした共分散でも同じ値になることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 直交行列で両方を回しても距離は変わらない
    generator = torch.Generator().manual_seed(1)
    rotation, _ = torch.linalg.qr(
        torch.randn(3, 3, generator=generator, dtype=torch.float64)
    )
    mean1 = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    mean2 = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64)
    cov1 = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    cov2 = torch.diag(torch.tensor([2.0, 2.0, 0.5], dtype=torch.float64))
    plain = frechet_distance(mean1, cov1, mean2, cov2)
    rotated = frechet_distance(
        rotation @ mean1, rotation @ cov1 @ rotation.T, rotation @ mean2,
        rotation @ cov2 @ rotation.T
    )
    assert rotated == pytest.approx(plain)


def test_frechet_distance_shape_error() -> None:
    """
    形の合わない平均と共分散ではエラーになることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 次元の違う平均
    with pytest.raises(ValueError, match="同じ D"):
        frechet_distance(
            torch.zeros(2), torch.eye(2), torch.zeros(3), torch.eye(2)
        )


def test_fid_with_feature_model() -> None:
    """
    渡した特徴のモデルで, 同じ画像は 0, ずれた画像は大きくなることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 同じ画像どうしの FID
    metric = FrechetInceptionDistance(pooled_model())
    images = make_images(32)
    metric.update_real(images)
    metric.update_fake(images)
    assert metric.compute() == pytest.approx(0.0, abs=1e-9)

    # 生成だけ捨てて, 明るさのずれた画像と比べる (本物は使い回す)
    metric.reset_fake()
    assert metric.has_real
    metric.update_fake(make_images(32, shift=0.3, seed=1))
    shifted = metric.compute()
    metric.reset_fake()
    metric.update_fake(make_images(32, seed=1))
    similar = metric.compute()
    assert shifted > similar > 0

    # 両方を捨てると本物も無い
    metric.reset()
    assert not metric.has_real


def test_fid_batches_match_whole() -> None:
    """
    バッチに分けて足しても, まとめて足した場合と同じ値になることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # まとめて足した場合
    real, fake = make_images(40), make_images(40, shift=0.2, seed=2)
    whole = FrechetInceptionDistance(pooled_model())
    whole.update_real(real)
    whole.update_fake(fake)

    # 大きさの違うバッチに分けて足した場合
    split = FrechetInceptionDistance(pooled_model())
    for start, end in [(0, 7), (7, 25), (25, 40)]:
        split.update_real(real[start:end])
        split.update_fake(fake[start:end])
    assert split.compute() == pytest.approx(whole.compute())


def test_fid_grayscale() -> None:
    """
    白黒の画像は 3 つのチャンネルに写した画像と同じ値になることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 1 チャンネルと, 写した 3 チャンネルの比較
    real, fake = make_images(16)[:, :1], make_images(16, 0.2, 3)[:, :1]
    gray = FrechetInceptionDistance(pooled_model())
    gray.update_real(real)
    gray.update_fake(fake)
    color = FrechetInceptionDistance(pooled_model())
    color.update_real(real.repeat(1, 3, 1, 1))
    color.update_fake(fake.repeat(1, 3, 1, 1))
    assert gray.compute() == pytest.approx(color.compute())


@pytest.mark.parametrize(
    "images, message",
    [
        (torch.rand(4, 3, 8), "テンソル"),
        (torch.rand(4, 2, 8, 8), "テンソル"),
        (torch.rand(4, 3, 8, 8) * 2 - 1, r"\[0, 1\]"),
        (torch.full((4, 3, 8, 8), float("nan")), r"\[0, 1\]")
    ]
)
def test_fid_rejects_images(images: torch.Tensor, message: str) -> None:
    """
    形か値の範囲の違う画像ではエラーになることを確かめる関数

    Parameters
    ----------
    images: torch.Tensor
        誤った画像
    message: str
        エラーの文に含まれる文字

    Returns
    ----------
    None
    """
    # 本物として足すときのエラー
    metric = FrechetInceptionDistance(pooled_model())
    with pytest.raises(ValueError, match=message):
        metric.update_real(images)


def test_fid_needs_two_images() -> None:
    """
    本物か生成が 2 枚未満では求めずにエラーになることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 生成が 1 枚だけ
    metric = FrechetInceptionDistance(pooled_model())
    metric.update_real(make_images(4))
    metric.update_fake(make_images(1))
    with pytest.raises(ValueError, match="生成の画像が足りません"):
        metric.compute()

    # 本物が無い
    metric.reset()
    with pytest.raises(ValueError, match="本物の画像が足りません"):
        metric.compute()


def test_fid_requires_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    pytorch-fid が無ければ, 入れ方を示して止まることを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        pytorch_fid を隠すのに使う

    Returns
    ----------
    None
    """
    # 特徴のモデルを渡さない場合は追加の依存 fid を確かめる
    hide_modules(monkeypatch, {"pytorch_fid"})
    with pytest.raises(MissingRequirementError) as caught:
        FrechetInceptionDistance()
    assert caught.value.feature == "fid"
    assert caught.value.missing == ["pytorch_fid"]
    assert "pip install" in str(caught.value)


def test_fid_registry_entry() -> None:
    """
    登録簿の fid が pip の追加の依存 fid で入り, 環境の指定に入ることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 入れ方と補足 (重みの取得)
    entry = describe_requirements(["fid"], check=False)[0]
    assert entry["install"]["kind"] == "pip"
    assert entry["install"]["extra"] == "fid"
    assert entry["install"]["command"] == 'pip install "pytorch-fid"'
    assert "重み" in entry["note"]

    # 環境を作るときの指定
    assert environment_packages(["fid"], include_found=False) == [
        "pytorch-fid"
    ]


class FakeInceptionV3(nn.Module):
    def __init__(self, output_blocks: list[int]):
        # 親クラスのコンストラクタを呼び出す
        super(FakeInceptionV3, self).__init__()

        # 選んだブロックを保存し, 重みを取らない小さな層を用意
        self.output_blocks = output_blocks
        self.conv = nn.Conv2d(3, 4, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """
        pytorch-fid と同じく, ブロックごとの (N, C, 1, 1) の出力の
        リストを返す関数

        Parameters
        ----------
        images: torch.Tensor
            (N, 3, H, W) の画像

        Returns
        ----------
        outputs: list[torch.Tensor]
            1 つのブロックの出力
        """
        # 畳み込みと平均のプーリング
        outputs = [self.pool(self.conv(images))]

        return outputs


def test_default_inception_adapter(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    特徴のモデルを渡さない場合に, pytorch-fid の Inception を選んだ次元の
    ブロックで作り, 出力を 2 次元にすることを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        重みを取らない偽の pytorch_fid を置くのに使う

    Returns
    ----------
    None
    """
    # 偽の pytorch_fid と pytorch_fid.inception (見つかるように spec も)
    package = types.ModuleType("pytorch_fid")
    package.__spec__ = importlib.machinery.ModuleSpec("pytorch_fid", None)
    package.__path__ = []
    inception = types.ModuleType("pytorch_fid.inception")
    inception.InceptionV3 = FakeInceptionV3
    package.inception = inception
    monkeypatch.setitem(sys.modules, "pytorch_fid", package)
    monkeypatch.setitem(sys.modules, "pytorch_fid.inception", inception)

    # 768 次元はブロック 2, 出力は (N, 4)
    metric = FrechetInceptionDistance(dims=768, device="cpu")
    model = metric.feature_model
    assert isinstance(model, fid_module.InceptionFeatures)
    assert model.inception.output_blocks == [2]
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    assert metric.features(make_images(5)).shape == (5, 4)

    # 対応の無い次元
    with pytest.raises(ValueError, match="dims"):
        FrechetInceptionDistance(dims=100)


class FIDTrainer(Trainer):
    def on_epoch_end(self, phase: str) -> dict:
        """
        検証のときだけ, 本物と生成の画像の FID を返す関数

        Parameters
        ----------
        phase: str
            train, train_eval, valid のどれか

        Returns
        ----------
        metrics: dict
            検証の場合は fid, それ以外は空の辞書
        """
        # 検証の他は何もしない
        metrics = {}
        if phase != "valid":
            return metrics

        # 本物は 1 回だけ足し, 生成はエポックごとに作り直す
        if not self.fid.has_real:
            self.fid.update_real(make_images(16))
        self.fid.reset_fake()
        self.fid.update_fake(make_images(16, shift=0.1, seed=self.epoch))
        metrics = {"fid": self.fid.compute()}

        return metrics


def test_fid_recorded_as_metric(tmp_path: Path) -> None:
    """
    on_epoch_end で返した FID が, 検証の間隔のエポックだけ valid/fid として
    記録されることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 4 エポックで間隔 2, valid/fid を監視
    experiment = make_experiment(
        tmp_path,
        "trainer:\n  epochs: 4\n  valid_every_n_epochs: 2\n"
        "  monitor: valid/fid\n"
    )
    trainer = FIDTrainer(nn.Linear(1, 1), experiment, regression_step)
    trainer.fid = FrechetInceptionDistance(pooled_model())
    history = trainer.fit(make_loader(), make_loader())

    # 検証したエポック 2 と 4 だけに値
    values = history.get("valid/fid")
    assert values[0] is None and values[2] is None
    assert values[1] > 0 and values[3] > 0
    header = (experiment.run_dir / "history.csv").read_text().splitlines()[0]
    assert "valid/fid" in header.split(",")
