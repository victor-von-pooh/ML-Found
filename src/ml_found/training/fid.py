from typing import Optional, Union

import torch
from torch import nn

from ml_found.utils.requirements import require

# pytorch-fid の Inception の特徴の次元と, 特徴を取り出すブロックの番号
INCEPTION_BLOCKS = {64: 0, 192: 1, 768: 2, 2048: 3}


class InceptionFeatures(nn.Module):
    def __init__(self, dims: int = 2048):
        # 親クラスのコンストラクタを呼び出す
        super(InceptionFeatures, self).__init__()

        # 特徴の次元の確認
        if dims not in INCEPTION_BLOCKS:
            raise ValueError(
                f"dims は {sorted(INCEPTION_BLOCKS)} のどれかにしてください"
                f" ({dims!r})"
            )

        # pytorch-fid の Inception (初めて作るときに学習済みの重みを取得)
        from pytorch_fid.inception import InceptionV3
        self.inception = InceptionV3([INCEPTION_BLOCKS[dims]])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        [0, 1] の画像から特徴を取り出す関数

        Parameters
        ----------
        images: torch.Tensor
            (N, 3, H, W) の [0, 1] の画像

        Returns
        ----------
        features: torch.Tensor
            (N, dims) の特徴
        """
        # 選んだブロックの出力を 2 次元に
        features = self.inception(images)[0].flatten(1)

        return features


class FeatureStatistics:
    def __init__(self):
        # 数, 和, 外積の和 (float64 で足し合わせる)
        self.count = 0
        self.total: Optional[torch.Tensor] = None
        self.outer: Optional[torch.Tensor] = None

    def add(self, features: torch.Tensor) -> None:
        """
        特徴を足し合わせる関数

        Parameters
        ----------
        features: torch.Tensor
            (N, D) の特徴

        Returns
        ----------
        None
        """
        # 次元の確認と, 和と外積の和の更新
        features = features.detach().to("cpu", torch.float64)
        if self.total is None:
            dims = features.shape[1]
            self.total = torch.zeros(dims, dtype=torch.float64)
            self.outer = torch.zeros(dims, dims, dtype=torch.float64)
        elif features.shape[1] != self.total.shape[0]:
            raise ValueError(
                f"特徴の次元が前と違います ({features.shape[1]} と "
                f"{self.total.shape[0]})"
            )
        self.count += features.shape[0]
        self.total += features.sum(dim=0)
        self.outer += features.T @ features

    def mean_cov(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        特徴の平均と共分散 (不偏) を求める関数

        Parameters
        ----------
        None

        Returns
        ----------
        mean: torch.Tensor
            (D,) の平均
        cov: torch.Tensor
            (D, D) の共分散
        """
        # 和と外積の和からの平均と不偏の共分散
        mean = self.total / self.count
        cov = (self.outer - self.count * torch.outer(mean, mean)) / (
            self.count - 1
        )

        return mean, cov


def symmetric_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    """
    対称で半正定値の行列の平方根を求める関数

    Parameters
    ----------
    matrix: torch.Tensor
        (D, D) の対称な行列

    Returns
    ----------
    root: torch.Tensor
        (D, D) の対称な平方根
    """
    # 固有値分解, 丸めの誤差の負の固有値は 0 に
    values, vectors = torch.linalg.eigh((matrix + matrix.T) / 2)
    root = (vectors * values.clamp(min=0).sqrt()) @ vectors.T

    return root


def frechet_distance(
    mean1: torch.Tensor, cov1: torch.Tensor, mean2: torch.Tensor,
    cov2: torch.Tensor
) -> float:
    """
    2 つの正規分布の間のフレシェ距離 (2 乗) を求める関数

    |mean1 - mean2|^2 + tr(cov1) + tr(cov2) - 2 tr((cov1 cov2)^(1/2)).
    行列の平方根は, 対称な cov1^(1/2) cov2 cov1^(1/2) の固有値から求める

    Parameters
    ----------
    mean1: torch.Tensor
        (D,) の 1 つ目の平均
    cov1: torch.Tensor
        (D, D) の 1 つ目の共分散
    mean2: torch.Tensor
        (D,) の 2 つ目の平均
    cov2: torch.Tensor
        (D, D) の 2 つ目の共分散

    Returns
    ----------
    distance: float
        フレシェ距離 (0 以上)
    """
    # 形の確認
    mean1, mean2 = mean1.double().cpu(), mean2.double().cpu()
    cov1, cov2 = cov1.double().cpu(), cov2.double().cpu()
    dims = mean1.shape[0] if mean1.dim() == 1 else -1
    shapes_ok = mean2.shape == (dims,) and cov1.shape == (dims, dims) and (
        cov2.shape == (dims, dims)
    )
    if not shapes_ok:
        raise ValueError(
            "平均は (D,), 共分散は (D, D) の同じ D の形にしてください"
        )

    # 平均の差と, 共分散の積の平方根のトレース
    root1 = symmetric_sqrt(cov1)
    middle = root1 @ cov2 @ root1
    values = torch.linalg.eigvalsh((middle + middle.T) / 2)
    trace_root = values.clamp(min=0).sqrt().sum()
    distance = (mean1 - mean2).pow(2).sum() + cov1.trace() + cov2.trace() - (
        2 * trace_root
    )
    distance = max(float(distance), 0.0)

    return distance


class FrechetInceptionDistance:
    def __init__(
        self, feature_model: Optional[nn.Module] = None, dims: int = 2048,
        device: Union[str, torch.device, None] = None
    ):
        # 特徴のモデル, 渡さなければ pytorch-fid の Inception (追加の依存 fid)
        if feature_model is None:
            require("fid")
            feature_model = InceptionFeatures(dims)
        self.device = torch.device(device) if device is not None else (
            next(feature_model.parameters(), torch.empty(0)).device
        )
        self.feature_model = feature_model.to(self.device).eval()
        for parameter in self.feature_model.parameters():
            parameter.requires_grad_(False)

        # 本物と生成の特徴の統計
        self.real = FeatureStatistics()
        self.fake = FeatureStatistics()

    def features(self, images: torch.Tensor) -> torch.Tensor:
        """
        画像を確かめて特徴を取り出す関数

        Parameters
        ----------
        images: torch.Tensor
            (N, 3, H, W) か (N, 1, H, W) の [0, 1] の画像

        Returns
        ----------
        features: torch.Tensor
            (N, D) の特徴
        """
        # 形と値の範囲の確認
        if not isinstance(images, torch.Tensor) or images.dim() != 4 or (
            images.shape[1] not in (1, 3)
        ):
            raise ValueError(
                "画像は (N, 3, H, W) か (N, 1, H, W) のテンソルにしてください"
            )
        images = images.detach().to(self.device, torch.float32)
        if images.numel() > 0 and not (
            torch.isfinite(images).all() and images.min() >= 0
            and images.max() <= 1
        ):
            raise ValueError(
                "画像の値は [0, 1] にしてください ([-1, 1] なら (x + 1) / 2)"
            )

        # 白黒は 3 つのチャンネルに写し, 勾配を計算せずに推論モードで特徴を抽出
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        self.feature_model.eval()
        with torch.no_grad():
            features = self.feature_model(images).flatten(1)

        return features

    def update_real(self, images: torch.Tensor) -> None:
        """
        本物の画像の特徴を足す関数

        Parameters
        ----------
        images: torch.Tensor
            (N, 3, H, W) か (N, 1, H, W) の [0, 1] の画像

        Returns
        ----------
        None
        """
        # 本物の統計の更新
        self.real.add(self.features(images))

    def update_fake(self, images: torch.Tensor) -> None:
        """
        生成した画像の特徴を足す関数

        Parameters
        ----------
        images: torch.Tensor
            (N, 3, H, W) か (N, 1, H, W) の [0, 1] の画像

        Returns
        ----------
        None
        """
        # 生成の統計の更新
        self.fake.add(self.features(images))

    @property
    def has_real(self) -> bool:
        """
        本物の画像の特徴を足したかを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        found: bool
            1 枚以上足した場合は True
        """
        # 本物の数の確認
        found = self.real.count > 0

        return found

    def reset_fake(self) -> None:
        """
        生成した画像の特徴だけを捨てる関数 (本物の特徴は使い回す)

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 生成の統計の作り直し
        self.fake = FeatureStatistics()

    def reset(self) -> None:
        """
        本物と生成の両方の特徴を捨てる関数

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 両方の統計の作り直し
        self.real = FeatureStatistics()
        self.fake = FeatureStatistics()

    def compute(self) -> float:
        """
        足した本物と生成の画像の特徴から FID を求める関数

        Parameters
        ----------
        None

        Returns
        ----------
        fid: float
            FID (小さいほど本物に近い)
        """
        # 共分散を求めるのに 2 枚以上ずつ要る
        for label, stats in [("本物", self.real), ("生成", self.fake)]:
            if stats.count < 2:
                raise ValueError(
                    f"{label}の画像が足りません ({stats.count} 枚. "
                    "2 枚以上が要る)"
                )

        # 平均と共分散のフレシェ距離
        mean1, cov1 = self.real.mean_cov()
        mean2, cov2 = self.fake.mean_cov()
        fid = frechet_distance(mean1, cov1, mean2, cov2)

        return fid
