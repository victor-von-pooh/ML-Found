from pathlib import Path

from omegaconf import DictConfig
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_found import Trainer, load_project_config, start_experiment


class WineClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        # 親クラスのコンストラクタを呼び出す
        super(WineClassifier, self).__init__()

        # 入力次元数, 隠れ層の次元数, クラス数を保存
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 線形層と活性化関数の積み重ね
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            説明変数

        Returns
        ----------
        logits: torch.Tensor
            クラスごとのスコア
        """
        # 線形層と活性化関数に順に適用
        logits = self.model(x)

        return logits


def make_loaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    """
    wine データセットから学習用と検証用の DataLoader を作る関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    train_loader: DataLoader
        学習用の DataLoader
    valid_loader: DataLoader
        検証用の DataLoader
    """
    # データを読み込み, クラスの比率を保って学習用と検証用に分割
    x, y = load_wine(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=cfg.data.valid_size, stratify=y,
        random_state=cfg.experiment.seed
    )

    # 学習用データで標準化の基準を決めて両方に適用
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_valid = ss.transform(x_valid)

    # テンソルにして DataLoader に集約
    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(x_valid).float(), torch.from_numpy(y_valid).long()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.data.batch_size, shuffle=False
    )

    return train_loader, valid_loader


def step(model: nn.Module, batch: tuple, phase: str) -> dict:
    """
    1 バッチ分の損失と正解率を求める関数

    Parameters
    ----------
    model: nn.Module
        学習するモデル
    batch: tuple
        説明変数と正解ラベル
    phase: str
        train または valid

    Returns
    ----------
    outputs: dict
        交差エントロピー損失と正解率
    """
    # 予測して損失の計算
    x, y = batch
    logits = model(x)
    ce = nn.CrossEntropyLoss()
    loss = ce(logits, y)

    # 最もスコアの高いクラスを予測として正解率を求め, 損失と集約
    acc = (logits.argmax(dim=1) == y).float().mean()
    outputs = {"loss": loss, "acc": acc}

    return outputs


# プロジェクトのディレクトリの決定
project_dir = Path(__file__).resolve().parent

# コマンドライン引数 (--experiment と「項目=値」) から config を読み込んで実験を開始
cfg, diff = load_project_config(str(project_dir))
experiment = start_experiment(cfg, diff, str(project_dir))

# データとモデルを用意
train_loader, valid_loader = make_loaders(cfg)
model = WineClassifier(
    input_dim=13, hidden_dim=cfg.model.hidden_dim, num_classes=3
)

# 学習
trainer = Trainer(model, experiment, step)
history = trainer.fit(train_loader, valid_loader)

# ベストモデルのエポックの結果を表示
best_epoch = trainer.checkpoint.best_epoch
best_acc = history.get("valid/acc")[best_epoch - 1]
print(f"保存先: {experiment.run_dir}")
print(f"ベストモデルのエポック: {best_epoch}")
print(f"ベストモデルの検証用データの正解率: {best_acc:.4f}")
