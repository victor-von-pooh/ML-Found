from pathlib import Path

from omegaconf import DictConfig
import torch
from torch import nn

from ml_found import Trainer, load_project_config, start_experiment


class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        # 親クラスのコンストラクタを呼び出す
        super(GraphConv, self).__init__()

        # 入力次元数と出力次元数を保存
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ノードごとの特徴量を変換する線形層を定義
        self.linear = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            ノードごとの特徴量
        adj: torch.Tensor
            正規化した隣接行列

        Returns
        ----------
        out: torch.Tensor
            隣接するノードの特徴量を集めた特徴量
        """
        # 特徴量を変換してから隣接するノードの特徴量を足し合わせ
        out = adj @ self.linear(x)

        return out


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        # 親クラスのコンストラクタを呼び出す
        super(GCN, self).__init__()

        # 入力次元数, 隠れ層の次元数, クラス数を保存
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # グラフ畳み込み層と活性化関数を定義
        self.conv1 = GraphConv(self.input_dim, self.hidden_dim)
        self.conv2 = GraphConv(self.hidden_dim, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            ノードごとの特徴量
        adj: torch.Tensor
            正規化した隣接行列

        Returns
        ----------
        logits: torch.Tensor
            ノードごとのクラスのスコア
        """
        # グラフ畳み込み層と活性化関数に順に適用
        h = self.relu(self.conv1(x, adj))
        logits = self.conv2(h, adj)

        return logits


def make_graph(cfg: DictConfig) -> dict:
    """
    2 つのコミュニティからなるグラフを作る関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    graph: dict
        特徴量, 正規化した隣接行列, ラベル, 学習用と検証用のマスク
    """
    # 乱数生成器を用意し, ノードを半分ずつ 2 つのクラスに分割
    generator = torch.Generator().manual_seed(cfg.experiment.seed)
    n = cfg.data.num_nodes
    y = (torch.arange(n) >= n // 2).long()

    # 同じクラスのノードほどつながりやすいように辺を引き, 無向グラフに変換
    same = y[:, None] == y[None, :]
    prob = torch.where(same, cfg.data.p_in, cfg.data.p_out)
    adj = (torch.rand(n, n, generator=generator) < prob).float()
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.T

    # 自分自身への辺を加え, 次数で正規化
    adj = adj + torch.eye(n)
    deg_inv_sqrt = adj.sum(dim=1).pow(-0.5)
    adj = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]

    # クラスごとに中心をずらし, ノイズを加えた特徴量の作成
    centers = torch.randn(2, cfg.data.num_features, generator=generator)
    noise = torch.randn(n, cfg.data.num_features, generator=generator)
    x = centers[y] + cfg.data.noise * noise

    # ノードを学習用と検証用に分割
    order = torch.randperm(n, generator=generator)
    valid_mask = torch.zeros(n, dtype=torch.bool)
    valid_mask[order[:int(n * cfg.data.valid_ratio)]] = True
    train_mask = ~valid_mask

    # 1 つの辞書に集約
    graph = {
        "x": x, "adj": adj, "y": y,
        "train_mask": train_mask, "valid_mask": valid_mask
    }

    return graph


def step(model: nn.Module, batch: dict, phase: str) -> dict:
    """
    phase に応じたノードだけで損失と正解率を求める関数

    Parameters
    ----------
    model: nn.Module
        学習するモデル
    batch: dict
        特徴量, 正規化した隣接行列, ラベル, 学習用と検証用のマスク
    phase: str
        train または valid

    Returns
    ----------
    outputs: dict
        交差エントロピー損失と正解率
    """
    # グラフ全体で予測し, phase に応じたノードだけの取り出し
    logits = model(batch["x"], batch["adj"])
    mask = batch[f"{phase}_mask"]
    logits = logits[mask]
    y = batch["y"][mask]

    # 損失の計算
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

# グラフとモデルを用意
graph = make_graph(cfg)
model = GCN(
    input_dim=cfg.data.num_features, hidden_dim=cfg.model.hidden_dim,
    num_classes=2
)

# 同じグラフを学習用と検証用の両方に渡し, step の中でマスクの使い分け
trainer = Trainer(model, experiment, step)
history = trainer.fit([graph], [graph])

# ベストモデルのエポックの結果を表示
best_epoch = trainer.checkpoint.best_epoch
best_acc = history.get("valid/acc")[best_epoch - 1]
print(f"保存先: {experiment.run_dir}")
print(f"ベストモデルのエポック: {best_epoch}")
print(f"ベストモデルの検証用データの正解率: {best_acc:.4f}")
