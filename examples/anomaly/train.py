from pathlib import Path

import numpy as np
from omegaconf import DictConfig
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_found import (
    Trainer, average_precision, load_project_config, precision_at_k,
    save_evaluation, start_experiment
)


class TabularAutoencoder(nn.Module):
    def __init__(self, features: int, hidden: int, bottleneck: int):
        # 親クラスのコンストラクタを呼び出す
        super(TabularAutoencoder, self).__init__()

        # 列の数, 中間の幅, 縮めた幅を保存
        self.features = features
        self.hidden = hidden
        self.bottleneck = bottleneck

        # 縮める側と戻す側の全結合の層 (ML-ModelControl の同梱の例
        # tabular_autoencoder と同じ形)
        self.encoder = nn.Sequential(
            nn.Linear(self.features, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.bottleneck, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.features)
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        行を縮めてから戻し, 入力と同じ幅の再構成を返す関数

        Parameters
        ----------
        x: torch.Tensor
            (件数, 列の数) の行

        Returns
        ----------
        outputs: dict
            入力と同じ形の再構成 recon
        """
        # 縮めてから戻す
        outputs = {"recon": self.decoder(self.encoder(x))}

        return outputs


def make_rows(
    rng: np.random.Generator, mixing: np.ndarray, rows: int,
    anomalies: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    正常の行 (隠れた要因の線形の和と小さな雑音) と異常の行を作る関数

    Parameters
    ----------
    rng: np.random.Generator
        乱数
    mixing: np.ndarray
        (要因の数, 列の数) の要因から列への重み
    rows: int
        正常の行の数
    anomalies: int
        異常の行の数 (要因に従わない一様な値)

    Returns
    ----------
    x: np.ndarray
        (行の数, 列の数) の float32 の行 (正常の後に異常)
    y: np.ndarray
        行ごとの正解 (異常なら 1)
    """
    # 正常は要因の和, 異常は列ごとに独立な一様な値
    factors = rng.normal(size=(rows, mixing.shape[0]))
    normal = factors @ mixing + 0.05 * rng.normal(size=(rows, mixing.shape[1]))
    odd = rng.uniform(-3.0, 3.0, size=(anomalies, mixing.shape[1]))
    x = np.concatenate([normal, odd]).astype(np.float32)
    y = np.concatenate([np.zeros(rows), np.ones(anomalies)]).astype(np.int64)

    return x, y


def make_data(cfg: DictConfig) -> dict:
    """
    学習用 (正常だけ), 検証用とテスト用 (正常と異常) の表を作る関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    data: dict
        train, valid, test ごとの (行, 正解)
    """
    # 同じ要因の重みで 3 つの分割を作る
    rng = np.random.default_rng(cfg.experiment.seed)
    mixing = rng.normal(size=(cfg.data.factors, cfg.data.features))
    data = {
        "train": make_rows(rng, mixing, cfg.data.train_rows, 0),
        "valid": make_rows(rng, mixing, cfg.data.valid_rows,
                           cfg.data.anomalies),
        "test": make_rows(rng, mixing, cfg.data.test_rows,
                          cfg.data.anomalies),
    }

    return data


def row_scores(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """
    行ごとの異常の度合い (再構成と入力の二乗誤差の, 列の平均) を返す関数

    Parameters
    ----------
    model: nn.Module
        自己符号化器
    x: np.ndarray
        (行の数, 列の数) の行

    Returns
    ----------
    scores: np.ndarray
        (行の数,) の度合い (大きいほど異常)
    """
    # モデルのデバイスで, 勾配を計算せずに再構成
    device = next(model.parameters()).device
    rows = torch.from_numpy(x).to(device)
    with torch.no_grad():
        recon = model(rows)["recon"]
    scores = ((recon - rows) ** 2).mean(dim=1).cpu().numpy()

    return scores


class AnomalyTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment, valid: tuple, k: int
    ):
        # step を上書きするので step_fn は None
        # 親クラスのコンストラクタを呼び出す
        super(AnomalyTrainer, self).__init__(model, experiment, None)

        # 検証用の全ての行と正解, 上位 K 件の K を保存
        self.valid_x, self.valid_y = valid
        self.k = k
        self.criterion = nn.MSELoss()

    def step(self, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の, 正常の行だけの再構成の二乗誤差を求める関数

        Parameters
        ----------
        batch: tuple
            行と正解 (異常なら 1)
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            正常の行の再構成の二乗誤差 (loss) と, 平均に重みを付ける
            正常の行の数 (batch_size)
        """
        # 異常の行は損失に入れない (検証用に混ぜた正解の付いた異常)
        x, y = batch
        model = self.model if phase == "train" else self.eval_model()
        normal = y == 0
        recon = model(x)["recon"]
        if not normal.any():
            loss = recon.sum() * 0.0
        else:
            loss = self.criterion(recon[normal], x[normal])
        outputs = {"loss": loss, "batch_size": int(normal.sum())}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        検証のときに, 検証用の全ての行の度合いの上位 K 件の適合率と
        PR-AUC を数える関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        metrics: dict
            検証なら precision_at_k と average_precision (valid/<名前> と
            して記録). 他は空
        """
        # 検証用の正常と異常の全ての行
        if phase != "valid":
            return {}
        scores = row_scores(self.eval_model(), self.valid_x)
        metrics = {
            "precision_at_k": precision_at_k(scores, self.valid_y, self.k),
            "average_precision": average_precision(scores, self.valid_y),
        }

        return metrics


def make_loader(part: tuple, batch_size: int, shuffle: bool) -> DataLoader:
    """
    行と正解の DataLoader を作る関数

    Parameters
    ----------
    part: tuple
        行と正解
    batch_size: int
        まとまりの大きさ
    shuffle: bool
        並べ替えるか

    Returns
    ----------
    loader: DataLoader
        (行, 正解) のまとまり
    """
    # テンソルにして集約
    x, y = part
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


def save_results(
    experiment, model: nn.Module, test: tuple, cfg: DictConfig
) -> dict:
    """
    テスト用の行の度合いから, 指標の表, 度合いの分布の曲線, 上位の行の
    要約の表を書く関数

    Parameters
    ----------
    experiment: Experiment
        実験
    model: nn.Module
        評価に使うモデル
    test: tuple
        テスト用の行と正解
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    metrics: dict
        テスト用の上位 K 件の適合率と PR-AUC
    """
    # 指標の表
    x, y = test
    scores = row_scores(model, x)
    k = cfg.anomaly.k
    metrics = {
        "rows": int(len(y)), "anomalies": int(y.sum()), "k": int(k),
        "precision_at_k": precision_at_k(scores, y, k),
        "average_precision": average_precision(scores, y),
    }
    save_evaluation(
        experiment, "anomaly_test", metrics, "test",
        description="テスト用の行の度合いの上位 K 件の適合率と PR-AUC",
        better={"precision_at_k": "higher", "average_precision": "higher"}
    )

    # 正常と異常の度合いの分布 (区間の真ん中ごとの行の数)
    edges = np.linspace(scores.min(), scores.max(), cfg.anomaly.bins + 1)
    normal, _ = np.histogram(scores[y == 0], edges)
    odd, _ = np.histogram(scores[y == 1], edges)
    middles = (edges[:-1] + edges[1:]) / 2
    save_evaluation(
        experiment, "anomaly_scores_test", {
            "score": middles.tolist(), "normal": normal.tolist(),
            "anomaly": odd.tolist(),
        }, "test", description="度合いの区間ごとの正常と異常の行の数",
        kind="curve", x="score"
    )

    # 度合いの上位の行 (行の番号, 度合い, 正解)
    order = np.argsort(-scores, kind="stable")[:cfg.anomaly.top]
    save_evaluation(
        experiment, "anomaly_top_test", [
            {"row": int(row), "score": float(scores[row]),
             "anomaly": int(y[row])} for row in order
        ], "test", description="度合いの上位の行"
    )

    return metrics


# プロジェクトのディレクトリの決定
project_dir = Path(__file__).resolve().parent

# コマンドライン引数 (--experiment と「項目=値」) から config を読み込んで実験を開始
cfg, diff = load_project_config(str(project_dir))
experiment = start_experiment(cfg, diff, str(project_dir))

# 合成の表とモデル
data = make_data(cfg)
model = TabularAutoencoder(cfg.data.features, cfg.model.hidden,
                           cfg.model.bottleneck)

# 正常の行だけで学び, 検証用の上位 K 件の適合率でベストモデルを選ぶ
trainer = AnomalyTrainer(model, experiment, data["valid"], cfg.anomaly.k)
train_loader = make_loader(data["train"], cfg.data.batch_size, True)
valid_loader = make_loader(data["valid"], cfg.data.batch_size, False)
trainer.fit(train_loader, valid_loader)

# テスト用の評価 (ベストモデルの重み)
metrics = save_results(experiment, trainer.eval_model(), data["test"], cfg)
print(f"保存先: {experiment.run_dir}")
print(f"ベストモデルのエポック: {trainer.checkpoint.best_epoch}")
print(f"テスト用の上位 {metrics['k']} 件の適合率: "
      f"{metrics['precision_at_k']:.3f}, PR-AUC: "
      f"{metrics['average_precision']:.3f}")
