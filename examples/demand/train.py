from decimal import Decimal
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_found import (
    PinballLoss, Trainer, load_project_config, order_cost, pinball_loss,
    save_evaluation, start_experiment
)


class DemandQuantiles(nn.Module):
    def __init__(self, features: int, hidden: int, levels: int):
        # 親クラスのコンストラクタを呼び出す
        super(DemandQuantiles, self).__init__()

        # 列の数, 中間の幅, 水準の数を保存
        self.features = features
        self.hidden = hidden
        self.levels = levels

        # 全結合の層 (ML-ModelControl の同梱の例 tabular_demand と同じ形)
        self.model = nn.Sequential(
            nn.Linear(self.features, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.levels)
        )
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> dict:
        """
        行から, 水準の小さい順に大きくなる分位点を返す関数

        Parameters
        ----------
        x: torch.Tensor
            (件数, 列の数) の行

        Returns
        ----------
        outputs: dict
            (件数, 水準の数) の分位点 quantiles
        """
        # 正の差を水準の次元で前から足す (分位点が交わらないように)
        quantiles = torch.cumsum(self.softplus(self.model(x)), dim=1)
        outputs = {"quantiles": quantiles}

        return outputs


def make_sales(cfg: DictConfig) -> np.ndarray:
    """
    商品ごとの日ごとの売れた数 (曜日の波と商品の大きさ) を作る関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    sales: np.ndarray
        (商品の数, 日の数) の売れた数
    """
    # 商品ごとの平均と曜日の波のポアソンの数
    rng = np.random.default_rng(cfg.experiment.seed)
    base = rng.uniform(2.0, 12.0, size=(cfg.data.products, 1))
    week = 1.0 + 0.4 * np.sin(2 * np.pi * np.arange(cfg.data.days) / 7)
    sales = rng.poisson(base * week[None, :]).astype(np.float64)

    return sales


def make_rows(cfg: DictConfig, sales: np.ndarray) -> dict:
    """
    発注を決める日ごとの行 (過去の売れた数の平均と曜日) と, 先の期間の
    売れた数の合計と, 前の期間の合計を分割ごとに作る関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config
    sales: np.ndarray
        (商品の数, 日の数) の売れた数

    Returns
    ----------
    rows: dict
        train, valid, test ごとの x (行), y (先の期間の合計), previous
        (前の期間の合計) の辞書
    """
    # 発注を決める日 t の行 (t より前だけを使う) と t からの先の期間の合計
    horizon, history = cfg.data.horizon, cfg.data.history
    parts: dict = {"train": [], "valid": [], "test": []}
    for day in range(history, cfg.data.days - horizon + 1):
        past = sales[:, day - history:day]
        weekday = np.zeros((sales.shape[0], 7))
        weekday[:, day % 7] = 1.0
        x = np.concatenate([
            past[:, -7:].mean(axis=1, keepdims=True),
            past.mean(axis=1, keepdims=True), weekday
        ], axis=1)
        y = sales[:, day:day + horizon].sum(axis=1)
        previous = sales[:, day - horizon:day].sum(axis=1)
        # 先の期間が分割の境目をまたぐ行は除く (後の分割の過去の売れた数と
        # 前の分割の先の期間が重ならないように)
        end = day + horizon
        if end <= cfg.data.train_until:
            split = "train"
        elif cfg.data.train_until <= day and end <= cfg.data.valid_until:
            split = "valid"
        elif cfg.data.valid_until <= day:
            split = "test"
        else:
            continue
        parts[split].append((x, y, previous))
    rows = {
        name: {
            "x": np.concatenate([p[0] for p in items]).astype(np.float32),
            "y": np.concatenate([p[1] for p in items]).astype(np.float32),
            "previous": np.concatenate([p[2] for p in items]),
        } for name, items in parts.items()
    }

    return rows


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """
    行の分位点を予測する関数

    Parameters
    ----------
    model: nn.Module
        需要のモデル
    x: np.ndarray
        (行の数, 列の数) の行

    Returns
    ----------
    quantiles: np.ndarray
        (行の数, 水準の数) の分位点
    """
    # モデルのデバイスで, 勾配を計算せずに予測
    device = next(model.parameters()).device
    with torch.no_grad():
        found = model(torch.from_numpy(x).to(device))["quantiles"]
    quantiles = found.cpu().numpy().astype(np.float64)

    return quantiles


def level_quantile(
    levels: list, quantiles: np.ndarray, tau: float
) -> np.ndarray:
    """
    水準 τ の分位点を, 学んだ水準の間は直線で補って求める関数

    Parameters
    ----------
    levels: list
        学んだ水準 (小さい順)
    quantiles: np.ndarray
        (行の数, 水準の数) の分位点
    tau: float
        水準 τ

    Returns
    ----------
    values: np.ndarray
        (行の数,) の分位点 (範囲の外は端の水準)
    """
    # 行ごとに水準の軸で直線の補間
    values = np.array([np.interp(tau, levels, row) for row in quantiles])

    return values


class DemandTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment, valid: dict, levels: list
    ):
        # step を上書きするので step_fn は None
        # 親クラスのコンストラクタを呼び出す
        super(DemandTrainer, self).__init__(model, experiment, None)

        # 検証用の行と先の期間の合計, 水準と分位点の損失を保存
        self.valid = valid
        self.levels = levels
        self.criterion = PinballLoss(levels)

    def step(self, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の分位点の損失を求める関数

        Parameters
        ----------
        batch: tuple
            行と先の期間の合計
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            分位点の損失 (loss)
        """
        # 学習は学習しているモデル, 検証は評価のモデル
        x, y = batch
        model = self.model if phase == "train" else self.eval_model()
        outputs = {"loss": self.criterion(y, model(x)["quantiles"])}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        検証のときに, 検証用の全ての行の分位点の損失の平均を数える関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        metrics: dict
            検証なら pinball_loss (valid/pinball_loss として記録). 他は空
        """
        # 検証用の全ての行
        if phase != "valid":
            return {}
        quantiles = predict(self.eval_model(), self.valid["x"])
        found = pinball_loss(self.valid["y"], quantiles, self.levels)
        metrics = {"pinball_loss": found["mean"]}

        return metrics


def make_loader(part: dict, batch_size: int, shuffle: bool) -> DataLoader:
    """
    行と先の期間の合計の DataLoader を作る関数

    Parameters
    ----------
    part: dict
        x と y
    batch_size: int
        まとまりの大きさ
    shuffle: bool
        並べ替えるか

    Returns
    ----------
    loader: DataLoader
        (行, 先の期間の合計) のまとまり
    """
    # テンソルにして集約
    dataset = TensorDataset(torch.from_numpy(part["x"]),
                            torch.from_numpy(part["y"]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


def save_results(
    experiment, model: nn.Module, test: dict, cfg: DictConfig
) -> dict:
    """
    テスト用の行の分位点から発注の数を決め, 指標の表, 水準ごとの表,
    行ごとの表を書く関数

    Parameters
    ----------
    experiment: Experiment
        実験
    model: nn.Module
        評価に使うモデル
    test: dict
        テスト用の x, y, previous
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    metrics: dict
        テスト用の分位点の損失, 発注の数の損と比
    """
    # 水準 τ (10 進の表記どおり) の分位点を発注の数に (在庫は 0)
    levels = list(cfg.model.levels)
    quantiles = predict(model, test["x"])
    shortage = Decimal(str(cfg.order.shortage_cost))
    holding = Decimal(str(cfg.order.holding_cost))
    tau = float(shortage / (shortage + holding))
    order = np.round(level_quantile(levels, quantiles, tau))
    order = np.maximum(order, 0.0)

    # 分位点の損失と, 前の期間と同じ数の発注との見込みの損の比べ
    pinball = pinball_loss(test["y"], quantiles, levels)
    cost = order_cost(
        test["y"], order, shortage_cost=cfg.order.shortage_cost,
        holding_cost=cfg.order.holding_cost, baseline=test["previous"]
    )
    median = level_quantile(levels, quantiles, 0.5)
    metrics = {
        "rows": int(len(test["y"])), "level": tau,
        "mae": float(np.abs(median - test["y"]).mean()),
        "pinball_loss": pinball["mean"], "order_cost": cost["total"],
        "order_cost_baseline": cost["baseline_total"],
        "order_cost_ratio": cost["ratio"],
        "order_cost_saving": cost["saving"],
    }
    save_evaluation(
        experiment, "demand_test", metrics, "test",
        description="テスト用の需要の 0.5 の分位点の誤差, 分位点の損失と, "
                    "発注の数の見込みの損 (前の期間と同じ数の発注との比べ)",
        baseline="前の期間と同じ数の発注",
        better={"mae": "lower", "pinball_loss": "lower",
                "order_cost": "lower", "order_cost_ratio": "lower",
                "order_cost_saving": "higher"}
    )
    save_evaluation(experiment, "pinball_levels_test", pinball["levels"],
                    "test", description="水準ごとの分位点の損失",
                    better={"loss": "lower"})
    save_evaluation(experiment, "order_items_test", {
        "demand": test["y"].tolist(), "order": order.tolist(),
        **cost["items"]
    }, "test", description="行ごとの需要, 発注の数, 足りない数, 余った数, 損")

    return metrics


# プロジェクトのディレクトリの決定
project_dir = Path(__file__).resolve().parent

# コマンドライン引数 (--experiment と「項目=値」) から config を読み込んで実験を開始
cfg, diff = load_project_config(str(project_dir))
experiment = start_experiment(cfg, diff, str(project_dir))

# 合成の売れた数の記録と, 発注を決める日ごとの行
rows = make_rows(cfg, make_sales(cfg))
levels = list(cfg.model.levels)
model = DemandQuantiles(rows["train"]["x"].shape[1], cfg.model.hidden,
                        len(levels))

# 分位点の損失で学び, 検証用の分位点の損失でベストモデルを選ぶ
trainer = DemandTrainer(model, experiment, rows["valid"], levels)
trainer.fit(make_loader(rows["train"], cfg.data.batch_size, True),
            make_loader(rows["valid"], cfg.data.batch_size, False))

# テスト用の評価 (ベストモデルの重み)
metrics = save_results(experiment, trainer.eval_model(), rows["test"], cfg)
print(f"保存先: {experiment.run_dir}")
print(f"ベストモデルのエポック: {trainer.checkpoint.best_epoch}")
print(f"テスト用の分位点の損失: {metrics['pinball_loss']:.3f}, 発注の数の"
      f"見込みの損の比 (前の期間と同じ数の発注に対して): "
      f"{metrics['order_cost_ratio']:.3f}")
