import csv
from array import array
from typing import Any

import torch

from ml_found.utils.plotting import get_keys


def to_float(value: Any) -> float:
    """
    指標の値を float にする関数

    Parameters
    ----------
    value: Any
        指標の値, テンソルまたは数値

    Returns
    ----------
    number: float
        float にした値
    """
    # テンソルの場合は 1 つの値であることを確かめ, 計算グラフから切り離して取り出し
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"指標は 1 つの値にしてください: 形状 {tuple(value.shape)}"
            )
        number = value.detach().item()
    else:
        number = float(value)

    return number


class MetricTracker:
    def __init__(self):
        # 指標ごとの重み付きの合計と重みの合計を保存
        self.sums: dict[str, float] = {}
        self.counts: dict[str, float] = {}

    def update(self, metrics: dict, n: int = 1) -> None:
        """
        1 バッチ分の指標を加える関数

        Parameters
        ----------
        metrics: dict
            指標の名前と値
        n: int = 1
            バッチのサンプル数, バッチごとの平均値をサンプル数で重み付けする

        Returns
        ----------
        None
        """
        # 平均値にサンプル数を掛けて加算
        for key, value in metrics.items():
            self.sums[key] = self.sums.get(key, 0.0) + to_float(value) * n
            self.counts[key] = self.counts.get(key, 0.0) + n

    def compute(self) -> dict[str, float]:
        """
        加えた指標の平均を求める関数

        Parameters
        ----------
        None

        Returns
        ----------
        means: dict[str, float]
            指標ごとの平均
        """
        # 重み付きの合計を重みの合計で除算
        means = {
            key: self.sums[key] / self.counts[key] for key in self.sums
        }

        return means

    def reset(self) -> None:
        """
        加えた指標を空にする関数

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 合計を空に初期化
        self.sums = {}
        self.counts = {}


class History:
    def __init__(self):
        # エポックごとの記録を保存
        self.records: list[dict] = []

    def append(self, record: dict) -> None:
        """
        1 エポック分の記録を加える関数

        Parameters
        ----------
        record: dict
            epoch と指標の名前と値

        Returns
        ----------
        None
        """
        # 記録を末尾に追加
        self.records.append(dict(record))

    def keys(self) -> list[str]:
        """
        記録されている項目の名前を取得する関数

        Parameters
        ----------
        None

        Returns
        ----------
        names: list[str]
            現れた順に並べた項目の名前
        """
        # 途中から増えた項目も含めて現れた順に整列
        names = get_keys(self.records)

        return names

    def get(self, key: str) -> list:
        """
        1 つの項目の値をエポック順に取得する関数

        Parameters
        ----------
        key: str
            項目の名前

        Returns
        ----------
        values: list
            エポック順の値, 記録されていないエポックは None
        """
        # エポックごとに値の取り出し
        values = [record.get(key) for record in self.records]

        return values

    def save_csv(self, path: str) -> None:
        """
        記録を CSV ファイルに保存する関数

        Parameters
        ----------
        path: str
            保存先のパス

        Returns
        ----------
        None
        """
        # 全項目を列にして 1 エポック 1 行で書き出し
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.keys())
            writer.writeheader()
            writer.writerows(self.records)

    def state_dict(self) -> dict:
        """
        チェックポイントに保存するための記録を取得する関数

        Parameters
        ----------
        None

        Returns
        ----------
        state: dict
            エポックごとの記録
        """
        # 後から書き換わらないように記録を複製して受け渡し
        state = {"records": [dict(record) for record in self.records]}

        return state

    def load_state_dict(self, state: dict) -> None:
        """
        チェックポイントから記録を戻す関数

        Parameters
        ----------
        state: dict
            state_dict で取得した記録

        Returns
        ----------
        None
        """
        # 記録の置き換え
        self.records = [dict(record) for record in state["records"]]


class LrSteps:
    def __init__(self):
        # 更新ごとの通しの番号, エポック, 学習率を, 長い学習でも軽くなるように
        # 数の配列で保存
        self.steps = array("q")
        self.epochs = array("q")
        self.lrs = array("d")

    def __len__(self) -> int:
        """
        記録した更新の数を取得する関数

        Parameters
        ----------
        None

        Returns
        ----------
        count: int
            記録した更新の数
        """
        count = len(self.steps)

        return count

    def append(self, step: int, epoch: int, lr: float) -> None:
        """
        1 回の更新の学習率を加える関数

        Parameters
        ----------
        step: int
            通しの更新の番号
        epoch: int
            更新したエポック
        lr: float
            その更新に使った学習率

        Returns
        ----------
        None
        """
        self.steps.append(step)
        self.epochs.append(epoch)
        self.lrs.append(lr)

    def drop_from_epoch(self, epoch: int) -> None:
        """
        指定したエポック以降の記録を捨てる関数

        Parameters
        ----------
        epoch: int
            捨て始めるエポック (途中で止めたエポック)

        Returns
        ----------
        None
        """
        keep = [
            index for index, value in enumerate(self.epochs) if value < epoch
        ]
        self.steps = array("q", (self.steps[index] for index in keep))
        self.epochs = array("q", (self.epochs[index] for index in keep))
        self.lrs = array("d", (self.lrs[index] for index in keep))

    def records(self) -> list[dict]:
        """
        記録を 1 回の更新ごとの辞書の一覧にする関数

        Parameters
        ----------
        None

        Returns
        ----------
        records: list[dict]
            更新ごとの step, epoch, lr
        """
        records = [
            {"step": step, "epoch": epoch, "lr": lr}
            for step, epoch, lr in zip(
                self.steps, self.epochs, self.lrs, strict=True
            )
        ]

        return records

    def save_csv(self, path: str) -> None:
        """
        記録を CSV ファイルに保存する関数

        Parameters
        ----------
        path: str
            保存先のパス

        Returns
        ----------
        None
        """
        # 1 回の更新を 1 行にして書き出し
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "lr"])
            writer.writerows(
                zip(self.steps, self.epochs, self.lrs, strict=True)
            )

    def state_dict(self) -> dict:
        """
        チェックポイントに保存するための記録を取得する関数

        Parameters
        ----------
        None

        Returns
        ----------
        state: dict
            番号, エポック, 学習率のテンソル (保存と読み込みを速くする形)
        """
        # 後から書き換わらないように複製してテンソルに変換
        state = {
            "step": torch.tensor(self.steps.tolist(), dtype=torch.int64),
            "epoch": torch.tensor(self.epochs.tolist(), dtype=torch.int64),
            "lr": torch.tensor(self.lrs.tolist(), dtype=torch.float64)
        }

        return state

    def load_state_dict(self, state: dict) -> None:
        """
        チェックポイントから記録を戻す関数

        Parameters
        ----------
        state: dict
            state_dict で取得した記録

        Returns
        ----------
        None
        """
        # 記録の置き換え
        self.steps = array("q", state["step"].tolist())
        self.epochs = array("q", state["epoch"].tolist())
        self.lrs = array("d", state["lr"].tolist())
