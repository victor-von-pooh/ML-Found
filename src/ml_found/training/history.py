import csv
from typing import Any

import torch


def to_float(value: Any) -> float:
    """
    指標の値を float にする関数

    Parameters
    ----------
    value: Any
        指標の値, テンソルまたは数値

    Returns
    -------
    number: float
        float にした値
    """
    # テンソルの場合は 1 つの値であることを確かめ, 計算グラフから切り離して取り出す
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
        # 指標ごとの重み付きの合計と重みの合計を保存する
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
        -------
        None
        """
        # 平均値にサンプル数を掛けて足し込む
        for key, value in metrics.items():
            self.sums[key] = self.sums.get(key, 0.0) + to_float(value) * n
            self.counts[key] = self.counts.get(key, 0.0) + n

    def compute(self) -> dict[str, float]:
        """
        加えた指標の平均を求める関数

        Returns
        -------
        means: dict[str, float]
            指標ごとの平均
        """
        # 重み付きの合計を重みの合計で割る
        means = {
            key: self.sums[key] / self.counts[key] for key in self.sums
        }

        return means

    def reset(self) -> None:
        """
        加えた指標を空にする関数

        Returns
        -------
        None
        """
        # 合計を空にする
        self.sums = {}
        self.counts = {}


class History:
    def __init__(self):
        # エポックごとの記録を保存する
        self.records: list[dict] = []

    def append(self, record: dict) -> None:
        """
        1 エポック分の記録を加える関数

        Parameters
        ----------
        record: dict
            epoch と指標の名前と値

        Returns
        -------
        None
        """
        # 記録を末尾に加える
        self.records.append(dict(record))

    def keys(self) -> list[str]:
        """
        記録されている項目の名前を取得する関数

        Returns
        -------
        names: list[str]
            現れた順に並べた項目の名前
        """
        # 途中から増えた項目も含めて現れた順に並べる
        names = []
        for record in self.records:
            for key in record:
                if key not in names:
                    names.append(key)

        return names

    def get(self, key: str) -> list:
        """
        1 つの項目の値をエポック順に取得する関数

        Parameters
        ----------
        key: str
            項目の名前

        Returns
        -------
        values: list
            エポック順の値, 記録されていないエポックは None
        """
        # エポックごとに値を取り出す
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
        -------
        None
        """
        # 全項目を列にして 1 エポック 1 行で書き出す
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.keys())
            writer.writeheader()
            writer.writerows(self.records)

    def state_dict(self) -> dict:
        """
        チェックポイントに保存するための記録を取得する関数

        Returns
        -------
        state: dict
            エポックごとの記録
        """
        # 記録をそのまま渡す
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
        -------
        None
        """
        # 記録を置き換える
        self.records = [dict(record) for record in state["records"]]
