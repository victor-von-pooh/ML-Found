import csv
from array import array
import math
from typing import Any, Optional, Sequence, Union

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


def to_tensor(values: array, dtype: torch.dtype) -> torch.Tensor:
    """
    数の配列を, 中身を写したテンソルにする関数

    Parameters
    ----------
    values: array
        数の配列 (d は float64, b は int8)
    dtype: torch.dtype
        テンソルの型

    Returns
    ----------
    tensor: torch.Tensor
        写したテンソル, 後から配列が増えても変わらない
    """
    # 空の配列はバッファから作れないので空のテンソル
    if len(values) == 0:
        tensor = torch.empty(0, dtype=dtype)
        return tensor
    tensor = torch.frombuffer(values, dtype=dtype).clone()

    return tensor


class History:
    def __init__(self):
        # エポックごとの記録を保存
        self.records: list[dict] = []

        # チェックポイント用の列ごとの数の配列 (記録が増えた分だけ足す)
        self.table_source: Optional[list[dict]] = None
        self.table_rows = 0
        self.table_columns: list[str] = []
        self.table_values: list[array] = []
        self.table_mask: list[array] = []
        # 列の値が全て整数か (読み込むときに整数に戻すため)
        self.table_integer: list[bool] = []

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
            # 記録が無い場合 (1 エポック目の途中で止めた場合) も見出しは書く
            writer = csv.DictWriter(f, fieldnames=self.keys() or ["epoch"])
            writer.writeheader()
            writer.writerows(self.records)

    def update_table(self) -> None:
        """
        チェックポイント用の列ごとの数の配列に, 増えた記録を足す関数

        記録の入れ物が置き換わった場合や減った場合は作り直す

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 記録が置き換わった場合と減った場合は作り直し
        if (
            self.table_source is not self.records
            or self.table_rows > len(self.records)
        ):
            self.table_source = self.records
            self.table_rows = 0
            self.table_columns = []
            self.table_values = []
            self.table_mask = []
            self.table_integer = []

        # 増えた記録を 1 行ずつ足し, 新しい列は前の行を記録なしで補う
        for record in self.records[self.table_rows:]:
            for key in record:
                if key not in self.table_columns:
                    self.table_columns.append(key)
                    self.table_values.append(
                        array("d", [math.nan] * self.table_rows)
                    )
                    self.table_mask.append(array("b", [0] * self.table_rows))
                    self.table_integer.append(True)
            for j, key in enumerate(self.table_columns):
                value = record.get(key)
                self.table_values[j].append(
                    math.nan if value is None else float(value)
                )
                self.table_mask[j].append(0 if value is None else 1)
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int)
                ):
                    self.table_integer[j] = False
            self.table_rows += 1

    def state_dict(self) -> dict:
        """
        チェックポイントに保存するための記録を取得する関数

        Parameters
        ----------
        None

        Returns
        ----------
        state: dict
            列の名前 (columns), 列ごとの値 (values, float64) と値があるか
            (mask), 列の値が全て整数か (integer)
        """
        # 記録が長くなっても保存が軽いように, 列ごとの数の配列にして受け渡し
        # (辞書のリストのままだと, 毎回の保存で全ての記録を pickle するため)
        # 記録の無い値は mask で区別し, 列の順は最初に現れた順
        self.update_table()
        state = {
            "columns": list(self.table_columns),
            "values": [to_tensor(column, torch.float64)
                       for column in self.table_values],
            "mask": [to_tensor(column, torch.int8)
                     for column in self.table_mask],
            "integer": list(self.table_integer)
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
        # 以前の形 (辞書のリスト) のチェックポイントはそのまま置き換え
        if "records" in state:
            self.records = [dict(record) for record in state["records"]]
            return

        # 列ごとの数の配列から, エポックごとの辞書に戻す (整数の列は整数)
        columns = list(state["columns"])
        integer = list(
            state.get("integer", [key == "epoch" for key in columns])
        )
        values = [column.tolist() for column in state["values"]]
        mask = [column.tolist() for column in state["mask"]]
        n_rows = len(values[0]) if values else 0
        records = []
        for i in range(n_rows):
            record: dict[str, Any] = {}
            for j, key in enumerate(columns):
                if mask[j][i]:
                    value = values[j][i]
                    record[key] = int(value) if integer[j] else value
            records.append(record)
        self.records = records


class LrSteps:
    def __init__(self, columns: Sequence[str] = ("lr",)):
        # 更新ごとの通しの番号, エポック, 学習率を, 長い学習でも軽くなるように
        # 数の配列で保存
        # 学習率の列は, 名前付きの Optimizer を使う場合は Optimizer ごと
        # (lr_<名前>)
        self.columns = list(columns)
        self.steps = array("q")
        self.epochs = array("q")
        self.lrs = {column: array("d") for column in self.columns}

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

    def append(
        self, step: int, epoch: int, lr: Union[float, Sequence[float]]
    ) -> None:
        """
        1 回の更新の学習率を加える関数

        Parameters
        ----------
        step: int
            通しの更新の番号
        epoch: int
            更新したエポック
        lr: Union[float, Sequence[float]]
            その更新に使った学習率, 列が複数の場合は列の順の値

        Returns
        ----------
        None
        """
        values = [lr] if isinstance(lr, (int, float)) else list(lr)
        if len(values) != len(self.columns):
            raise ValueError(
                f"学習率の数 ({len(values)}) が列の数 ({len(self.columns)}) と"
                "違います"
            )
        self.steps.append(step)
        self.epochs.append(epoch)
        for column, value in zip(self.columns, values):
            self.lrs[column].append(float(value))

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
        self.lrs = {
            column: array("d", (values[index] for index in keep))
            for column, values in self.lrs.items()
        }

    def records(self) -> list[dict]:
        """
        記録を 1 回の更新ごとの辞書の一覧にする関数

        Parameters
        ----------
        None

        Returns
        ----------
        records: list[dict]
            更新ごとの step, epoch と, 学習率の列
        """
        records = []
        for index, (step, epoch) in enumerate(
            zip(self.steps, self.epochs, strict=True)
        ):
            record: dict[str, Any] = {"step": step, "epoch": epoch}
            for column in self.columns:
                record[column] = self.lrs[column][index]
            records.append(record)

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
            writer.writerow(["step", "epoch"] + self.columns)
            writer.writerows(
                zip(
                    self.steps, self.epochs,
                    *(self.lrs[column] for column in self.columns),
                    strict=True
                )
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
            番号, エポック, 学習率の列ごとのテンソル (保存と読み込みを速く
            する形)
        """
        # 後から書き換わらないように複製してテンソルに変換
        state = {
            "step": torch.tensor(self.steps.tolist(), dtype=torch.int64),
            "epoch": torch.tensor(self.epochs.tolist(), dtype=torch.int64)
        }
        for column in self.columns:
            state[column] = torch.tensor(
                self.lrs[column].tolist(), dtype=torch.float64
            )

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
        # 学習率の列が今の設定と違う場合 (Optimizer の名前を変えたなど) はエラー
        missing = [column for column in self.columns if column not in state]
        if missing:
            raise ValueError(
                f"再開元の更新ごとの学習率に {missing} の記録がありません"
            )

        # 記録の置き換え
        self.steps = array("q", state["step"].tolist())
        self.epochs = array("q", state["epoch"].tolist())
        self.lrs = {
            column: array("d", state[column].tolist())
            for column in self.columns
        }
