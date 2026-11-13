import csv
from pathlib import Path

from torch import nn

# model_summary.csv の列
SUMMARY_COLUMNS = [
    "name", "kind", "module", "shape", "dtype", "numel", "requires_grad",
    "same_as"
]


def summarize_model(model: nn.Module) -> list[dict]:
    """
    モデルのパラメータとバッファを 1 つずつ表にする関数

    print(model) では中身が分からない層 (外部ライブラリの層など) も,
    nn.Module のパラメータやバッファとして持つものは名前と形が残る

    Parameters
    ----------
    model: nn.Module
        表にするモデル

    Returns
    ----------
    rows: list[dict]
        名前, 種類 (parameter か buffer), 持っている層のクラス名, 形,
        型, 要素数, 学習するかどうか, 共有している場合は最初の名前
    """
    # 名前から, それを持つ層のクラス名を引けるように用意
    modules = dict(model.named_modules())

    def owner(name: str) -> str:
        """
        パラメータやバッファを持つ層のクラス名を返す関数

        Parameters
        ----------
        name: str
            パラメータやバッファの名前 (例: encoder.0.weight)

        Returns
        ----------
        module: str
            持っている層のクラス名
        """
        # 最後の部分を除いた名前が層の名前
        prefix = name.rsplit(".", 1)[0] if "." in name else ""
        module = type(modules.get(prefix, model)).__name__

        return module

    # パラメータ, バッファの順に 1 行ずつ作成
    # 重みの共有 (同じ層を 2 回使う, 入力と出力の埋め込みを共有するなど) も
    # 分かるように, 同じテンソルの 2 つ目以降の名前も最初の名前付きで出力
    rows = []
    first_names: dict[int, str] = {}
    entries = [
        ("parameter", name, tensor)
        for name, tensor in model.named_parameters(remove_duplicate=False)
    ] + [
        ("buffer", name, tensor)
        for name, tensor in model.named_buffers(remove_duplicate=False)
    ]
    for kind, name, tensor in entries:
        same_as = first_names.setdefault(id(tensor), name)
        rows.append({
            "name": name, "kind": kind, "module": owner(name),
            "shape": "x".join(str(size) for size in tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "numel": tensor.numel(),
            "requires_grad": kind == "parameter" and tensor.requires_grad,
            "same_as": "" if same_as == name else same_as
        })

    return rows


def count_parameters(rows: list[dict]) -> dict[str, int]:
    """
    summarize_model の表から, 種類ごとの要素数の合計を求める関数

    Parameters
    ----------
    rows: list[dict]
        summarize_model の戻り値

    Returns
    ----------
    counts: dict[str, int]
        学習するパラメータ, 学習しないパラメータ, バッファの要素数
    """
    # 共有した 2 つ目以降の名前は数えず, 種類と学習するかどうかで合計
    counts = {"trainable": 0, "frozen": 0, "buffer": 0}
    for row in rows:
        if row["same_as"]:
            continue
        if row["kind"] == "buffer":
            counts["buffer"] += row["numel"]
        elif row["requires_grad"]:
            counts["trainable"] += row["numel"]
        else:
            counts["frozen"] += row["numel"]

    return counts


def save_model_summary(rows: list[dict], path: str) -> None:
    """
    summarize_model の表を CSV に保存する関数

    Parameters
    ----------
    rows: list[dict]
        summarize_model の戻り値
    path: str
        保存先のパス

    Returns
    ----------
    None
    """
    # 決まった列の順で書き出し
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
