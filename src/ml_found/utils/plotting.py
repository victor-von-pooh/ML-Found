from pathlib import Path

import matplotlib.pyplot as plt


def get_keys(records: list[dict]) -> list[str]:
    """
    記録されている項目の名前を取得する関数

    Parameters
    ----------
    records: list[dict]
        エポックごとの記録

    Returns
    -------
    keys: list[str]
        現れた順に並べた項目の名前
    """
    # 途中から増えた項目も含めて現れた順に並べる
    keys = []
    for record in records:
        for key in record:
            if key not in keys:
                keys.append(key)

    return keys


def get_metric_names(records: list[dict]) -> list[str]:
    """
    学習曲線を描く指標の名前を取得する関数

    Parameters
    ----------
    records: list[dict]
        エポックごとの記録

    Returns
    -------
    names: list[str]
        train/ と valid/ を除いた指標の名前
    """
    # train/loss と valid/loss は同じ loss として 1 つの図にまとめる
    names = []
    for key in get_keys(records):
        if key == "epoch":
            continue
        name = key.split("/", 1)[1] if "/" in key else key
        if name not in names:
            names.append(name)

    return names


def plot_metric(records: list[dict], name: str, output_path: str) -> None:
    """
    1 つの指標の学習曲線を描いて保存する関数

    Parameters
    ----------
    records: list[dict]
        エポックごとの記録
    name: str
        指標の名前
    output_path: str
        画像の保存先のパス

    Returns
    -------
    None
    """
    # 出力画像の設定
    fig, ax = plt.subplots(figsize=(18, 12), tight_layout=True)
    ax.set_title(f"{name} over Epochs", size=15, color="red")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel(name)

    # train と valid, または接頭辞の無い指標を, 記録があるエポックだけ描く
    keys = get_keys(records)
    for key in (f"train/{name}", f"valid/{name}", name):
        if key not in keys:
            continue
        points = [
            (record["epoch"], record[key])
            for record in records
            if record.get(key) is not None
        ]
        ax.plot(
            [epoch for epoch, _ in points], [value for _, value in points],
            label=key
        )

    # 凡例を表示して保存する
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    fig.savefig(output_path)
    plt.close(fig)


def plot_history(records: list[dict], output_dir: str) -> list[Path]:
    """
    全ての指標の学習曲線を描いて保存する関数

    Parameters
    ----------
    records: list[dict]
        エポックごとの記録, History.records を渡す
    output_dir: str
        画像の保存先のディレクトリ

    Returns
    -------
    paths: list[Path]
        保存した画像のパス
    """
    # 指標ごとに <指標名>_curve.png として保存する
    paths = []
    for name in get_metric_names(records):
        path = Path(output_dir) / f"{name}_curve.png"
        plot_metric(records, name, str(path))
        paths.append(path)

    return paths
