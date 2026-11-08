# コーディングスタイル

ML-Found のコードを書くときの基準をまとめる.

## 判断の優先順位

書き方に迷った場合は, 次の順に優先する.

1. このファイルに明示的に書いたルール
2. 参考リポジトリ全体での多数派
    - `~/myprograms/MIRAGE-GS` の `src/` と `experiment_tools/`
    - `~/myprograms/sub` の各リポジトリ (Clairvoyant, GNN_Colmap, ImageProcessingLearning, Pooh)
3. PEP 8

2 で決めた場合は, 実際に数えた結果をこのファイルに追記する.

## 明示的なルール

- `torch.nn.functional` は使わない. 損失関数や活性化関数は `nn.MSELoss()` や `nn.ReLU()` のようにモジュール版を使う
- テストのコードにも同じ書き方を適用する

## docstring

日本語の NumPy 形式で書く. 1 行目は「〜する関数」で終える.

```python
def get_device(device: str = "auto") -> torch.device:
    """
    学習に使うデバイスを取得する関数

    Parameters
    ----------
    device: str = "auto"
        デバイス名, auto の場合は使えるものから自動で選ぶ

    Returns
    -------
    selected: torch.device
        学習に使うデバイス
    """
```

- `Parameters` の下線は 10 本, `Returns` の下線は 7 本
- 引数は `名前: 型` で書き, 既定値がある場合は `名前: 型 = 既定値` と書く
- 戻り値にも変数名を付ける
- 引数が無い場合も `Parameters` を書き, 中身を `None` にする
- 戻り値が無い場合も `Returns` を書き, 中身を `None` にする
- クラス自体と `__init__` には docstring を書かない

## コメント

- 処理の塊ごとに, 何をするかを日本語で 1 行書く
- 文末は「〜する」「〜を取り出す」のように動詞で終える
- 読点は半角の「, 」を使い, 英数字の前後には空白を入れる (例: 「2 の冪」)

## 関数とクラス

- 1 つの関数に 1 つの責務を持たせ, 取りまとめ役の関数が補助関数を順に呼ぶ
- 処理の塊の間と `return` の前には空行を入れる
- 結果は名前の付いた変数に入れてから `return` する
- `__init__` では `super(クラス名, self).__init__()` を呼び, 前に「# 親クラスのコンストラクタを呼び出す」と書く
- 引数は「〜を保存する」とコメントを付けて `self` に保存する

## 形式

- 型ヒントは全ての引数と戻り値に付ける
    - `tuple[int, int]` のような組み込みジェネリクスを使う
    - `Optional` と `Union` は `typing` から使い, `X | Y` とは書かない
- 1 行は 79 文字以内にする
- 長いシグネチャは引数を詰めて折り返し, 閉じ括弧を単独の行に置く. 末尾カンマは付けない
- 文字列はダブルクォートで囲む
- import は「標準ライブラリ / サードパーティ / ローカル」の 3 グループに分け, 絶対 import で書く
- PyTorch は `import torch` と `from torch import nn` で読み込む
- エラーは `raise ValueError("日本語のメッセージ")` のように日本語で出す
- ログのメッセージは日本語で書く. `train/loss` のような指標名は英語のままにする

## 多数派で決めた項目

2026-11-08 に参考リポジトリの関数とコメントを数えた結果.

| 項目 | 結果 | 採用 |
| --- | --- | --- |
| `Returns` の下線 | 7 本: 142, 10 本: 37 | 7 本 |
| `Parameters` の下線 | 10 本: 184 | 10 本 |
| コメントの文末 | 動詞: 662, 体言止め: 388 | 動詞 |
| 引数が無い関数の `Parameters` | 書く: 7, 省く: 0 | 書く |
| 戻り値が無い関数の `Returns` | 書く: 9, 省く: 5 | 書く |
| 損失関数 | `nn.XxxLoss()`: 5 (ノートブックを含む), `nn.functional`: 1 | `nn.XxxLoss()` |
| PyTorch の import | `from torch import nn` のみ, `from torch.nn import Xxx` は 0 | `from torch import nn` |
