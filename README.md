# ML-Found
PyTorchで作られたモデルの学習の基盤を作る.

## 概要

PyTorch のモデルを config から自動で学習させる基盤モジュール.
今後のプロジェクトで git submodule として取り込んで使い回すことを想定している.

プロジェクト側で書くのは **データ**, **モデル**, **1 ステップの計算** の 3 つだけで, それ以外の実験管理と学習ループは ML-Found が受け持つ.
対象は学習までで, 推論や後処理はプロジェクト側で行う.

## 主な特徴

- **default と experiment の 2 段の config**: default を基準とし, experiment には差分だけを書く. experiment で default に無い項目を書くとエラーになるので, default を壊さずに実験できる
- **実験の記録**: 実行ごとに `outputs/<名前>/<日付>/<時刻>/` を作り, config 全体, default との差分, git の commit id と未コミットの変更, OS や PyTorch やパッケージの情報を残す
- **ミニバッチ学習とフルバッチ学習の両対応**: DataLoader でも, グラフ全体のような 1 つのデータでも, 同じ書き方で学習できる
- **torch の Optimizer と Scheduler をそのまま使える**: クラス名と引数を config に書くだけで, `torch.optim` にあるものは全て使える
- **ベストモデルと再開**: 監視する指標が最も良いモデルを保存して最後に戻す. Early Stopping, 勾配クリッピング, チェックポイントからの再開にも対応する
- **Lightning などに依存しない**: 素の PyTorch だけで書いている

## 導入

ML-Found を取り込みたいプロジェクトで, submodule として追加してからインストールする.

```bash
git submodule add https://github.com/victor-von-pooh/ML-Found.git ML-Found
pip install -e ./ML-Found
```

Python 3.10 以上が必要.
依存ライブラリは `torch`, `omegaconf`, `numpy`, `tqdm`, `matplotlib` で, インストール時に一緒に入る.

## 使い方

### プロジェクトの構成

```
NewProject/
├── ML-Found/                   # git submodule
├── configs/
│   ├── default/train.yaml      # 基準の config
│   └── experiment/train.yaml   # default からの差分, 差分が無くてもファイルは置く
├── outputs/                    # ML-Found が実験ごとに作る
└── train.py
```

`examples/minibatch/` か `examples/fullbatch/` をコピーすると, そのまま動く状態から始められる.

### config

default には ML-Found の項目とプロジェクト固有の項目を書く.
ML-Found の項目は, 書かなかったものが既定値になる.

```yaml
# configs/default/train.yaml
experiment:
  name: wine

optimizer:
  name: Adam
  params:
    lr: 0.001

trainer:
  epochs: 100

# プロジェクト固有の項目は自由に追加できる
model:
  hidden_dim: 32
```

experiment には default から変えたい項目だけを書く.

```yaml
# configs/experiment/train.yaml
optimizer:
  params:
    lr: 0.01
```

experiment で書けるのは default にある項目の値だけで, 項目の追加や構造の変更はエラーになる.
例外は `optimizer.params` と `scheduler.params` で, クラスを変えると必要な引数も変わるので自由に書ける.
クラス名を変えた場合は default の引数は捨てられる.

### 学習スクリプト

```python
from pathlib import Path

from torch import nn

from ml_found import Trainer, load_config, start_experiment


def step(model: nn.Module, batch: tuple, phase: str) -> dict:
    """
    1 バッチ分の損失と正解率を求める関数 (引数と戻り値の説明は省略)
    """
    # 予測して損失を求める
    x, y = batch
    logits = model(x)
    ce = nn.CrossEntropyLoss()
    loss = ce(logits, y)

    # 正解率を求めて損失とまとめる
    acc = (logits.argmax(dim=1) == y).float().mean()
    outputs = {"loss": loss, "acc": acc}

    return outputs


# config を読み込んで実験を開始する
project_dir = Path(__file__).resolve().parent
cfg, diff = load_config(
    str(project_dir / "configs/default/train.yaml"),
    str(project_dir / "configs/experiment/train.yaml")
)
experiment = start_experiment(cfg, diff, str(project_dir))

# model, train_loader, valid_loader はプロジェクト側で用意して学習する
trainer = Trainer(model, experiment, step)
history = trainer.fit(train_loader, valid_loader)
```

`step(model, batch, phase)` の決まりは以下の通り.

- `batch` は ML-Found がデバイスに移してから渡す
- `phase` は学習時に `"train"`, 検証時に `"valid"` になる
- 戻り値は辞書で, `loss` は必須. その他の値は `train/acc` や `valid/acc` のように指標として記録され, 学習曲線も描かれる
- 指標はバッチのサンプル数で重み付けして平均する. サンプル数は自動で推定するが, 戻り値に `batch_size` を入れると, そちらを使う

### フルバッチ学習

グラフ全体のように 1 度に全データを流す場合は, 要素が 1 つのリストを渡す.
学習用と検証用で同じデータを使い, `phase` でマスクを使い分ける.

```python
def step(model: nn.Module, batch: dict, phase: str) -> dict:
    mask = batch[f"{phase}_mask"]
    ...


history = trainer.fit([graph], [graph])
```

### 継承して細かく変える

`Trainer` を継承すると, 次のメソッドを上書きできる.

| メソッド | 呼ばれるタイミング | 用途 |
| --- | --- | --- |
| `step(batch, phase)` | バッチごと | `step_fn` を渡す代わりに書く |
| `on_epoch_start(phase)` | エポックの学習または検証の前 | 予測を集める入れ物を用意する |
| `on_epoch_end(phase)` | エポックの学習または検証の後 | AUC など, バッチの平均では求められない指標を辞書で返す |

継承した中では `self.epoch` (今のエポック) と `self.global_step` (パラメータの更新回数) を参照できる.
Optimizer や Scheduler を自分で作る場合は `Trainer(model, experiment, step, optimizer, scheduler)` のように渡す.

### 学習の再開

`trainer.resume_from` に前回の `last.pt` を指定すると, モデル, Optimizer, Scheduler, 記録, Early Stopping, 乱数の状態を戻して続きのエポックから学習する.

```yaml
# configs/experiment/train.yaml
trainer:
  resume_from: /path/to/NewProject/outputs/wine/2026-11-08/16-18-27/last.pt
```

相対パスで書いた場合は, 学習スクリプトを実行した場所から辿る.

### 学習後のモデル

`fit` の後の `trainer.model` はベストモデルの重みになっている.
後から使う場合は `final_model.pt` を読み込む.

```python
import torch

model.load_state_dict(torch.load("outputs/wine/.../final_model.pt"))
```

## config の項目

ML-Found が使う項目と既定値.

| 項目 | 既定値 | 説明 |
| --- | --- | --- |
| `experiment.name` | `experiment` | 実験の名前, 出力先のディレクトリ名になる |
| `experiment.output_dir` | `outputs` | 出力先, 相対パスの場合はプロジェクトのディレクトリから辿る |
| `experiment.seed` | `42` | 乱数シード |
| `experiment.deterministic` | `true` | 速度より再現性を優先して cuDNN を設定するかどうか |
| `experiment.device` | `auto` | デバイス, `auto` の場合は cuda, mps, cpu の順に選ぶ |
| `experiment.log_format` | (省略) | ログの書式 |
| `optimizer.name` | `Adam` | `torch.optim` のクラス名, `.` を含めると自作のクラスも読み込める |
| `optimizer.params` | `{lr: 0.001}` | Optimizer に渡す引数 |
| `scheduler.name` | `null` | `torch.optim.lr_scheduler` のクラス名, `null` の場合は使わない |
| `scheduler.params` | `{}` | Scheduler に渡す引数 |
| `scheduler.interval` | `epoch` | 学習率を更新する間隔, `epoch` か `step` |
| `trainer.epochs` | `100` | エポック数 |
| `trainer.monitor` | `null` | ベストモデルの判定に使う指標, `null` の場合は `valid/loss`, 検証しない場合は `train/loss` |
| `trainer.mode` | `min` | 指標は小さい方と大きい方のどちらが良いか |
| `trainer.grad_clip_norm` | `null` | 勾配のノルムの上限, `null` の場合はクリッピングしない |
| `trainer.resume_from` | `null` | 再開するチェックポイントのパス |
| `trainer.progress_bar` | `true` | 進捗バーを表示するかどうか |
| `trainer.early_stopping.use` | `false` | Early Stopping を使うかどうか |
| `trainer.early_stopping.patience` | `10` | 改善しないまま待つエポック数 |
| `trainer.early_stopping.min_delta` | `0.0` | これより小さい変化は改善とみなさない |
| `trainer.checkpoint.save_best` | `true` | `best.pt` を保存するかどうか |
| `trainer.checkpoint.save_last` | `true` | `last.pt` を保存するかどうか |

ReduceLROnPlateau は `trainer.monitor` の値で学習率を下げるので, `scheduler.interval` は `epoch` にする.
`mode` を省略した場合は `trainer.mode` と同じになる.

## 出力

実験ごとに `<output_dir>/<name>/<日付>/<時刻>/` が作られる.

| ファイル | 内容 |
| --- | --- |
| `config.yaml` | 学習に使った config 全体, default として読み込めば同じ条件で学習できる |
| `diff.yaml` | default から変えた項目 |
| `train.log` | 差分, 環境の情報, モデルの構造, エポックごとの指標, 結果 |
| `git_diff.patch` | 未コミットの変更がある場合のみ, commit id からの差分 |
| `history.csv` | エポックごとの指標 |
| `<指標>_curve.png` | 指標ごとの学習曲線 |
| `best.pt` | ベストモデルのチェックポイント |
| `last.pt` | 最後のエポックのチェックポイント, 再開に使う |
| `final_model.pt` | ベストモデルの重みだけ |

## ディレクトリ構成

```
ML-Found/
├── src/ml_found/
│   ├── config/             # config の型定義, 読み込み, 差分
│   ├── experiment/         # 出力先の作成, ロガー, シード, 環境の情報
│   ├── training/           # Trainer, Optimizer と Scheduler, Early Stopping, チェックポイント, 記録
│   └── utils/              # デバイス, バッチの転送, 学習曲線
├── examples/
│   ├── minibatch/          # wine データセットの分類, DataLoader で学習する例
│   └── fullbatch/          # グラフのノード分類, マスクで学習と検証を分ける例
├── tests/
├── CODING_STYLE.md         # コーディングスタイルの基準
└── pyproject.toml
```

## 開発

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,examples]"

# テスト
pytest

# 例の実行
python examples/minibatch/train.py
python examples/fullbatch/train.py
```

コードの書き方は [CODING_STYLE.md](CODING_STYLE.md) に従う.
