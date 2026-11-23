# ML-Found

「機械学習を全て GUI 上で行う」うちの「モデルの学習・評価」の段階のライブラリ. PyTorch のモデルを config から学習させ, 実験ごとの記録を残す. 画面のアプリは [ML-Dashboard](https://github.com/victor-von-pooh/ML-Dashboard).

## できること

プロジェクト側で書くのは **データ**, **モデル**, **1 ステップの計算** の 3 つだけで, それ以外の実験管理と学習ループは ML-Found が受け持つ. 対象は学習までで, 推論は ML-Inference が受け持つ.

- **default と experiment の 2 段の config**: default を基準とし, experiment には差分だけを書く. experiment で default に無い項目を書くとエラーになるので, default を壊さずに実験できる
- **実験の記録**: 実行ごとに `outputs/<名前>/<日付>/<時刻>/` を作り, config 全体, default との差分, git の commit id と未コミットの変更の差分, OS や PyTorch やパッケージの情報を残す
- **ミニバッチ学習とフルバッチ学習の両対応**: DataLoader でも, グラフ全体のような 1 つのデータでも, 同じ書き方で学習できる
- **torch の Optimizer と Scheduler をそのまま使える**: クラス名と引数を config に書くだけで, `torch.optim` にあるものを使える. closure が必要な LBFGS も自動で見分けて使える. Scheduler は学習の始めの warmup (`scheduler.warmup_steps`, `warmup_ratio`) と, 学習全体のステップ数に合わせた引数の補い (`auto`) も書ける. 自分で作って渡した Scheduler は `run.json` に残る. GAN や Actor-Critic のように, 部分ごとに別の損失と Optimizer で更新する複数の Optimizer も書ける
- **ベストモデルと再開**: 監視する指標が最も良いモデルを保存して最後に戻す. Early Stopping, 勾配クリッピング, チェックポイントからの再開にも対応する
- **重みの指数移動平均 (EMA)**: `trainer.ema.decay` に減衰率を書くだけで, EMA の重みで検証し, ベストモデルと `final_model.pt` も EMA の重みにする. EMA の状態はチェックポイントに入り, 再開しても通しの学習と一致する
- **大きな学習済みのモデルの微調整**: 勾配の蓄積 (`trainer.accumulate_grad_batches`), bf16 の混合精度 (`trainer.precision: bf16`), 始める前に重みの一部を前置きを付け替えて読む (`trainer.init_weights`), 凍結した重みを省いたチェックポイント (`trainer.checkpoint.trainable_only`) を config に書くだけで使える. 同じ設定で再開すれば通しの学習と一致する
- **Lightning などに依存しない**: 素の PyTorch だけで書いている
- **強化学習と方策の書き出し (任意)**: Gymnasium の形の環境とのやり取りを loader にして, 同じ `Trainer` で方策を学習し, 守り (行動の範囲, 止める条件, 外れたときの動き) と一緒に書き出せる. 評価で走らせた軌跡は, 1 歩が 1 行の表 (エピソードの番号, 歩数, 観測, 行動, 報酬, 終了と打ち切りの別, 守りの結果) として `trajectories/` に書き出せる (`save_trajectory`. 既定では書かない. 表の形は docs/FORMAT.md の「trajectories」). 再開では環境の乱数と状態も戻り, 通しの学習と一致する
- **何の実験かと評価, 見本**: 実験の `task.yaml` (`write_task`, `task_from_dataset`), 学習の後のデータ全体の評価 (`save_evaluation`, `load_evaluations`), 入力と予測と正解の見本 (`save_sample`, `load_samples`) を, 実験の記録に残して読める
- **追加の依存の案内**: 任意の機能 (強化学習, MuJoCo, ROS 2, CUDA, 生成の画像の FID, 他の段階のライブラリ ML-EDA, ML-ModelControl, ML-Inference) ごとに, 要るもの, 入っているかの確かめ方 (版の下限があれば版も), 入れ方の一覧を持ち, 足りなければ次の一手を案内して止まる

## どのアプリから使われるか

| 相手 | 使うもの |
| --- | --- |
| ML-Dashboard (学習・評価) | 実験のディレクトリ (config, 履歴, 進み具合, 評価, 見本) の読み取りと, config の確認と学習の起動. 追加の依存の案内 (`describe_requirements`), Optimizer と Scheduler の登録簿 (`describe_optimizers`, `describe_schedulers`) |
| ML-Quest (問題の設定) | プロジェクトの形. 問題の設定から書き出す骨組みはこの形で動く. 追加の依存の案内と, Optimizer と Scheduler の登録簿 |
| ML-Inference (推論) | 実験のディレクトリの学習した重みと config. 強化学習の方策の一式 `policy/` (`policy.json` と `policy.pt`, 守り) |
| ML-ModelControl, ML-EDA | モデルの定義は実験の出力先に写し (`Experiment.model_spec_path`), データセットは `task_from_dataset` で説明を読む |

## 導入

ML-Found を取り込みたいプロジェクト (git のリポジトリ) で, submodule として追加してから, プロジェクトの仮想環境にインストールする.

```bash
cd NewProject
git init                      # まだ git のリポジトリでない場合
python -m venv .venv
source .venv/bin/activate
git submodule add https://github.com/victor-von-pooh/ML-Found.git ML-Found
pip install -e ./ML-Found
```

- Python 3.10 以上が必要
- PyTorch は, 先に [PyTorch の手順](https://pytorch.org/get-started/locally/) で環境 (CPU, CUDA) に合わせて入れておくと, 大きな版を入れ直さずに済む (入れていない場合は `pip` が既定の版を入れる. Linux では CUDA 付きの大きな版)
- submodule にしない場合は `pip install "ml-found @ git+https://github.com/victor-von-pooh/ML-Found.git"` でも入る (PyPI には公開していない)
- 依存ライブラリは `torch`, `omegaconf`, `numpy`, `tqdm`, `matplotlib` で, インストール時に一緒に入る
- `-e` を付けてインストールすると, 実験ごとに ML-Found の commit id も記録される. 付けない場合はバージョン番号だけが記録される
- `examples/minibatch/` を動かす場合は scikit-learn も必要なので, `pip install -e "./ML-Found[examples]"` でインストールする. 例は `python ML-Found/examples/minibatch/train.py` (か `fullbatch`) で動き, 出力は例のディレクトリの `outputs/` にできる (ML-Found の `.gitignore` で除外している)
- `examples/cartpole/` と `examples/pendulum/` (強化学習) を動かす場合は Gymnasium も必要なので, `pip install -e "./ML-Found[rl]"` でインストールする (docs/DESIGN.md の「追加の依存」)

プロジェクトの `.gitignore` には `outputs/` を加えておく.
加えないと, 2 回目以降の実験で前回の出力が未コミットの変更として記録される.

開発する場合は, リポジトリを取ってきて入れ, `pytest` でテストする (rl を入れない場合, CartPole と Pendulum のテストは飛ばす).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,examples,rl]"
pytest
python examples/minibatch/train.py   # fullbatch, cartpole, pendulum も同じ
python examples/minibatch/train.py --experiment configs/experiment/examples/ema.yaml   # EMA の重みで検証する例
```

任意の機能 (強化学習, MuJoCo, ROS 2, CUDA, 生成の画像の FID, 他の段階のライブラリ) に要るもの (追加の依存) と入れ方は [docs/DESIGN.md](docs/DESIGN.md) の「追加の依存」.

## 使い方

### プロジェクトの構成

```
NewProject/
├── .gitignore                  # outputs/ を加える
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

モデルの定義 (ML-ModelControl の定義の YAML) を使う場合は, `model.spec` に場所 (プロジェクトからの相対パス) を書くと, プロジェクトの中の普通のファイルなら `start_experiment` が実験のディレクトリに `model_spec.yaml` として写し, 場所と中身の sha256 を `source.yaml` に残す. プロジェクトの外とリンクは写さず, 外の場所は絶対パスで記録する (docs/DESIGN.md の「モデルの定義の写し」). モデルは `model.spec` の場所でなく, `start_experiment` が返す `experiment.model_spec_path` (写した `model_spec.yaml`. 写さなかった場合は `None`) から組み立てるのを勧める. 写しから組み立てると, 記録の sha256 と学習に使った定義がそろう. 定義の中の相対の参照 (外部の重みなど) は元の定義の場所から辿るので, 組み立てる関数に起点として `experiment.model_spec_source.parent` (元の定義のディレクトリ) を渡す.

```python
# 例: ML-ModelControl の build_model で写しから組み立てる (写した場合.
# 写さなかった場合は model_spec_path と model_spec_source が None)
model = build_model(
    experiment.model_spec_path, base_dir=experiment.model_spec_source.parent
)
```

experiment には default から変えたい項目だけを書く.

```yaml
# configs/experiment/train.yaml
optimizer:
  params:
    lr: 0.01
```

### 学習スクリプト

```python
from pathlib import Path

from torch import nn

from ml_found import Trainer, load_project_config, start_experiment


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


# コマンドライン引数から config を読み込んで実験を開始する
project_dir = Path(__file__).resolve().parent
cfg, diff = load_project_config(str(project_dir))
experiment = start_experiment(cfg, diff, str(project_dir))

# model, train_loader, valid_loader はプロジェクト側で用意して学習する
# (データからモデルまで揃った形は examples/minibatch/train.py)
trainer = Trainer(model, experiment, step)
history = trainer.fit(train_loader, valid_loader)
```

`load_project_config(project_dir)` はコマンドライン引数を読み, `--experiment` で experiment の config を, 「項目=値」で値の上書きを受け取る (順番は自由). experiment の相対パスはプロジェクトのディレクトリから辿り, 無い場合は実行した場所から辿る. `--experiment` を省くと `configs/experiment/train.yaml` を使う. ノートブックの中では `sys.argv` を読まず, 引数が無いものとして扱う (引数を渡す場合は `load_project_config(project_dir, ["trainer.epochs=10"])`).

```bash
# configs/experiment/examples/my.yaml は自分で作る差分の config の例
python train.py --experiment configs/experiment/examples/my.yaml trainer.epochs=10 optimizer.params.lr=0.05
```

フルバッチ学習, 継承して細かく変える書き方, 複数の Optimizer, 検証の間隔, 再開, 大きな学習済みのモデルの微調整, 何の実験かと評価と見本の書き方, 強化学習と評価の軌跡の表, config の項目の一覧, 出力のファイルは [docs/DESIGN.md](docs/DESIGN.md) にある.

## 文書

- [docs/DESIGN.md](docs/DESIGN.md): 今の設計と決まり, 他のリポジトリとの約束 (プロジェクトと実験のディレクトリ)
- [docs/FORMAT.md](docs/FORMAT.md): 他のリポジトリと受け渡すプロジェクトと実験のディレクトリの形と版
- [TODO.md](TODO.md): まだやっていないこと (優先度の順)
- [CODING_STYLE.md](CODING_STYLE.md): コードの書き方
- [CLAUDE.md](CLAUDE.md): このリポジトリで作業するときの方針

## ライセンス

MIT
