# ML-Found
PyTorch で作られたモデルの学習の基盤を作る.

## 概要

PyTorch のモデルを config から自動で学習させる基盤モジュール.
今後のプロジェクトで git submodule として取り込んで使い回すことを想定している.

プロジェクト側で書くのは **データ**, **モデル**, **1 ステップの計算** の 3 つだけで, それ以外の実験管理と学習ループは ML-Found が受け持つ.
対象は学習までで, 推論や後処理はプロジェクト側で行う.

## 主な特徴

- **default と experiment の 2 段の config**: default を基準とし, experiment には差分だけを書く. experiment で default に無い項目を書くとエラーになるので, default を壊さずに実験できる
- **実験の記録**: 実行ごとに `outputs/<名前>/<日付>/<時刻>/` を作り, config 全体, default との差分, git の commit id と未コミットの変更の差分, OS や PyTorch やパッケージの情報を残す
- **ミニバッチ学習とフルバッチ学習の両対応**: DataLoader でも, グラフ全体のような 1 つのデータでも, 同じ書き方で学習できる
- **torch の Optimizer と Scheduler をそのまま使える**: クラス名と引数を config に書くだけで, `torch.optim` にあるものを使える. closure が必要な LBFGS も自動で見分けて使える
- **ベストモデルと再開**: 監視する指標が最も良いモデルを保存して最後に戻す. Early Stopping, 勾配クリッピング, チェックポイントからの再開にも対応する
- **Lightning などに依存しない**: 素の PyTorch だけで書いている

## 導入

ML-Found を取り込みたいプロジェクトで, submodule として追加してからインストールする.

```bash
git submodule add https://github.com/victor-von-pooh/ML-Found.git ML-Found
pip install -e ./ML-Found
```

- Python 3.10 以上が必要
- 依存ライブラリは `torch`, `omegaconf`, `numpy`, `tqdm`, `matplotlib` で, インストール時に一緒に入る
- `-e` を付けてインストールすると, 実験ごとに ML-Found の commit id も記録される. 付けない場合はバージョン番号だけが記録される
- `examples/minibatch/` を動かす場合は scikit-learn も必要なので, `pip install -e "./ML-Found[examples]"` でインストールする

プロジェクトの `.gitignore` には `outputs/` を加えておく.
加えないと, 2 回目以降の実験で前回の出力が未コミットの変更として記録される.

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
`start_experiment` は出力先を作る前に, `optimizer.params` と `scheduler.params` の名前がクラスの引数にあるか (typo の確認), 必須の引数 (`StepLR` の `step_size` など) があるか, 値が登録簿の範囲 (下の `describe_optimizers()` などで返すもの. 例: `lr` は 0 以上, `ReduceLROnPlateau` の `factor` は 0 以上 1 未満) に入るかを確かめる. 範囲は意味のある値の範囲で, PyTorch の確認より狭いものもある (`momentum` は 1 以下など). 補間 (`${...}`) は解決した値で確かめ, 範囲を持つ引数に引用符で囲んだ数や真偽値を書いた場合もエラーにする. 値の範囲は torch のクラスだけを確かめ, 自作のクラスは確かめない. 任意の名前の引数 (`**kwargs`) や位置だけの引数を受け取る自作のクラスは確かめない. 自分で作った Optimizer や Scheduler を `Trainer` に渡す場合も, config の引数は確かめるので, 使わない引数は消しておく.

選べる Optimizer と Scheduler とその引数 (種類, 既定値, 必須か, 説明, 範囲. 範囲の端を含まない場合は `exclusive_minimum` と `exclusive_maximum` が true) は `describe_optimizers()` と `describe_schedulers()` で取得できる. ダッシュボードの入力フォームなどに使う.

```python
from ml_found.training import describe_optimizers

adam = next(spec for spec in describe_optimizers() if spec["name"] == "Adam")
print([(param["name"], param["default"]) for param in adam["params"]])
```

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
- 指標は 1 つの値にし, 名前には `/` を使わない (学習曲線の画像のファイル名に使うため)
- 指標はバッチのサンプル数で重み付けして平均する. サンプル数は自動で推定するが, 戻り値に `batch_size` を入れると, そちらを使う
- 重み付き平均してよいのは, 損失, MAE, 正解率のように「サンプルごとの値の平均」になる指標だけ. RMSE, AUC, F1 のようにエポック全体から求める指標は, 下の `on_epoch_end` で求める

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
| `on_epoch_end(phase)` | エポックの学習または検証の後 | AUC など, バッチの平均では求められない指標を辞書で返す. `train/` や `valid/` を付けて記録される |

継承した中では `self.epoch` (今のエポック) と `self.global_step` (パラメータの更新回数) を参照できる.
`step_fn` を渡す形では引数にエポックが渡らないので, エポックに応じて損失を変える場合 (KL annealing など) は継承して `self.epoch` を使う.

`__init__` を上書きする場合は, 親クラスに `step_fn` などを渡してから自分の属性を足す.

```python
class MyTrainer(Trainer):
    def __init__(self, model: nn.Module, experiment: Experiment, pos_weight: torch.Tensor):
        # step を上書きするので step_fn は None
        super(MyTrainer, self).__init__(model, experiment, None)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
```

ほかにも次のメソッドを上書きできる.

| メソッド | 用途 |
| --- | --- |
| `update_parameters(loss)` | 更新の手順を変える. `super().update_parameters(loss)` の後に EMA の重みを更新する, など |
| `extra_state_dict()` | チェックポイントに加える状態を辞書で返す (2 つ目の Optimizer, EMA の重みなど). `load_extra_state_dict(state)` も合わせて上書きすると, 再開時に戻る |

`on_epoch_end` は勾配の計算を止める範囲の外で呼ばれる (検証の後でも勾配の計算が有効). 中で推論する場合は `torch.no_grad()` を書く.
指標を返さない場合 (画像を保存するだけ, など) は空の辞書を返す. `step` の指標と同じ名前は使えない (以前は `on_epoch_end` の値で黙って上書きしていたが, エラーにした).
Optimizer や Scheduler を自分で作る場合は `Trainer(model, experiment, step, optimizer, scheduler)` のように渡す.
渡した Scheduler にも, config で作った場合と同じく `scheduler.interval` と ReduceLROnPlateau の確認が行われる.
自分で作った Scheduler は `config.yaml` に残らない (`train.log` にはクラス名と状態が残る).

`OneCycleLR` のように, データの量 (バッチ数) で決まる引数が要る Scheduler を config で使う場合は, `load_config` の後, `start_experiment` の前に config に書き込むと `config.yaml` にも残る.

```python
cfg, diff = load_config(default_path, experiment_path)
train_loader = make_loader(cfg)
cfg.scheduler.params.steps_per_epoch = len(train_loader)
experiment = start_experiment(cfg, diff, str(project_dir))
```

### 更新の手順

- 勾配法の Optimizer は `zero_grad`, `loss.backward()`, 勾配のクリッピング, `optimizer.step()` の順に更新する
- LBFGS のように `step` に closure が必要な Optimizer は自動で見分け (`step` の `closure` が既定値の無い引数かどうかで判定), 同じバッチで損失と勾配を計算し直す closure を渡す. closure は `step` を何度も呼ぶので, `step` で予測を貯めるなどの処理は, closure の中では `self.in_closure` が `True` になることを使って飛ばす. `closure` に既定値があるのに実際には必要な Optimizer は見分けられない
- GAN のように複数の Optimizer を自分で使う場合は, `trainer.manual_optimization: true` にして `step` の中で更新する. ML-Found は逆伝播, 勾配のクリッピング, 更新を行わず (`grad_clip_norm` と `log_grad_norm` と同時に使うとエラー), 更新回数と Scheduler (`interval` に応じて step ごとかエポックごと) だけを進める. `step` は勾配の流れない損失 (`loss.detach()`) を返してよい

### 検証の設定

- PINN のように検証でも入力に対する微分を使う場合は, `trainer.valid_grad: true` にする
- 拡散モデルのように損失に乱数を含む場合は, `trainer.valid_seed` を決めると毎エポック同じ乱数で検証し, ベストモデルの判定が乱数で揺れない. 検証の後は学習の乱数の状態に戻すので, 学習には影響しない. 固定するのは Python, NumPy, torch の全体の乱数なので, 自分の `generator` を渡した DataLoader や, ワーカーを使い回す DataLoader のワーカーの乱数には届かない

### 学習の再開

`trainer.resume_from` に前回の `last.pt` を指定すると, モデル, Optimizer, Scheduler, 記録, Early Stopping, 乱数の状態を戻して続きのエポックから学習する.

```yaml
# configs/experiment/train.yaml
trainer:
  resume_from: outputs/wine/2026-11-08/16-18-27/last.pt
```

- 相対パスは, `experiment.output_dir` と同じくプロジェクトのディレクトリから辿る
- `last.pt` にはベストモデルの重みも含まれるので, `last.pt` だけで再開できる
- `trainer.monitor` や `trainer.mode` を再開元から変えるとエラーになる
- Early Stopping で止めた学習を再開しても学習しない
- Scheduler の状態は引数 (`T_max` など) ごと再開元から戻る. config で引数を変えて再開すると効かず, 警告が出る. 例えば `T_max: ${trainer.epochs}` は再開時に `epochs` を増やしても変わらない. 警告は config と同じ名前で状態に入っている引数だけを比べるので, `OneCycleLR` の `epochs` や `max_lr` のように名前を変えて持つ引数の変更は警告されない. Optimizer の引数 (`lr` など) も再開元の値が使われる
- `last.pt` の保存の間隔を空けていて, 再開元の `best.pt` が `last.pt` より後のエポックのベストモデルの場合は, 再開で捨てる履歴のものなので新しい出力先に写さない (`train.log` に警告が出る)
- `persistent_workers=True` の DataLoader では, ワーカーの乱数の種を最初の 1 回だけ引くので, 再開した学習は通しで学習した場合と一致しない (警告が出る). 一致させたい場合は `num_workers: 0` にする
- 存在しない `resume_from` は, 出力先を作る前にエラーになる

### 学習後のモデル

`fit` の後の `trainer.model` はベストモデルの重みになっている (`trainer.restore_best: true` の場合).
後から使う場合は `final_model.pt` を読み込む. `final_model.pt` は cpu のテンソルで保存するので, 学習したデバイスに関わらず読み込める.

GAN のように損失が良し悪しの指標にならない場合や, 学習の途中で損失の定義が変わる場合 (β-VAE など) は, `trainer.restore_best: false` にすると最後のエポックの重みが残る.

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
| `trainer.monitor_tiebreak` | `null` | 監視する指標が最良値と同じ値 (相対誤差 1e-6 以内) か, 最良値より悪くないが `min_delta` 未満の改善の場合に比べる指標 (例: `valid/loss`), `null` の場合は同じ値を改善とみなさない |
| `trainer.monitor_tiebreak_mode` | `min` | 同じ値の場合に比べる指標は小さい方と大きい方のどちらが良いか |
| `trainer.grad_clip_norm` | `null` | 勾配のノルムの上限, `null` の場合はクリッピングしない. Optimizer が更新するパラメータだけが対象 |
| `trainer.log_grad_norm` | `false` | クリップ前の勾配のノルムを `train/grad_norm` として記録するかどうか |
| `trainer.restore_best` | `true` | 学習後にベストモデルの重みに戻すかどうか, `false` の場合は最後のエポックの重み |
| `trainer.valid_grad` | `false` | 検証時にも勾配を計算するかどうか |
| `trainer.valid_seed` | `null` | 検証の前に乱数を固定するシード, 検証の後は学習の乱数の状態に戻す |
| `trainer.manual_optimization` | `false` | ML-Found が逆伝播と更新を行わず, `step` の中で更新するかどうか |
| `trainer.resume_from` | `null` | 再開するチェックポイントのパス, 相対パスの場合はプロジェクトのディレクトリから辿る |
| `trainer.progress_bar` | `true` | 進捗バーを表示するかどうか |
| `trainer.progress_bar_batches` | `false` | エポックの中のバッチ単位の進捗バーも表示するかどうか (1 エポックが長い場合に使う) |
| `trainer.early_stopping.use` | `false` | Early Stopping を使うかどうか |
| `trainer.early_stopping.patience` | `10` | 改善しないまま待つエポック数 |
| `trainer.early_stopping.min_delta` | `0.0` | これより小さい変化は改善とみなさない |
| `trainer.checkpoint.save_best` | `true` | `best.pt` を保存するかどうか |
| `trainer.checkpoint.save_last` | `true` | `last.pt` を保存するかどうか |
| `trainer.checkpoint.every_n_epochs` | `1` | `last.pt` を保存するエポックの間隔, 最後のエポックと Early Stopping で止めたエポックは必ず保存 |

ReduceLROnPlateau は `trainer.monitor` の値で学習率を下げる.
`scheduler.interval` を `step` にした場合や, `scheduler.params.mode` を `trainer.mode` と違う値にした場合はエラーになる.
`mode` を省略した場合は `trainer.mode` と同じになる.

監視する指標が nan や inf になった場合はエラーで止まる.
監視する指標が最良値とちょうど同じ値の場合は改善とみなさないので, 正解率のように同じ値を取りやすい指標を監視すると, ベストモデルは最初にその値に達したエポックのままになり, Early Stopping も止まりやすい. `trainer.monitor_tiebreak` に `valid/loss` などを指定すると, 同じ値の場合はその指標で比べ, ベストモデルの判定と Early Stopping の両方に使う. バッチごとの平均の丸め誤差で同じ正解率がわずかに違う値になることがあるので, 相対誤差 1e-6 以内は同じ値とみなす. Early Stopping では, 最良値より悪くないが `min_delta` 未満の改善の場合も比べる指標で判定する (`min_delta` は Early Stopping だけに使い, ベストモデルの判定には使わない). 比べる指標が nan や inf になった場合もエラーで止まる. 再開するときは, 比べる指標と向きを再開元と同じにする (違う場合はエラー). この設定を入れる前のチェックポイントから再開した場合は, 次に監視する指標が改善するまで比べない.
学習の途中で起きたエラーは `train.log` にも記録する. Optimizer と Scheduler のクラス名, `scheduler.interval`, ReduceLROnPlateau の組み合わせの誤りと, 存在しない `resume_from` は, 出力先を作る前に見つかる. Optimizer と Scheduler の引数の誤りは `Trainer` を作るときにエラーになる (出力先は作られた後). 自分で作った Optimizer や Scheduler を `Trainer` に渡す場合も, config の `optimizer` と `scheduler` の名前と組み合わせは確かめるので, 使わない設定も正しい値にしておく.

`last.pt` と `best.pt` には, モデル, Optimizer, Scheduler, 記録の全て, 乱数の状態, ベストモデルの重みが入る. 1 エポックが短い学習 (フルバッチなど) では保存が学習時間の大部分を占めることがあるので, `trainer.checkpoint.every_n_epochs` で `last.pt` の間隔を空けるか, 再開しない場合は `save_best` と `save_last` を `false` にする. `best.pt` は改善するたびに保存するので `every_n_epochs` では減らない (ベストモデルの重みは `last.pt` にも入るので, 再開だけなら `save_best: false` でもよい). 大きなモデルでは 1 ファイルが数 GB になる.

`train.log` のモデルの構造は `print(model)` の内容なので, 中身を `nn.Module` の子として持たない層 (外部ライブラリの層など) は構造が残らない. 再現に必要な情報は, モデルに `extra_repr` を書いて出すと良い.

`experiment.device: auto` は小さなモデルでも GPU を選ぶ. パラメータが少ないモデルや, 中で cpu を使うモデル (外部ライブラリの計算など) は `cpu` を明示した方が速いことがある.

## 出力

実験ごとに `<output_dir>/<name>/<日付>/<時刻>/` が作られる.

| ファイル | 内容 |
| --- | --- |
| `config.yaml` | 学習に使った config 全体, default として読み込めば同じ条件で学習できる. 再開した実験の場合は `trainer.resume_from` も残っているので, 最初から学習し直す場合は消す |
| `diff.yaml` | default から変えた項目 |
| `train.log` | 差分, 環境の情報, モデルの構造, エポックごとの指標, 結果 |
| `git_diff.patch` | 未コミットの変更がある場合のみ, commit id からの差分. 未追跡のファイルも含むが, `.gitignore` で除外したファイルと submodule の中の変更は含まない. バイナリファイルは変更されたことだけが記録される |
| `history.csv` | エポックごとの指標と, エポックを始めた時点の学習率 (`lr`, step ごとに変える場合はエポックの中の変化は残らない) |
| `<指標>_curve.png` | 指標ごとの学習曲線, 学習率の推移 (`lr_curve.png`) も含む |
| `best.pt` | ベストモデルのチェックポイント |
| `last.pt` | 最後のエポックのチェックポイント, ベストモデルの重みも含む. 再開に使う |
| `final_model.pt` | ベストモデルの重みだけ (`restore_best: false` の場合は最後のエポックの重み), cpu のテンソル |
| `run.json` | 実験の状態 (`running`, `completed`, `stopped`, `failed`), 開始, 更新, 終了の時刻 (タイムゾーン付き), 予定と終えたエポック数, 再開した最初のエポック, 監視する指標, ベストのエポックと値, Early Stopping で止めたエポック, `final_model.pt` の重み (`best` か `last`), 乱数シード, デバイス, プロセスの ID とホスト名, エラーの内容, 形式の版. 学習を始めたときに作り, エポックごとと終了時に更新 |
| `progress.json` | 学習中だけ置かれる, 今のフェーズ (`train`, `valid`), エポック, エポックの中で終えたバッチの数と総数, 経過時間. フェーズの始めと終わりに書き, 途中は 1 秒以上の間隔で上書きする. 学習を終えると消す |

### 学習の外からの確認と中断

`run.json` と `progress.json` は, 書きかけを読まないように一時ファイルに書いてから置き換える. ダッシュボードや別の端末のスクリプトから, 学習中の状態と進み具合を読める. 書き込めない場合 (ディスクの不足など) は `train.log` に警告を 1 回残し, 学習は続ける. 状態は `Trainer.fit` が書くので, `fit` を使わずに `train_epochs` などを直接呼んだ場合や, `train_epochs` を上書きした場合は残らない.

学習中の実験のディレクトリに `STOP` という名前のファイルを置くと, そのエポックを終えたところで学習を止める. 結果 (`final_model.pt` など) を保存し, 状態を `stopped` にする. `checkpoint.save_last` が true の場合は, 保存の間隔に関係なく `last.pt` も保存するので, そこから再開できる. エポックの途中では止めないので, 1 エポックが長い学習では止まるまで時間がかかる. 予定の最後のエポックで置いた場合と, Early Stopping と同じエポックで止まった場合は `completed` になる. `STOP` は学習を止めた後も残るので, 同じ出力先で `fit` をやり直す場合は消しておく.

```bash
touch outputs/wine/2026-11-12/10-00-00/STOP
```

キーボードの割り込み (Ctrl-C) で止めた場合も `stopped`, エラーで止まった場合は `failed` になる. プロセスが強制終了された場合は `running` のまま残るので, 同じ端末であれば `run.json` の `hostname` と `pid` のプロセスが動いているかで見分ける. 別の端末からは, `progress.json` の `updated_at` が長く更新されていないかで見分ける.

## ディレクトリ構成

```
ML-Found/
├── src/ml_found/
│   ├── config/             # config の型定義, 読み込み, 差分
│   ├── experiment/         # 出力先の作成, ロガー, シード, 環境の情報
│   ├── training/           # Trainer, Optimizer と Scheduler, Early Stopping, チェックポイント, 記録, 実験の状態
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
