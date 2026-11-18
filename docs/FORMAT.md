# プロジェクトと実験のディレクトリの形

ML-Found が他のシステムと受け渡すファイルの形を定める.
他のシステムは, 実験の結果を見る画面 (ML-Dashboard) や, 将来の問題の設定と推論の段階.
この形は ML-Found が持つ約束. ML-Found の中のコードに頼らずに読み書きする側は, この文書を頼りにする.
各項目の細かな意味は [README](../README.md) の「出力」と「config の項目」にある.

## 版の決まり

版は整数で, `run.json` と `source.yaml` の `format_version` に書く.

- **`run.json` の版**: 実験のディレクトリ全体の版を兼ねる. 版を持たないファイル (`history.csv`, `diff.yaml`, `progress.json`, `model_summary.csv` など) の形を変える場合は, `run.json` の版を上げる
- **`source.yaml` の版**: `source.yaml` の形だけの版
- **版を変えない変更**: ファイル, 項目, 列を足す. 読む側は, 知らないファイル, 項目, 列を無視する
- **版を上げる変更**: ファイル, 項目, 列を消す. 名前, 意味, 値の型を変える
- **読む側**: 知っている版より新しいファイルは, 読める項目だけを使い, 新しい版の ML-Found の出力であることを利用者に知らせる (読むのを止めない)
- 版を書く前の ML-Found が作った `source.yaml` には `format_version` が無い. 中身は版 1 と同じ形なので, 版 1 として読む

## プロジェクトの形 (版 1)

プロジェクトは, ML-Found を使って学習するディレクトリ.

```
<プロジェクト>/
├── configs/
│   ├── default/
│   │   ├── train.yaml          # 必須. 基準の config
│   │   └── <名前>.yaml         # 任意. スクリプトごとの基準の config
│   └── experiment/
│       ├── train.yaml          # 必須. 既定の experiment の config (差分が無くても置く. `--experiment` を省くと使う)
│       └── ...                 # 任意. 差分の config (下の階層にも置ける)
├── train.py                    # 学習のスクリプト (複数あってよい)
└── outputs/                    # 実験のディレクトリ (experiment.output_dir)
```

- **見分け方**: `configs/default/train.yaml` を持つディレクトリをプロジェクトとする
- **config の重ね方**: default に experiment を重ね, その上に「項目=値」の上書きを重ねる. experiment と上書きには, default にある項目だけを書ける. ただし Optimizer と Scheduler の引数 (`optimizer.params`, `scheduler.params`, `optimizers.<名前>.params`, `optimizers.<名前>.scheduler.params`) は, クラスを変えると必要な引数も変わるので, 自由に足せる
- **スクリプトの引数**: 学習のスクリプトは `load_project_config` で config を作り, `start_experiment` に渡す. そのため, どのスクリプトも次の形で起動できる. 起動する場所はどこでもよい

  ```bash
  python <スクリプト> [--experiment <experiment の config>] [項目=値 ...]
  ```

  `--experiment` の相対パスは, プロジェクトのディレクトリから辿り, 無い場合は実行した場所から辿る

- **スクリプトが複数ある場合**: 別の default を使うスクリプトは, `train_<名前>.py` に `configs/default/<名前>.yaml` と `configs/experiment/<名前>.yaml` を使わせる (`load_project_config` の `default` と `experiment` に渡す) のを推奨の名付けとする. スクリプトの外からスクリプトの default を選ぶ側 (ML-Dashboard) は, 次の順に選ぶ
    1. スクリプトの本文に `configs/default/<名前>.yaml` が 1 種類だけ書かれていれば, それ (本文は文字を探すだけで, 実行しない)
    2. `configs/default/<名前>.yaml`, 次に `configs/default/train_<名前>.yaml`
    3. どれも無ければ `configs/default/train.yaml`

  experiment の既定も同じく, `configs/experiment/<名前>.yaml`, `train_<名前>.yaml`, `train.yaml` の順. 前の実験と同じ設定で起動し直す場合は, その実験の `source.yaml` の `default` を使う
- **ML-Found が書くもの**: ML-Found が書くのは実験のディレクトリの中だけ. 外から学習を止める `STOP` のファイルは, 実験のディレクトリに外から置く (下の「外から書くもの」)

## 実験のディレクトリの形 (版 1)

1 回の学習で, `<output_dir>/<experiment.name>/<日付>/<時刻>/` を作る.
`experiment.output_dir` が相対パスならプロジェクトのディレクトリから辿り, 絶対パスならプロジェクトの外にもなる.
`<日付>` は `YYYY-MM-DD`, `<時刻>` は `HH-MM-SS` (どちらも学習したホストの地方時) で, 同じ秒に始めた実験は `HH-MM-SS_1` のように番号を付ける.

### ファイル

「fit を終えたとき」は, `fit` が `completed` か `stopped` (`STOP`, Early Stopping, `add_callback` の関数による停止) で終えたときを指す. エラー (`failed`) とキーボードの割り込み (`stopped`) で止まった実験には, このときに書くファイルが無い.

| ファイル | いつあるか | 中身 |
| --- | --- | --- |
| `config.yaml` | `start_experiment` の後 | 学習に使った config 全体. Scheduler の引数の `auto` などは, 学習を始めるときに補った値で 1 回書き換える |
| `diff.yaml` | `start_experiment` の後 | default から変えた項目. `trainer.epochs` のように `.` でつないだ名前ごとに, default の値 (`default`) と変えた後の値 (`experiment`. 上書きも含む). 片方にしか無い項目 (足した Optimizer の引数など) の値は, 無いことを表す印の文字 `<未定義>` |
| `source.yaml` | `start_experiment` の後 | 起動のしかた (下の「source.yaml」) |
| `train.log` | `start_experiment` の後 | 人が読むログ. 形は決めない. 学習中と失敗した実験の指標は, ここからしか分からない |
| `run.json` | `fit` を始めたとき. `Trainer` の作成や `fit` の外のエラーで止まった場合も書く | 実験の状態 (下の「run.json」). ノートブックの中のエラー, `sys.exit()`, メインでないスレッドのエラーで止まった場合は無いことがある |
| `progress.json` | 学習中だけ | 進み具合 (下の「progress.json」). 学習を終えると消す |
| `model_summary.csv` | エポックのループに入る直前 (設定の確認と再開の復元の後) | パラメータとバッファの一覧 (下の「model_summary.csv」). それより前に止まった実験には無い |
| `history.csv` | fit を終えたとき (記録が 0 件でも見出しは書く) | エポックごとの記録 (下の「history.csv」) |
| `final_model.pt` | fit を終えたとき | 最終の重み (`state_dict`, cpu のテンソル) |
| `<指標>_curve.png`, `lr_curve.png`, `lr_<名前>_curve.png` | fit を終えたとき (記録が 1 件以上) | 学習曲線 |
| `lr_steps.csv`, `lr_steps.png` | fit を終えたとき (`scheduler.interval: step` で, 更新が 1 回以上の場合) | 更新ごとの学習率. 列は `step`, `epoch`, `lr` (名前付きの Optimizer では `lr_<名前>`) |
| `best.pt`, `last.pt` | 設定による | チェックポイント. 中の形は決めない (ML-Found で読む) |
| `git_diff.patch` | 未コミットの変更がある場合 | commit id からの差分 |

- プロジェクトが実験のディレクトリに自分のファイル (生成した画像, 写したモデルの定義など) を足してよい. 読む側は, 知らないファイルを無視する. 学習曲線とプロジェクトの画像は, 上の名前で見分ける
- 書きかけを読まないように, ML-Found は一時ファイル (`*.tmp`, `.run.json.*.tmp` など) に書いてから置き換える. 一時ファイルが一瞬見えることがあるので, 読む側は無視する

### source.yaml

| 項目 | 型 | 中身 |
| --- | --- | --- |
| `format_version` | 整数 | 形式の版 (1) |
| `script` | 文字か null | 実行したスクリプト |
| `default` | 文字か null | default の config |
| `experiment` | 文字か null | experiment の config |
| `overrides` | 文字のリストか null | 「項目=値」の上書き (書いた文字のまま) |

- パスは, プロジェクトの中ならプロジェクトからの相対パス (区切りは `/`), 外なら絶対パス. 外のパス (画面で書き換えた config を置いた場所など) は, 後で消えていることがある
- null は分からないことを表す (ノートブックの中, `load_config` などを通らずに作った config など)

### run.json

| 項目 | 型 | 中身 |
| --- | --- | --- |
| `format_version` | 整数 | 形式の版 (1) |
| `status` | 文字 | `running`, `completed`, `stopped`, `failed` のどれか |
| `pid`, `hostname` | 整数, 文字 | 学習のプロセスと, 動かしたホスト |
| `started_at`, `updated_at`, `finished_at` | 文字か null | タイムゾーン付きの ISO 8601 の時刻 |
| `duration_sec` | 数か null | 所要時間 (秒) |
| `planned_epochs`, `completed_epochs`, `start_epoch` | 整数 | 予定と終えたエポック数, 始めたエポック |
| `monitor` | 文字か null | 監視する指標. `fit` を始める前に止まった実験では, config に書いた値 (書かなければ null) |
| `mode` | 文字 | 監視する指標の向き (`min` か `max`) |
| `select_best` | 真偽値 | ベストモデルを選ぶか |
| `best_epoch`, `best_value` | 整数と数か null | ベストのエポックと値. まだ無い場合とベストモデルを選ばない場合は null |
| `early_stopped_epoch` | 整数か null | Early Stopping で止めたエポック |
| `final_weights` | 文字か null | `final_model.pt` の重み (`best`, `last`, `partial`) |
| `seed` | 整数か null | 乱数のシード |
| `device` | 文字 | 学習したデバイス |
| `optimizers` | リストか null | Optimizer ごとの辞書. `name` (名前付きの Optimizer の名前, 1 つの場合は null), `optimizer` (クラスの名前), `modules` (受け持つ部分の名前のリスト, 1 つの場合は null), `scheduler` (クラスの名前, 使わない場合は null). `Trainer` を作れなかった場合は null |
| `error` | 文字か null | 止まった原因のエラー |

- `running` のまま残った実験は, 主にプロセスが強制終了されたもの (`fit` を使わずに学習した場合も残る). 同じホストなら `pid` で, 別のホストからは `progress.json` の `updated_at` で見分ける

### progress.json

| 項目 | 型 | 中身 |
| --- | --- | --- |
| `phase` | 文字 | 今のフェーズ (`train`, `train_eval`, `valid`) |
| `epoch`, `total_epochs` | 整数 | 今のエポックと予定のエポック数 |
| `batch`, `total_batches` | 整数, 整数か null | エポックの中で終えたバッチの数と総数 (分からない場合は null) |
| `stage` | 文字 | フェーズの中の段階. `batches` (バッチを回している) か `epoch_end` (エポックの終わりの処理 `on_epoch_end` を継承した場合だけ, その間. FID など高価な評価が入りうる). 処理を終えると `batches` に戻す. 無ければ `batches` と読む (この項目を足す前の ML-Found が書いたファイル. 版は同じ 1). `add_callback` の関数の処理の間は `epoch_end` にならない |
| `elapsed_sec` | 数 | 経過時間 (秒) |
| `updated_at` | 文字 | 更新した時刻 (タイムゾーン付きの ISO 8601). 1 秒以上の間隔で上書きする |

### model_summary.csv

| 列 | 中身 |
| --- | --- |
| `name` | パラメータかバッファの名前 (`state_dict` のキー) |
| `kind` | `parameter` か `buffer` |
| `module` | 持っている層のクラスの名前 |
| `shape` | 形. 次元を `x` でつなぐ (`16x4`. 0 次元は空) |
| `dtype` | 型 (`float32` など. `torch.` を除いたもの) |
| `numel` | 要素数 |
| `requires_grad` | 学習するか (`True` か `False`) |
| `same_as` | 同じ重みを共有している場合は最初の名前. 共有していなければ空 |

### history.csv

1 行が 1 エポック. 列は `epoch` (1 から) と, 次のもの.

- `train/<指標>`, `valid/<指標>`, `train_eval/<指標>` (学習用データを推論モードで評価し直した値). `train/grad_norm` と `train/grad_norm_<名前>` は勾配のノルムに使う名前
- `lr` (エポックを始めた時点の学習率. 名前付きの Optimizer では `lr_<名前>`)
- `step` が `extra` で返した値 (名前に `/` を含まない)

指標の名前と数はプロジェクトによって違う. 値の無いところは空.

### 外から書くもの

学習中の実験のディレクトリに `STOP` を置くと, ML-Found が学習を止める. 中身が `now` (前後の空白と大文字・小文字は問わない) ならエポックの途中でも, それ以外ならエポックの区切りで止める. ML-Found は `STOP` を消さない (同じ `Trainer` で `fit` を呼び直した場合を除く).

外のシステムが実験のディレクトリに書くのは `STOP` だけにする. メモやタグなど, 外のシステム自身の情報は, 外のシステムの置き場に保存する.
