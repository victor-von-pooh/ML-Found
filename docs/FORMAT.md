# プロジェクトと実験のディレクトリの形

ML-Found が他のシステムと受け渡すファイルの形を定める.
他のシステムは, 実験の結果を見る画面 (ML-Dashboard) や, 将来の問題の設定と推論の段階.
この形は ML-Found が持つ約束. ML-Found の中のコードに頼らずに読み書きする側は, この文書を頼りにする.
各項目の細かな意味は [DESIGN.md](DESIGN.md) の「出力」と「config の項目」にある.

## 版の決まり

版は整数で, `run.json`, `source.yaml`, `task.yaml`, `samples/index.json`, `trajectories/<名前>.json` の `format_version` に書く.

- **`run.json` の版**: 実験のディレクトリ全体の版を兼ねる. 版を持たないファイル (`history.csv`, `diff.yaml`, `progress.json`, `model_summary.csv` など) の形を変える場合は, `run.json` の版を上げる
- **`source.yaml` の版**: `source.yaml` の形だけの版
- **`task.yaml` の版**: `task.yaml` の形だけの版 (後の段階 (推論など) が実験のディレクトリの他のファイルと別に読むため). `evaluations/` は版を持たず, `run.json` の版に従う. 新しい版の読み方は, 下の「読む側」と違う (「task.yaml」の節)
- **`samples/index.json` の版**: 見本の一覧の形だけの版 (今は 2). 新しい版の読み方は, 下の「読む側」と違う (「samples」の節)
- **`trajectories/<名前>.json` の版**: 評価の軌跡の表 (`<名前>.csv`) とその説明の形の版 (今は 1). 表は他の段階 (ML-EDA) が実験のディレクトリの他のファイルと別に読むため, 独自の版を持つ
- **版を変えない変更**: ファイル, 項目, 列を足す. 選択肢の値 (`task.yaml` の `output` や `target.origin.type`, 見本の `kind` など) を足す. 読む側は, 知らないファイル, 項目, 列を無視し, 知らない選択肢の値は分からないもの (`unknown` や `file` の扱い) として読む (言い切りの元にしない)
- **版を上げる変更**: ファイル, 項目, 列を消す. 名前, 意味, 値の型を変える
- **読む側**: 知っている版より新しいファイルは, 読める項目だけを使い, 新しい版の ML-Found の出力であることを利用者に知らせる (読むのを止めない)
- `format_version` の無い `source.yaml` は版 1 として読む (中身は版 1 と同じ形). `format_version` の無い `samples/index.json` も版 1 として読む

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
| `final_model.pt` | fit を終えたとき | 最終の重み (`state_dict`, cpu のテンソル). `config.yaml` の `trainer.ema.decay` が null でない実験では, 重みの指数移動平均 (EMA) の重み (名前と形は学習したモデルと同じ). `trainer.precision` と `trainer.checkpoint.trainable_only` に関わらず, 全ての重み (型はモデルの重みのまま. bf16 の混合精度でも変わらない) |
| `<指標>_curve.png`, `lr_curve.png`, `lr_<名前>_curve.png` | fit を終えたとき (記録が 1 件以上) | 学習曲線 |
| `lr_steps.csv`, `lr_steps.png` | fit を終えたとき (`scheduler.interval: step` で, 更新が 1 回以上の場合) | 更新ごとの学習率. 列は `step`, `epoch`, `lr` (名前付きの Optimizer では `lr_<名前>`) |
| `best.pt`, `last.pt` | 設定による | チェックポイント. 中の形は決めない (ML-Found で読む). `config.yaml` の `trainer.checkpoint.trainable_only` が true の実験では, 学習しないパラメータの重みを含まない |
| `git_diff.patch` | 未コミットの変更がある場合 | commit id からの差分 |
| `model_spec.yaml` | `start_experiment` の後. config の `model.spec` の場所にプロジェクトの中の普通のファイル (16 MiB まで) があり, 写せた場合 (再開した学習でも写す) | 学習に使ったモデルの定義 (ML-ModelControl の定義の YAML) の写し. 中身は変えずに写す. 写したかどうかと中身の sha256 は `source.yaml` の `model_spec` |
| `task.yaml` | 任意. config の `task` か `write_task` で書いた場合 | 何の実験か (下の「task.yaml」). 受け持ち, 出力の意味, 正解の出どころ, 作り方 |
| `evaluations/<名前>.json` | 任意. `save_evaluation` で書いた場合 | データ全体の評価の表 (下の「evaluations」) |
| `samples/` | 任意. `save_sample` で書いた場合 | 見本の出力と一覧 `samples/index.json` (下の「samples」) |
| `policy/` | 任意. 強化学習の方策を `export_policy` で書き出した場合 | 方策の一式 (下の「方策の一式」). 独自の版を持つ |
| `trajectories/<名前>.csv`, `trajectories/<名前>.json` | 任意. 強化学習の評価の軌跡を `save_trajectory` で書いた場合 | 1 歩が 1 行の軌跡の表と, その説明 (下の「trajectories」). 独自の版を持つ |

- `task.yaml`, `evaluations/`, `samples/`, `trajectories/` は任意のファイル. 書かなかった実験には無い. 読む側は, 無い場合を「分からない」として扱う
- プロジェクトが実験のディレクトリに自分のファイル (生成した画像, 問題の設定の写しなど) を足してよい. 読む側は, 知らないファイルを無視する. 学習曲線とプロジェクトの画像は, 上の名前で見分ける
- `model_spec.yaml` はプロジェクトが学習の後に同じ名前で作るか書き直してもよい (学習のスクリプトが自分で定義を写す場合). そのときの中身は `source.yaml` の `sha256` と合わないことがあるので, 読む側はファイルがあることでなく `source.yaml` の `model_spec` の `status` と `sha256` で, 学習を始めたときの定義かを見分ける
- 書きかけを読まないように, ML-Found は一時ファイル (`*.tmp`, `.run.json.*.tmp` など) に書いてから置き換える. 一時ファイルが一瞬見えることがあるので, 読む側は無視する

### source.yaml

| 項目 | 型 | 中身 |
| --- | --- | --- |
| `format_version` | 整数 | 形式の版 (1) |
| `script` | 文字か null | 実行したスクリプト |
| `default` | 文字か null | default の config |
| `experiment` | 文字か null | experiment の config |
| `overrides` | 文字のリストか null | 「項目=値」の上書き (書いた文字のまま) |
| `model_spec` | 辞書か null | モデルの定義 (config の `model.spec`) の写し (下). config に `model.spec` が無い, 空, 文字でない, 参照を解決できない場合は null |

- パスは, プロジェクトの中ならプロジェクトからの相対パス (区切りは `/`), 外なら絶対パス. 外のパス (画面で書き換えた config を置いた場所など) は, 後で消えていることがある
- null は分からないことを表す (ノートブックの中, `load_config` などを通らずに作った config など)
- `model_spec` の欄は次の 3 つ
    - `path` (文字): config の `model.spec` の場所. 相対パスはプロジェクトから辿り (`~` はホームに展開), 他のパスと同じく, プロジェクトの中ならプロジェクトからの相対パス (区切りは `/`), 外なら絶対パス. リンクは辿らずに書き, `..` と `.` は消して正規化する. ただしリンクの後の `..` で `link` にした場合 (`extlink/../m.yaml`) と `invalid` の場合は書いた文字のまま. プロジェクトの中の絶対パスと, リンクを通したプロジェクトの場所 (macOS の `/tmp` など) から書いた絶対パスは, プロジェクトからの相対パスにする
    - `status` (文字): `copied` (`model_spec.yaml` に写した), `missing` (ファイルが無い. `..` の直前が無いかディレクトリでない場合 (`nosuch/../m.yaml`) も), `outside` (プロジェクトの外を指す), `link` (場所か途中のディレクトリがシンボリックリンク), `not_file` (ディレクトリ, FIFO など普通のファイルでない. 末尾が区切りの `m.yaml/` と `m.yaml/.` も), `unreadable` (権限などで読めない), `too_large` (16 MiB を超える), `invalid` (場所として読めない書き方. `path` は書いた文字のまま), `write_failed` (読めたが実験のディレクトリに書けなかった). 読む側は知らない値を写していないものとして読む. 再開した学習 (`config.yaml` の `trainer.resume_from`) でも写す
    - `sha256` (文字か null): 写した中身の SHA-256 (16 進の小文字). `copied` の他は null
- `model_spec` が無い `source.yaml` (この欄の前の ML-Found が書いたもの) は, 写したかどうか分からないものとして読む

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
| `optimizers` | リストか null | Optimizer ごとの辞書. `name` (名前付きの Optimizer の名前, 1 つの場合は null), `optimizer` (クラスの名前), `modules` (受け持つ部分の名前のリスト, 1 つの場合は null), `scheduler` (クラスの名前, 使わない場合は null), `scheduler_params` (Scheduler の主な引数の辞書, 使わない場合は null. 下の「Scheduler の引数」). `Trainer` を作れなかった場合は null |
| `error` | 文字か null | 止まった原因のエラー |

- **Scheduler の引数** (`scheduler_params`): config から作った Scheduler も, 自分で作って `Trainer` に渡した Scheduler も, 同じ形で書く. 値は学習を始めるときのもの
    - **元にする値**: Scheduler の状態 (`state_dict()`) を, 状態の名前のまま書く. 学習の進み具合で変わる値 (`last_epoch`, `CyclicLR` 以外の `base_lrs`, ReduceLROnPlateau の `best` など), 内部の値, 他の引数から決まる値 (ReduceLROnPlateau の `mode_worse`, `default_min_lr`) は除く
    - **名前を変えて持つ引数**: 状態の名前になる (`OneCycleLR` の `epochs` と `steps_per_epoch` は `total_steps`, `pct_start` は `_schedule_phases` の区切りのステップ, `anneal_strategy` は `_anneal_func_type`, `CyclicLR` の `max_lr` は `max_lrs`, `base_lr` は `base_lrs`)
    - **Optimizer に書き込む値**: `OneCycleLR` の `max_lr`, `min_lr`, `base_momentum`, `max_momentum` を, パラメータグループごとのリスト `groups` に書く
    - **渡した関数**: `LambdaLR` と `MultiplicativeLR` の関数はリスト `lr_lambdas` に, `CyclicLR` の自分の `scale_fn` は `scale_fn` に書く. 関数は名前 (無名の関数は `<lambda>`, 名前を持たないオブジェクトはクラスの名前) の文字. `functools.partial` は `{"function": 元の関数, "args": 固定した位置の引数のリスト, "keywords": 固定した名前付きの引数の辞書}` (transformers の `get_linear_schedule_with_warmup` の `num_warmup_steps` と `num_training_steps` は `keywords` に入る)
    - **中に Scheduler を持つもの** (`SequentialLR`, `ChainedScheduler`): `schedulers` に中の Scheduler ごとの `{"scheduler": クラスの名前, "params": 同じ形の辞書}` を並べる. `SequentialLR` は区切り `milestones` も書く
    - **JSON に書けない値**: 回数の辞書 (`MultiStepLR` の `milestones`) は小さい順の並び (同じ値は回数だけ並べる). 無限大と非数は `"inf"`, `"-inf"`, `"nan"`, 数でも文字でもない値はその文字の表し
- `running` のまま残った実験は, 主にプロセスが強制終了されたもの (`fit` を使わずに学習した場合も残る). 同じホストなら `pid` で, 別のホストからは `progress.json` の `updated_at` で見分ける

### progress.json

| 項目 | 型 | 中身 |
| --- | --- | --- |
| `phase` | 文字 | 今のフェーズ (`train`, `train_eval`, `valid`) |
| `epoch`, `total_epochs` | 整数 | 今のエポックと予定のエポック数 |
| `batch`, `total_batches` | 整数, 整数か null | エポックの中で終えたバッチの数と総数 (分からない場合は null) |
| `stage` | 文字 | フェーズの中の段階. `batches` (バッチを回している) か `epoch_end` (エポックの終わりの処理 `on_epoch_end` を継承した場合だけ, その間. FID など高価な評価が入りうる). 処理を終えると `batches` に戻す. 無ければ `batches` と読む (版 1 のファイルにも無い場合がある). `add_callback` の関数の処理の間は `epoch_end` にならない |
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

指標の名前と数はプロジェクトによって違う. 値の無いところは空. 検証の間隔 (`trainer.valid_every_n_epochs`) が 2 以上の場合, 検証しないエポックの `valid/` と `train_eval/` の値は空 (列は最初に現れた順なので, それらの列が `lr` などの後に来ることもある. 読む側は列の名前で読む).

### task.yaml

何の実験かを書く. 指標が何との一致かを, 実験のディレクトリだけから読めるようにするため (正解が今ある処理の出力なら, 測れるのはその処理との一致で, モデルはその処理を超えられない). 値の名前は, 問題の設定 (ML-Quest) とデータの書き出し (ML-EDA の `dataset.yaml` の版 2 と 3) のものを写す. 全ての項目を書き, 分からない値は null.

| 項目 | 型 | 中身 |
| --- | --- | --- |
| `format_version` | 整数 | 形式の版 (1) |
| `role` | 文字か null | 受け持ち. `predict` (予測する), `generate` (作る), `rank` (並べる), `group` (まとめる), `effect` (効果を見積もる) |
| `output` | 文字か null | 出力の形. `class`, `value`, `distribution`, `region`, `shape`, `text`, `image`, `table`, `ranking`, `cluster`, `effect` (ML-Quest の出力の形), `action` (続けて選ぶ行動. 強化学習の方策) |
| `what` | 文字か null | 出力の意味 (何を出すか. 人が読む文) |
| `target` | 辞書 | 正解. `kind` (正解の種類. `mask`, `point_cloud`, `column` など. 方策は `reward` (環境の報酬) を推奨. 文字か null), `origin` (正解の出どころ. 下) |
| `method` | 文字か null | 作り方. `rule` (規則), `pretrained` (既製のモデルをそのまま), `head` (既製の特徴の上に頭だけ学習), `partial` (一部を微調整), `finetune` (全体を微調整), `scratch` (一から学習) |
| `base` | 辞書 | 元にしたもの. `experiment` (元にした実験のディレクトリ), `weights` (元にした重みの名前か場所. Hugging Face のモデルの名前など, 書いた文字のまま). どちらも文字か null. `scratch` では両方 null |
| `dataset` | 辞書 | 学習に使ったデータセット. `path` (ML-EDA が書き出したディレクトリ), `name`, `format_version` (その `dataset.yaml` の版. 2 か 3). どれも null がありうる |

`target.origin` の欄 (ML-EDA の `dataset.yaml` の版 2 と 3 の `target.origin` と同じ).

| 欄 | 型 | 中身 |
| --- | --- | --- |
| `type` | 文字 | `process` (今ある処理の出力), `annotation` (人が付けた), `measurement` (測った), `synthetic` (作った正確な正解), `environment` (環境の報酬. シミュレータなど. ML-EDA の `dataset.yaml` には無い値), `unknown` (分からない. 既定) |
| `name` | 文字か null | 正解を作った処理 (か道具) の名前 |
| `settings` | 任意か null | その処理の設定 |

- パス (`base.experiment`, `dataset.path`) は, `source.yaml` と同じく, プロジェクトの中ならプロジェクトからの相対パス (区切りは `/`), 外なら絶対パス
- 読む側は, `target.origin.type` が `process` なら「指標はその処理との一致を測り, その処理を超えられない」, `annotation` なら「付けた人との一致」, `environment` なら「環境の報酬で測る. 実機での結果は別に確かめる」と指標の横に書く (ML-Found の `describe_origin` が文を返す). 知らない値は `unknown` として説明する
- **方策の実験の例**: `role: effect`, `output: action`, `target.kind: reward`, `target.origin.type: environment`, `target.origin.name` に環境の名前 (`CartPole-v1` など), `method: scratch`. 方策は問題の設定 (ML-Quest) では効果の見積もりの扱いなので `role` は `effect`
- **`format_version` は 1 以上の整数**. 文字 (`'2'`), 小数 (`2.0`), 真偽値, 無い場合は誤りとして止まる (版 1 として読み違えないため)
- **新しい版の読み方 (`task.yaml` だけの決まり)**: 知っている版より新しい `task.yaml` は, 他の項目は読める項目だけを使うが, `target.origin` は分からないもの (`unknown`) として扱い, 新しい版であることを画面で知らせる (`load_task` は `target.origin` を `unknown` に置き換え, 返す辞書に `newer_version: true` を足す. `describe_origin` に版を渡すと言い切らない文を返す). `task.yaml` は「この処理を超えられない」のような言い切りの元なので, 意味の変わったかもしれない欄から言い切ると誤った説明になる (他のファイルは表示が欠けるだけで, 誤りは言わない)
- 書くのは `start_experiment` (config の `task` がある場合) と `write_task`. 後から書いた方で置き換える
- 読む側は, シンボリックリンク, 普通でないファイル (FIFO など), 1 MiB を超える `task.yaml` を読まず, 無いものとして扱う (ML-Found の `load_task`)
- `dataset.yaml` から埋めた `target.origin` は, 上の 3 つの欄だけを写す (ML-EDA が足した知らない欄は写さない)

### evaluations

学習の後のデータ全体の評価. 1 つの評価が 1 つのファイル `evaluations/<名前>.json` (JSON の辞書). グループごとの表, 曲線, 比べる相手 (今の処理, 前の実験) との差, 守る指標を, どれも表として書く.

| 項目 | 型 | 中身 |
| --- | --- | --- |
| `name` | 文字 | 評価の名前 (ファイルの名前と同じ. 英数字と `_`, `.`, `@`, `-` だけ. 先頭は `.` 以外) |
| `split` | 文字 | 評価に使った分割 (`test`, `valid` など) |
| `kind` | 文字 | `table` (表) か `curve` (曲線. `x` の列を横軸に, 他の数の列を縦軸に描く) |
| `x` | 文字か null | 曲線の横軸の列 (`curve` だけ) |
| `columns` | 文字のリスト | 列の名前 (並びは表示の順) |
| `rows` | 辞書のリスト | 1 行が 1 つの辞書 (列の名前から値). 値は数, 文字, 真偽値, null (nan と無限大は null) |
| `description` | 文字か null | 何の評価か (人が読む文) |
| `baseline` | 文字か null | 比べる相手の名前 (差の列を持つ場合. 列の中身はプロジェクトが決める) |
| `guard` | 真偽値 | 守る指標 (問題の設定で決めた, 悪くしてはいけない指標) の表か |
| `better` | 辞書 (無いファイルは空として読む) | 列ごとの指標の向き. 列の名前から `lower` (小さいほど良い) か `higher` (大きいほど良い). 書いていない列は向きが分からない (下の「指標の向き」) |
| `epoch` | 整数か null | 評価した重みのエポック (分からなければ null) |
| `created_at` | 文字 | 書いた時刻 (タイムゾーン付きの ISO 8601) |

- 同じ名前で書き直すと置き換える
- 読む側は `evaluations/*.json` を全て読む (一時ファイル `.<名前>.json.*.tmp` は無視する). シンボリックリンク, 普通でないファイル (FIFO など), 20 MiB を超えるファイルは読まず, `evaluations/` 自体がリンクなら評価が無いものとして扱う (ML-Found の `load_evaluations` と ML-Dashboard の画面で同じ)
- **指標の向き** (`better`): 書いた側が列ごとに指標の向きを決める (指標の名前だけでは向きが決まらないもの (`neg_log_loss`, 独自の指標など) を, 読む側が取り違えないため). 比べる相手との表の形の列 `<指標>`, `<指標>_baseline`, `<指標>_diff` は同じ向き (差はこの実験 - 比べる相手なので, `lower` の指標では負の差が良い). ML-Found は書くときにこの 3 つの列へ向きを広げて書く. 読む側も, `better` に無い列は同じ指標の列に書いた向きを使う (1 つの書いていない列に違う向きが届くファイルでは, その列は埋めず, 書いた列は書いた向きのまま使う). 終わり (`_baseline`, `_diff`) は 1 段ずつ外して指標とし, 広げた列からさらに変化が無くなるまで広げる (`acc_baseline_diff` の向きは `acc_baseline` に, そこから `acc` と `acc_diff` に届く). 広げるのは終わりを外す向きだけで, `<指標>` の向きは `<指標>_baseline_diff` には届かない. 広げる先は表 (`columns`) にある列だけで, 間の列が表に無いとその先には届かない (`acc_baseline` が無い表では, `acc_baseline_diff` の向きは `acc` と `acc_diff` に届かない). ML-Found が書いた `better` は, もう広げるところの無い形. 読む側は, `better` の向きを名前からの推測より優先し, `better` に無い列 (`better` を持たないファイルも) は名前から決めるか分からないとする. 値が `lower` と `higher` でない欄と, `columns` に無い列の欄は無視する. `better` を持たないファイルは空の辞書として読む
- **比べる相手との表の形 (推奨)**: 1 行を 1 つのグループ (全体なら 1 行) にし, 指標ごとに `<指標>` (この実験), `<指標>_baseline` (比べる相手), `<指標>_diff` (この実験 - 比べる相手) の 3 列を並べ, `baseline` に相手の名前を書く. 差の符号は値の差のままで, 良し悪しの向きは `better` に書く (書かなければ指標の名前から読む側が決める. 小さいほど良い指標では負が良い). 画面は `_diff` の列を差として色分けできる
- **別の形 (許す)**: 1 行を (グループ, 答えた側) の組にし, 答えた側の列 `answer` に `model` か比べる相手の名前を書く (差の列は持たない). 相手が 2 つ以上ある場合や, 相手ごとに指標の列が違う場合に使う. このときも `baseline` に相手の名前 (複数なら `, ` でつなぐ) を書く

### samples

見本の出力 (画像, 文, 音, 点群). 学習の途中 (エポックごと) と学習の後に書く.

```
samples/
├── index.json            # 一覧 (ML-Found が持つ名前)
├── recon/                # 見本の名前ごとのディレクトリ
│   ├── epoch_0005.png    # エポック 5 の見本 (4 桁の 0 埋め. 1 万以上はそのまま)
│   ├── epoch_0010.png
│   └── final.png         # 学習の後の見本 (エポックを渡さない場合)
├── caption/
│   └── epoch_0005.txt
└── epoch_010.png         # 直下の番号付きの画像 (プロジェクトが置いたもの)
```

- **ML-Found が持つもの**: `samples/index.json` と, `samples/<名前>/` の中の `epoch_<番号>.<拡張子>` と `final.<拡張子>`. ML-Found は `samples/` の直下に `index.json` の他のファイルを書かない
- **直下の番号付きの画像**: プロジェクト (GAN や拡散の生成画像など) が `samples/` の直下に置いた番号付きの画像 (`samples/epoch_010.png`) も正しい形. 読む側は, 一覧の項目に加えて, 直下の画像 (名前が `<前の部分><番号>.<png, jpg, jpeg, gif, webp>`) を見本として読む. 名前は番号の前の部分から末尾の `_`, `.`, `-` を除いたもの (空か `epoch` なら `samples`), エポックは番号. 見本の名前が直下のファイルとぶつかる場合 (`epoch_010.png` という名前など) と, 名前が `index.json` の場合は誤り
- 名前ごとのディレクトリに分けるのは, ML-Dashboard の画像の推移 (同じディレクトリで番号の前の部分が同じ画像を 1 つの並びにする) が, 直下の番号付きの画像とも混ぜずに名前ごとの並びにするため

`index.json` は `{"format_version": 2, "samples": [...]}` で, `format_version` は一覧の形式の版 (整数. 無ければ版 1), `samples` の並びの 1 つが 1 つの見本 (書いた順).

| 項目 | 型 | 中身 |
| --- | --- | --- |
| `epoch` | 整数か null | エポック (学習の後は null) |
| `name` | 文字 | 見本の名前 (同じエポックの中で 1 つ. 評価の名前と同じ決まり) |
| `kind` | 文字 | `image` (PNG など), `text` (UTF-8 の文), `audio` (WAV など), `point_cloud` (PLY), `mask` (領域の番号を色分けした PNG), `file` (その他). 読む側は知らない種類を `file` として扱う |
| `file` | 文字 | `samples/` からの相対パス (区切りは `/`) |
| `split` | 文字か null | 見本の入力の分割 |
| `caption` | 文字か null | 添える説明 |
| `classes` | リストか null | `mask` の番号の表 (下). 他の種類は null |
| `convention` | 辞書か null | 値の決まり (点群の座標の単位や正規化で足した量など. 下). 無ければ null |
| `created_at` | 文字 | 書いた時刻 (タイムゾーン付きの ISO 8601) |

- `classes` と `convention` は任意の項目. 読む側は無い場合を null と読む (`load_samples` は null を補って返す)
- **版 1 と版 2 の違い**: 値の決まりの `shift` の向きだけ. 版 2 は ML-EDA の書き出しの `shift` と同じ向き (正規化した座標 = 元の座標 + shift). 版 1 は逆の向き (元の座標 = 値 × scale + shift) なので, 版 1 の `shift` の数の符号を反転すると版 2 の意味になる. `load_samples` は版 1 の一覧の `shift` を版 2 の意味に写して返し, `save_sample` は版 1 の一覧に書き足すとき前の項目も版 2 の意味に写して版 2 で書く
- **新しい版の読み方 (`samples/index.json` だけの決まり)**: 知っている版より新しい一覧は, 他の項目は読める項目だけを使うが, `convention` は読まない (意味が変わったかもしれず, 座標を取り違えるため). `load_samples` は `convention` を null にし, 項目に `newer_version: true` を付ける (知っている版の一覧の項目は `newer_version: false`). `save_sample` は新しい版の一覧に書き足さずに止まる. 版 2 の一覧に書き足すのは版 2 を知っている ML-Found だけにする (版を知らない ML-Found は `format_version` を書かずに一覧を書き直すので, 版 1 と読まれて `shift` の向きを取り違えるため. 実験を続けるときは同じかより新しい ML-Found を使う). 版が 1 以上の整数でない一覧 (文字, 真偽値など) は誤り (`load_samples` は一覧を読めないものとして直下の画像だけを返す)
- **`mask` の番号の表** (`classes`): 並びの 1 つが 1 つの番号で, `id` (0 から 255 の整数), `name` (名前. 文字か null), `color` (PNG での色. `#rrggbb`). 書いた番号の全て (名前を渡した番号と, 配列に現れた番号) を番号の順に並べる. 色は決まった色の表 (PASCAL VOC と同じ, 番号のビットから作る 256 色. 0 は黒) から番号で選ぶので, 実験が違っても同じ番号は同じ色. 画面は PNG をそのまま出し, この表から凡例を作る
- **値の決まり** (`convention`): 書いた側の補足で, 形は決めない (JSON の辞書). 点群では `units` (座標の単位. `m`, `mm` など), `shift` (正規化で足した量. ML-EDA と同じ向きで, 正規化した座標 = 元の座標 + shift. 数か軸ごとの 3 つの数), `scale` (正規化の後に割った大きさ. 数か軸ごとの 3 つの数. 元の座標は `値 × scale - shift`), `axes` (軸の向き. `y_up` など) を推奨する
- ML-Found が配列から書く形: 画像は PNG (8 bit), 領域の番号は色分けした PNG (8 bit の RGB), 音は WAV (16 bit の PCM), 点群は PLY (ascii. 頂点の `x`, `y`, `z` だけ). ファイルを渡すと形を変えずに写す
- 同じエポックと名前で書き直すと, ファイルと一覧の項目を置き換える
- 一覧に無いファイル (書きかけの一時ファイル `.<名前>.*.tmp` など) は無視する. ただし直下の番号付きの画像は上の「直下の番号付きの画像」として読む
- 読む側は, シンボリックリンク, 普通でないファイル (FIFO など), 20 MiB を超える `index.json` を読まず, 読めない一覧として扱う. `samples/` 自体をリンクにすると, ML-Found は見本を読み書きしない (`load_samples` は一覧も直下の画像も読まずに空を返し, `save_sample` は書かずに止まる)
- 一覧の書き換えは同じプロセスの中では 1 つずつ行う. 複数のプロセス (DDP のランクなど) から同じ実験に書くと一覧の項目が落ちるので, 1 つのプロセス (ランク 0) だけが書く
- ML-Found が書く `task.yaml`, `evaluations/`, `samples/`, `trajectories/` のファイルは, `config.yaml` と同じ権限 (umask に従う) で作る

### trajectories

強化学習の評価で走らせた軌跡 (エピソードの 1 歩ごとの観測, 行動, 報酬, 終わったか). 1 つの軌跡が, 表 `trajectories/<名前>.csv` と説明 `trajectories/<名前>.json` の 2 つのファイル. 名前は評価の名前と同じ決まり. 表は 1 歩が 1 行の一般の形で, ML-Found の外で作った表 (利用者の手元の CSV や Parquet) も同じ列の決まりで読めるようにする. 説明は ML-Found が書く補足で, 利用者の表には無くてよい.

```
trajectories/
├── valid_best.csv      # 1 歩が 1 行の表
├── valid_best.json     # 表の説明 (版, 列と型, 観測と行動の空間, 全てのエピソードが入ったか)
├── guarded.csv
└── guarded.json
```

表の列 (並びはこの順. 観測と行動は要素の数だけ並ぶ).

| 列 | 型 | 中身 |
| --- | --- | --- |
| `episode` | 整数 | エピソードの番号 (0 から. 評価の中の番号) |
| `step` | 整数 | エピソードの中の歩の番号 (0 から) |
| `obs` か `obs_<番号>` | 数 | 行動を選ぶ前の観測. 下の「観測と行動の列」 |
| `action` か `action_<番号>` | 数か空 | 環境に渡した行動. 守りで止めた判断の行は空 |
| `reward` | 数か空 | 行動の後に環境が返した報酬. 守りで止めた判断の行は空 |
| `terminated` | 真偽値 | 環境が終了を知らせた歩か (棒が倒れたなど) |
| `truncated` | 真偽値 | 打ち切りの歩か (環境の打ち切り (TimeLimit など) と, ML-Found の歩数の上限 `max_episode_steps`, `max_steps` で切った歩) |
| `seed` | 整数か空 | エピソードを始めた乱数の種 (環境の乱数の続きなら空) |
| `guard` | 文字か空 | 守りの結果 (`ok`, `clipped`, `fallback`, `stop`, `handover`). 説明の `guarded` が `true` の表 (守りを当てた評価 `evaluate_policy`) だけにある列 |
| `guard_reasons` | 文字 | 守りが働いた理由 (複数は `; ` でつなぐ. 働かなければ空). `guard` と同じく, 説明の `guarded` が `true` の表だけ |
| `action_prob` | 数か空 | 記録した方策が, その観測でその行の行動を選んだ確率 (0 より大きく 1 以下). 守りで止めた判断の行は空. 記録した方策が確率を返した表 (説明の `with_action_prob` が `true`) だけにある列. 方策の見積もり (重点サンプリング) に使う |

- **観測と行動の列**: 1 つの値の観測と行動 (Discrete と形が () の Box) は 1 つの列 `obs`, `action`. 並び (形のある Box) は, 平らにした (C の順) 要素ごとに列を分け, `obs_0`, `obs_1`, ... と `action_0`, ... にする (形が (1,) でも `obs_0`). 1 つの列に並びを入れる形は使わない (CSV には並びの型が無く, Parquet でも同じ列の決まりで読めるようにするため). 同じ表に `obs` と `obs_<番号>` は並ばない. 要素の数の上限は観測と行動のそれぞれ 256 (超える観測は表にしない). Discrete と Box の他の空間 (Dict, Tuple など) は表にしない
- **値の書き方**: UTF-8 で, 1 行目が見出し. 区切りは `,`, 改行は `\n`, `,` や改行を含む文字は `"` で囲む (CSV の決まり). 空の値は欠け (null) で, 守りで止めた判断の行の行動, 報酬, 確率, 形の合わない観測と数でない観測 (全ての観測の列を空にする) だけに使う. 真偽値は `true` か `false`. 整数は 10 進. 小数は元の型 (観測の float32 など) で読み直せる最短の書き方 (`1e-05` のような指数も). nan は `nan`, 無限大は `inf` か `-inf` (欠けと見分けるため空にしない). 番号の空間 (Discrete) の観測と行動は整数, 他は小数
- **1 エピソードの並び**: 行は `episode`, `step` の順で, 1 つのエピソードの行は続けて並ぶ. エピソードの最後の行は, `terminated` か `truncated` が `true` の行か, 守りで止めた判断の行 (`guard` が `stop` か `handover`. 行動と報酬は空で, 環境は進めていない. 観測は止めたときの観測). 最後の歩の後の観測 (環境が返した次の観測) は表に無い
- **報酬の和と歩数**: エピソードの報酬の和は `reward` の和 (空を除く), 環境を進めた歩数は `reward` のある行の数. `evaluate_policy` の行 (`reward`, `length`) と同じになる
- **大きさ**: 表は 64 MiB まで. ML-Found は先頭のエピソードから, 貯めた歩の数の上限 (`max_rows`. 既定 10 万歩) と大きさの上限 (記録の間も, 書く文字数を足し上げて同じ 64 MiB で止める) に収まるエピソードまでをエピソードの単位で書き, 途中で切ったエピソードは書かない (入らなかったエピソードの数は説明の `dropped_episodes`)
- 同じ名前で書き直すと置き換える. ML-Found は表と説明の両方を一時ファイルに書き終えてから, 説明, 表の順に置き換える. どちらかに失敗すると, 両方の一時ファイルを消し, 元の表と説明を残す. 一時ファイル (`.<名前>.csv.*.tmp` など) は無視する
- 読む側は, シンボリックリンク, 普通でないファイル (FIFO など), 64 MiB を超える表と 1 MiB を超える説明を読まない (上限は ML-Found の公開の `MAX_TRAJECTORY_BYTES` と `MAX_TRAJECTORY_INFO_BYTES`). `trajectories/` 自体がリンクなら軌跡が無いものとして扱う (ML-Found は書かずに止まる)
- **利用者の表**: ML-Found の外で作った表も, 上の列の名前と意味で読む. 少なくとも `episode`, `step`, 観測の列, 行動の列, `reward`, `terminated`, `truncated` を持つ. 他の列は無くてよく, 知らない列は無視する. 説明 (`<名前>.json`) の無い表は, 列の名前から観測と行動の列を決め (`obs`, `obs_<番号>`, `action`, `action_<番号>`), 型は値から決める. Parquet の表も同じ列の名前と型 (真偽値は真偽値の型) で読む

#### trajectories/<名前>.json

| 項目 | 型 | 中身 |
| --- | --- | --- |
| `format_version` | 整数 | 軌跡の表と説明の形の版 (今は 1) |
| `name` | 文字 | 軌跡の名前 (ファイルの名前と同じ) |
| `file` | 文字 | 表のファイルの名前 (`<名前>.csv`) |
| `source` | 文字か null | 何の評価の軌跡か. `evaluation_loader` (学習中の検証の `EvaluationLoader`) か `evaluate_policy` (書き出した方策の守り付きの評価) |
| `env` | 文字か null | 環境の名前 (Gymnasium の登録の名前. `CartPole-v1` など). 分からなければ null |
| `epoch` | 整数か null | 評価した重みのエポック (分からなければ null) |
| `description` | 文字か null | 何の軌跡か (人が読む文) |
| `columns` | 文字のリスト | 表の列 (表の見出しと同じ順) |
| `types` | 辞書 | 列ごとの型. `integer`, `number`, `boolean`, `string` |
| `observation_columns` | 文字のリスト | 観測の列 (要素の順) |
| `action_columns` | 文字のリスト | 行動の列 (要素の順) |
| `observation_space` | 辞書か null | 観測の空間 (「方策の一式」の「空間」と同じ形). Discrete と Box の他は null |
| `action_space` | 辞書か null | 行動の空間 (同じ形) |
| `guarded` | 真偽値 | 守りの結果の列 (`guard`, `guard_reasons`) があるか |
| `with_action_prob` | 真偽値 | 記録した方策が行動を選んだ確率の列 (`action_prob`) があるか |
| `episodes` | 整数 | 表に入れたエピソードの数 |
| `rows` | 整数 | 表の行の数 (見出しを除く) |
| `complete` | 真偽値 | 評価で走らせた全てのエピソードが表に入ったか |
| `dropped_episodes` | 整数 | 上限で表に入らなかったエピソードの数 (表の最後のエピソードの後のもの) |
| `created_at` | 文字 | 書いた時刻 (タイムゾーン付きの ISO 8601) |

- `with_action_prob` の無い説明は, 表に `action_prob` の列があるかで決める
- **新しい版の読み方**: 知っている版より新しい説明は, 読める項目だけを使い, 新しい版の ML-Found の出力であることを利用者に知らせる (上の「読む側」). 版が 1 以上の整数でない説明は誤りとして読まず, 説明の無い表 (利用者の表) として読む

### 外から書くもの

学習中の実験のディレクトリに `STOP` を置くと, ML-Found が学習を止める. 中身が `now` (前後の空白と大文字・小文字は問わない) ならエポックの途中でも, それ以外ならエポックの区切りで止める. ML-Found は `STOP` を消さない (同じ `Trainer` で `fit` を呼び直した場合を除く). `STOP` は普通のファイルで置く. リンク, ディレクトリ, FIFO の `STOP` もエポックの区切りで止める要求とするが, 中身は読まない (`now` と書いてもエポックの途中では止めない). 中身はリンクでない普通のファイルの 1 KiB までだけを読み, 超える場合は `now` でないものとする.

外のシステムが実験のディレクトリに書くのは `STOP` だけにする. メモやタグなど, 外のシステム自身の情報は, 外のシステムの置き場に保存する.

## 方策の一式 (policy/, 版 1)

強化学習で学習した方策を, 守りと一緒に書き出すディレクトリ (`ml_found.rl.export_policy` が書く). 実験のディレクトリの `policy/` に置くのを推奨する. 任意のディレクトリで, 置かない実験もある (実験のディレクトリの版は変えない). 後の推論の段階 (ML-Inference) と画面が, ML-Found のコードに頼らずに読めるように形を決める.

```
policy/
├── policy.json     # 中身の説明 (下の表)
└── policy.pt       # 方策の重み (state_dict, cpu のテンソル)
```

`policy.pt` はモデルの `state_dict` 全体で, 連続の行動の方策の対数の標準偏差 (`log_std`) と価値の関数 (`value`) の重みのように, 行動を決めるのに使わない重みも入る. `head` が読むのはモデルの `forward` の出力だけで, 読む側は使わないキーがあってよい.

- **版**: `policy.json` の `format_version`. 守りを読み違えると危ないので, 他のファイルと違い, 読む側は知っている版より新しい一式と版の無い一式を読まずに止まる. 版を変えない変更と上げる変更の区別は, 上の「版の決まり」と同じ. ただし空間, `guards`, `fallback` は書き損じを見つけるため, 知らない項目があれば読まずに止まる (これらに項目を足す場合は版を上げる). 一番上の知らない項目は無視する
- **実機**: 一式は方策と守りの説明まで. 実機で動かす仕組みは ML-Found と推論の段階の外にあり, 方策を使ってよいかと守りの範囲は人が決める

### policy.json

| 項目 | 型 | 中身 |
| --- | --- | --- |
| `format_version` | 整数 | 形式の版 (1) |
| `kind` | 文字 | `policy` |
| `head` | 文字 | モデルの出力の読み方. `categorical` (行動ごとのスコア. 最も高い行動を選ぶ. 行動が `discrete` の場合) か `deterministic` (出力が行動. 行動が `box` の場合) |
| `observation_space` | 辞書 | 観測の空間. `box` だけ (下の「空間」) |
| `action_space` | 辞書 | 行動の空間. `discrete` か `box` |
| `guards` | 辞書 | 守り (下の「guards」) |
| `weights` | 文字 | 重みのファイルの名前 (`policy.pt` に限る) |
| `weights_sha256` | 文字 | 重みのファイルの sha256 (16 進の小文字 64 文字). 読む側は確かめ, 違えば読まない |
| `model_class` | 文字 | 書き出したモデルのクラスの名前 (`モジュール.クラス`. 読む側は同じ組み立てのモデルを用意する. コードは自動では読まない) |
| `created_at` | 文字 | 書き出した時刻 (タイムゾーン付きの ISO 8601) |
| `metadata` | 辞書 | 書き出した側の補足 (環境の名前, 学習の方法など). 形は決めない. 推奨の形は下の「metadata の推奨の形」 |

### 空間

| 項目 | 型 | 中身 |
| --- | --- | --- |
| `type` | 文字 | `discrete` (番号で選ぶ) か `box` (数の配列) |
| `n`, `start` | 整数 | `discrete` の行動の数と最初の番号 (行動は `start` から `start + n - 1`). 真偽値は不可 |
| `shape` | 整数のリスト | `box` の形 |
| `low`, `high` | 数か null のリスト | `box` の要素ごとの下限と上限を平らにしたもの. null は制限なし (無限大). 数は有限 (NaN と Infinity は JSON の外なので書かない. 読む側はエラーにする) |

### guards

| 項目 | 型 | 中身 |
| --- | --- | --- |
| `action_low`, `action_high` | 数か null のリストか null | `box` の行動の範囲. 空間の範囲との狭い方に clip する. null は空間の範囲のまま |
| `allowed_actions` | 整数のリストか null | `discrete` で選んでよい行動. 最もスコアの高い行動が含まれなければ, 含まれる中の最も高い行動にする. null は全て |
| `observation_low`, `observation_high` | 数か null のリストか null | 止める条件. 観測の要素がこの範囲を外れたら外れたときの動きにする. 要素の null とリストの null は制限なし |
| `fallback` | 辞書 | 外れたときの動き. `kind` が `stop` (止める), `handover` (元の制御に戻す), `action` (決めた行動 `action` にする. `discrete` は番号, `box` は数のリストで, 行動の範囲の中) |

守りを当てる順は次のとおり. 結果は `action` (行動. `stop` と `handover` では null), `status` (`ok`, `clipped`, `fallback`, `stop`, `handover`), `reasons` (守りが働いた理由の文字のリスト).

1. 観測が数の配列でない (文字や, float にできない桁の大きな整数など), 形が空間と違う (同じ大きさの平らな配列は可), 有限でない値がある (float32 に丸めて無限大になるものも), 止める条件の範囲の外のどれかなら, 外れたときの動き. 範囲は元の値で比べる
2. モデルがエラーになるか, 出力が数にできないか (桁の大きな整数も), 出力に有限でない値があるか, 大きさが違えば, 外れたときの動き
3. 行動の範囲を当てる (`box` は clip, `discrete` は選んでよい行動). 変えた場合は `clipped`
4. `box` の行動は float32 にする. 丸めで範囲の外に出た要素は内側の隣の float32 に寄せ, 無限大になるか範囲に float32 が無い場合は外れたときの動き. 決めた行動 (`fallback` の `action`) も同じく寄せる

### metadata の推奨の形

`model_class` は読み込んだモジュールの名前 (スクリプトの隣の `model.py` なら `model.PolicyNet`) なので, それだけではモデルを組み立て直せない. 読む側 (推論の段階, 画面) が同じ組み立てのモデルを用意できるように, `metadata` に次の項目を書くのを推奨する. どれも任意で, 読む側は無い項目を分からないものとして扱う (方策の一式の版は変えない).

- `env_id`: 環境の名前 (`CartPole-v1` など)
- `algorithm`: 学習の方法 (`REINFORCE` など)
- `experiment`: 書き出した実験のディレクトリ (分かれば)
- `model`: 組み立て方の辞書. 次のどちらか (両方でもよい)
    - `class` と `args`: クラスの名前 (`モジュール.クラス`) と, コンストラクタに名前で渡す引数の辞書 (`{"obs_dim": 4, "hidden_dim": 64, "n_actions": 2}` など). 読む側は同じクラスを用意し, `args` を渡して作る
    - `spec`: モデルの定義 (ML-ModelControl の定義のファイル) の場所. 一式のディレクトリからの相対パス (区切りは `/`) で, 一式に写して置くのを推奨する (一式だけを渡しても組み立てられるように)
