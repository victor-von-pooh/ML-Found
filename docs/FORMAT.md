# プロジェクトと実験のディレクトリの形

ML-Found が他のシステムと受け渡すファイルの形を定める.
他のシステムは, 実験の結果を見る画面 (ML-Dashboard) や, 将来の問題の設定と推論の段階.
この形は ML-Found が持つ約束. ML-Found の中のコードに頼らずに読み書きする側は, この文書を頼りにする.
各項目の細かな意味は [DESIGN.md](DESIGN.md) の「出力」と「config の項目」にある.

## 版の決まり

版は整数で, `run.json`, `source.yaml`, `task.yaml` の `format_version` に書く.

- **`run.json` の版**: 実験のディレクトリ全体の版を兼ねる. 版を持たないファイル (`history.csv`, `diff.yaml`, `progress.json`, `model_summary.csv` など) の形を変える場合は, `run.json` の版を上げる
- **`source.yaml` の版**: `source.yaml` の形だけの版
- **`task.yaml` の版**: `task.yaml` の形だけの版 (後の段階 (推論など) が実験のディレクトリの他のファイルと別に読むため). `evaluations/` と `samples/` は版を持たず, `run.json` の版に従う. 新しい版の読み方は, 下の「読む側」と違う (「task.yaml」の節)
- **版を変えない変更**: ファイル, 項目, 列を足す. 選択肢の値 (`task.yaml` の `output` や `target.origin.type`, 見本の `kind` など) を足す. 読む側は, 知らないファイル, 項目, 列を無視し, 知らない選択肢の値は分からないもの (`unknown` や `file` の扱い) として読む (言い切りの元にしない)
- **版を上げる変更**: ファイル, 項目, 列を消す. 名前, 意味, 値の型を変える
- **読む側**: 知っている版より新しいファイルは, 読める項目だけを使い, 新しい版の ML-Found の出力であることを利用者に知らせる (読むのを止めない)
- `format_version` の無い `source.yaml` は版 1 として読む (中身は版 1 と同じ形)

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
| `task.yaml` | 任意. config の `task` か `write_task` で書いた場合 | 何の実験か (下の「task.yaml」). 受け持ち, 出力の意味, 正解の出どころ, 作り方 |
| `evaluations/<名前>.json` | 任意. `save_evaluation` で書いた場合 | データ全体の評価の表 (下の「evaluations」) |
| `samples/` | 任意. `save_sample` で書いた場合 | 見本の出力と一覧 `samples/index.json` (下の「samples」) |
| `policy/` | 任意. 強化学習の方策を `export_policy` で書き出した場合 | 方策の一式 (下の「方策の一式」). 独自の版を持つ |

- `task.yaml`, `evaluations/`, `samples/` は版 1 の任意のファイル. 書かなかった実験には無い. 読む側は, 無い場合を「分からない」として扱う
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

何の実験かを書く. 指標が何との一致かを, 実験のディレクトリだけから読めるようにするため (正解が今ある処理の出力なら, 測れるのはその処理との一致で, モデルはその処理を超えられない). 値の名前は, 問題の設定 (ML-Quest) とデータの書き出し (ML-EDA の `dataset.yaml` の版 2) のものを写す. 全ての項目を書き, 分からない値は null.

| 項目 | 型 | 中身 |
| --- | --- | --- |
| `format_version` | 整数 | 形式の版 (1) |
| `role` | 文字か null | 受け持ち. `predict` (予測する), `generate` (作る), `rank` (並べる), `group` (まとめる), `effect` (効果を見積もる) |
| `output` | 文字か null | 出力の形. `class`, `value`, `distribution`, `region`, `shape`, `text`, `image`, `table`, `ranking`, `cluster`, `effect` (ML-Quest の出力の形), `action` (続けて選ぶ行動. 強化学習の方策) |
| `what` | 文字か null | 出力の意味 (何を出すか. 人が読む文) |
| `target` | 辞書 | 正解. `kind` (正解の種類. `mask`, `point_cloud`, `column` など. 方策は `reward` (環境の報酬) を推奨. 文字か null), `origin` (正解の出どころ. 下) |
| `method` | 文字か null | 作り方. `rule` (規則), `pretrained` (既製のモデルをそのまま), `head` (既製の特徴の上に頭だけ学習), `partial` (一部を微調整), `finetune` (全体を微調整), `scratch` (一から学習) |
| `base` | 辞書 | 元にしたもの. `experiment` (元にした実験のディレクトリ), `weights` (元にした重みの名前か場所. Hugging Face のモデルの名前など, 書いた文字のまま). どちらも文字か null. `scratch` では両方 null |
| `dataset` | 辞書 | 学習に使ったデータセット. `path` (ML-EDA が書き出したディレクトリ), `name`, `format_version` (その `dataset.yaml` の版). どれも null がありうる |

`target.origin` の欄 (ML-EDA の `dataset.yaml` の版 2 の `target.origin` と同じ).

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
| `epoch` | 整数か null | 評価した重みのエポック (分からなければ null) |
| `created_at` | 文字 | 書いた時刻 (タイムゾーン付きの ISO 8601) |

- 同じ名前で書き直すと置き換える
- 読む側は `evaluations/*.json` を全て読む (一時ファイル `.<名前>.json.*.tmp` は無視する)
- **比べる相手との表の形 (推奨)**: 1 行を 1 つのグループ (全体なら 1 行) にし, 指標ごとに `<指標>` (この実験), `<指標>_baseline` (比べる相手), `<指標>_diff` (この実験 - 比べる相手) の 3 列を並べ, `baseline` に相手の名前を書く. 差の符号は値の差のままで, 良し悪しの向きは指標の名前から読む側が決める (小さいほど良い指標では負が良い). 画面は `_diff` の列を差として色分けできる
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

`index.json` は `{"samples": [...]}` で, 並びの 1 つが 1 つの見本 (書いた順).

| 項目 | 型 | 中身 |
| --- | --- | --- |
| `epoch` | 整数か null | エポック (学習の後は null) |
| `name` | 文字 | 見本の名前 (同じエポックの中で 1 つ. 評価の名前と同じ決まり) |
| `kind` | 文字 | `image` (PNG など), `text` (UTF-8 の文), `audio` (WAV など), `point_cloud` (PLY), `mask` (領域の番号を色分けした PNG), `file` (その他). 読む側は知らない種類を `file` として扱う |
| `file` | 文字 | `samples/` からの相対パス (区切りは `/`) |
| `split` | 文字か null | 見本の入力の分割 |
| `caption` | 文字か null | 添える説明 |
| `classes` | リストか null | `mask` の番号の表 (下). 他の種類は null |
| `convention` | 辞書か null | 値の決まり (点群の座標の単位や正規化で引いた中心など. 下). 無ければ null |
| `created_at` | 文字 | 書いた時刻 (タイムゾーン付きの ISO 8601) |

- `classes` と `convention` は版 1 の任意の項目. 読む側は無い場合を null と読む (`load_samples` は null を補って返す)
- **`mask` の番号の表** (`classes`): 並びの 1 つが 1 つの番号で, `id` (0 から 255 の整数), `name` (名前. 文字か null), `color` (PNG での色. `#rrggbb`). 書いた番号の全て (名前を渡した番号と, 配列に現れた番号) を番号の順に並べる. 色は決まった色の表 (PASCAL VOC と同じ, 番号のビットから作る 256 色. 0 は黒) から番号で選ぶので, 実験が違っても同じ番号は同じ色. 画面は PNG をそのまま出し, この表から凡例を作る
- **値の決まり** (`convention`): 書いた側の補足で, 形は決めない (JSON の辞書). 点群では `units` (座標の単位. `m`, `mm` など), `shift` (正規化で引いた中心の座標. 元の座標は `値 × scale + shift`), `scale` (正規化で割った大きさ), `axes` (軸の向き. `y_up` など) を推奨する
- ML-Found が配列から書く形: 画像は PNG (8 bit), 領域の番号は色分けした PNG (8 bit の RGB), 音は WAV (16 bit の PCM), 点群は PLY (ascii. 頂点の `x`, `y`, `z` だけ). ファイルを渡すと形を変えずに写す
- 同じエポックと名前で書き直すと, ファイルと一覧の項目を置き換える
- 一覧に無いファイル (書きかけの一時ファイル `.<名前>.*.tmp` など) は無視する. ただし直下の番号付きの画像は上の「直下の番号付きの画像」として読む
- 一覧の書き換えは同じプロセスの中では 1 つずつ行う. 複数のプロセス (DDP のランクなど) から同じ実験に書くと一覧の項目が落ちるので, 1 つのプロセス (ランク 0) だけが書く
- ML-Found が書く `task.yaml`, `evaluations/`, `samples/` のファイルは, `config.yaml` と同じ権限 (umask に従う) で作る

### 外から書くもの

学習中の実験のディレクトリに `STOP` を置くと, ML-Found が学習を止める. 中身が `now` (前後の空白と大文字・小文字は問わない) ならエポックの途中でも, それ以外ならエポックの区切りで止める. ML-Found は `STOP` を消さない (同じ `Trainer` で `fit` を呼び直した場合を除く).

外のシステムが実験のディレクトリに書くのは `STOP` だけにする. メモやタグなど, 外のシステム自身の情報は, 外のシステムの置き場に保存する.

## 方策の一式 (policy/, 版 1)

強化学習で学習した方策を, 守りと一緒に書き出すディレクトリ (`ml_found.rl.export_policy` が書く). 実験のディレクトリの `policy/` に置くのを推奨する. 任意のディレクトリで, 置かない実験もある (実験のディレクトリの版は変えない). 後の推論の段階 (ML-Inference) と画面が, ML-Found のコードに頼らずに読めるように形を決める.

```
policy/
├── policy.json     # 中身の説明 (下の表)
└── policy.pt       # 方策の重み (state_dict, cpu のテンソル)
```

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

1. 観測が数の配列でない (文字など), 形が空間と違う (同じ大きさの平らな配列は可), 有限でない値がある (float32 に丸めて無限大になるものも), 止める条件の範囲の外のどれかなら, 外れたときの動き. 範囲は元の値で比べる
2. モデルがエラーになるか, 出力に有限でない値があるか, 大きさが違えば, 外れたときの動き
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
