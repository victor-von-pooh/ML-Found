# プロジェクトと実験のディレクトリの形

ML-Found が他のシステムと受け渡すファイルの形を定める.
他のシステムは, 実験の結果を見る画面 (ML-Dashboard) や, 将来の問題の設定と推論の段階.
この形は ML-Found が持つ約束. ML-Found の中のコードに頼らずに読み書きする側は, この文書を頼りにする.
各項目の細かな意味は [README](../README.md) の「出力」と「config の項目」にある.

## 版の決まり

版は整数で, ファイルごとに `format_version` に書く (今は `run.json` と `source.yaml`).

- **版を変えない変更**: ファイルや項目を足す. 読む側は, 知らないファイルと項目を無視する
- **版を上げる変更**: ファイルや項目を消す. 名前, 意味, 値の型を変える
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
│       ├── train.yaml          # 既定の experiment の config (差分が無くても置く)
│       └── ...                 # 任意. 差分の config (下の階層にも置ける)
├── train.py                    # 学習のスクリプト (複数あってよい)
└── outputs/                    # 実験のディレクトリ (experiment.output_dir)
```

- **見分け方**: `configs/default/train.yaml` を持つディレクトリをプロジェクトとする
- **config の重ね方**: default に experiment を重ね, その上に「項目=値」の上書きを重ねる. experiment と上書きには, default にある項目だけを書ける
- **スクリプトの引数**: 学習のスクリプトは `load_project_config` で config を作り, `start_experiment` に渡す. そのため, どのスクリプトも次の形で起動できる. 相対パスはプロジェクトのディレクトリから辿る

  ```bash
  python <スクリプト> [--experiment <experiment の config>] [項目=値 ...]
  ```

- **スクリプトが複数ある場合**: 別の default を使うスクリプトは, `train_<名前>.py` から `configs/default/<名前>.yaml` と `configs/experiment/<名前>.yaml` を使う (`load_project_config` の `default` と `experiment` に渡す). スクリプトの外からは, この名前の対応で default を選ぶ
- **ML-Found が書くもの**: プロジェクトの中で ML-Found が書くのは `<output_dir>` の下だけ. 外から学習を止める `STOP` のファイルは, 実験のディレクトリに外から置く (下の「外から書くもの」)

## 実験のディレクトリの形 (版 1)

1 回の学習で, `<プロジェクト>/<output_dir>/<experiment.name>/<日付>/<時刻>/` を作る.
`<日付>` は `YYYY-MM-DD`, `<時刻>` は `HH-MM-SS` で, 同じ秒に始めた実験は `HH-MM-SS_1` のように番号を付ける.

### ファイル

| ファイル | いつあるか | 中身 |
| --- | --- | --- |
| `config.yaml` | 常に | 学習に使った config 全体. default として読み込めば同じ条件で学習できる |
| `diff.yaml` | 常に | default から変えた項目. `trainer.epochs` のように `.` でつないだ名前ごとに, default の値 (`default`) と変えた後の値 (`experiment`. 上書きも含む) |
| `source.yaml` | 常に | 起動のしかた (下の「source.yaml」) |
| `train.log` | 常に | 人が読むログ. 形は決めない |
| `run.json` | 学習を始めた後 | 実験の状態 (下の「run.json」) |
| `progress.json` | 学習中だけ | 進み具合. 学習を終えると消す |
| `history.csv` | 1 エポック以上を終えた後 | エポックごとの記録 (下の「history.csv」) |
| `model_summary.csv` | 学習を始めた後 | パラメータとバッファの一覧 |
| `final_model.pt` | 学習を終えた後 | 最終の重み (`state_dict`, cpu のテンソル) |
| `best.pt`, `last.pt` | 設定による | チェックポイント. 中の形は決めない (ML-Found で読む) |
| `git_diff.patch` | 未コミットの変更がある場合 | commit id からの差分 |
| `lr_steps.csv` | 更新ごとに学習率を変える場合 | `step`, `epoch`, `lr` の列 |
| `*.png` | 学習を終えた後 | 学習曲線. 名前と数は決めない |

プロジェクトが実験のディレクトリに自分のファイル (生成した画像, 写したモデルの定義など) を足してよい. 読む側は, 知らないファイルを無視する.

### source.yaml

| 項目 | 型 | 中身 |
| --- | --- | --- |
| `format_version` | 整数 | 形式の版 (1) |
| `script` | 文字か null | 実行したスクリプト |
| `default` | 文字か null | default の config |
| `experiment` | 文字か null | experiment の config |
| `overrides` | 文字のリストか null | 「項目=値」の上書き (書いた文字のまま) |

- パスは, プロジェクトの中ならプロジェクトからの相対パス (区切りは `/`), 外なら絶対パス
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
| `monitor`, `mode` | 文字 | 監視する指標と, その向き (`min` か `max`) |
| `select_best` | 真偽値 | ベストモデルを選ぶか |
| `best_epoch`, `best_value` | 整数と数か null | ベストのエポックと値 |
| `early_stopped_epoch` | 整数か null | Early Stopping で止めたエポック |
| `final_weights` | 文字か null | `final_model.pt` の重み (`best`, `last`, `partial`) |
| `seed` | 整数か null | 乱数のシード |
| `device` | 文字 | 学習したデバイス |
| `optimizers` | リストか null | Optimizer ごとの `name`, `optimizer`, `modules`, `scheduler` |
| `error` | 文字か null | 止まった原因のエラー |

- `running` のまま残った実験は, プロセスが強制終了されたもの. 同じホストなら `pid` で, 別のホストからは `progress.json` の `updated_at` で見分ける
- 書きかけを読まないように, 一時ファイルに書いてから置き換える

### history.csv

1 行が 1 エポック. 列は `epoch` (1 から) と, 次の名前の指標.

- `train/<指標>`, `valid/<指標>`, `train_eval/<指標>` (学習用データを推論モードで評価し直した値)
- `lr` (エポックを始めた時点の学習率. 名前付きの Optimizer では `lr_<名前>`)
- `step` が `extra` で返した値 (接頭辞なし)

指標の名前と数はプロジェクトによって違う. 値の無いところは空.

### 外から書くもの

学習中の実験のディレクトリに `STOP` を置くと, ML-Found が学習を止める. 中身が `now` ならエポックの途中でも, それ以外ならエポックの区切りで止める. ML-Found は `STOP` を消さない (同じ `Trainer` で `fit` を呼び直した場合を除く).

外のシステムが実験のディレクトリに書くのは `STOP` だけにする. メモやタグなど, 外のシステム自身の情報は, 外のシステムの置き場に保存する.
