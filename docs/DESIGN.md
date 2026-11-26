# 設計

ML-Found の今の設計と決まりをまとめる. まだやっていないことは [TODO.md](../TODO.md) にある.

## 1. 役割と位置づけ

PyTorch のモデルを config から自動で学習させる基盤のライブラリ. プロジェクトから git submodule として取り込んで使い回す.

プロジェクト側で書くのは **データ**, **モデル**, **1 ステップの計算** の 3 つだけで, それ以外の実験管理と学習ループは ML-Found が受け持つ.
対象は学習までで, 推論は推論の段階 (ML-Inference) かプロジェクトが行う.

「機械学習を全て GUI 上で行う」アプリ (問題の設定, データの分析, モデルの作成・選定, モデルの学習・評価, 推論の段階に分けて作ることを目指している) の中では, 「モデルの学習・評価」の段階のうち学習を受け持つ (画面は ML-Dashboard). 単体でも使える. ML-Found が行う評価は, 学習中の検証用データでの評価と, 学習の後にプロジェクトが書くデータ全体の評価の置き場まで.

- プロジェクトの形 (README の「プロジェクトの構成」) と, 実験ごとに残すディレクトリの形は ML-Found が決める. ML-Dashboard は実験のディレクトリに残るファイルを読み, config の確認と学習の起動には ML-Found のコードを使う
- プロジェクトのスクリプトは人が書くか, 問題の設定の段階 (ML-Quest) が書き出す骨組みを使う. どちらも同じ形のプロジェクトとして動く
- モデルの組み立て (ML-ModelControl) と推論 (ML-Inference) は対象外. Lightning などに依存せず, 素の PyTorch だけで書く

## 2. 他のリポジトリとの約束

形と版の決まりの詳しいところは [FORMAT.md](FORMAT.md) にある.

| 約束 | 持つか読むか | 版 |
| --- | --- | --- |
| プロジェクトの形 (`configs/default`, `configs/experiment`, 学習のスクリプト) | 持つ (ML-Dashboard と ML-Quest の骨組みが使う) | 版 1 |
| 実験のディレクトリ (`config.yaml`, `source.yaml`, `run.json`, `progress.json`, `history.csv`, `final_model.pt` など) | 持つ (ML-Dashboard と ML-Inference が読む) | 版 1. ファイルごとの版は FORMAT.md |
| 何の実験か (`task.yaml`), データ全体の評価 (`evaluations/`), 見本 (`samples/`) | 持つ (ML-Dashboard が読む) | FORMAT.md の「task.yaml」「evaluations」「samples」. 見本の一覧は版 2 (点群の `shift` は ML-EDA の書き出しと同じ向き). 版 1 も読む |
| 方策の一式 (`policy/`) | 持つ (`policy.json` と `apply_guards` で, 他の段階も同じ守りを当てられる) | 版 1 |
| 評価の軌跡の表 (`trajectories/<名前>.csv` と説明の `<名前>.json`) | 持つ (ML-EDA が読み, ML-Fathom が分析する. 利用者の手元の表も同じ列の決まりで読めるように, 一般の表の形にする. ML-Found も方策の見積もり (`estimate_policy_value`) で自分の書いた表を読む) | 版 1 |
| ML-EDA が書き出したデータセット (`dataset.yaml`) | 読む (`task.dataset` から受け持ちと正解を埋める. ml_eda には頼らない) | 版 2 と 3 (版 3 は版 2 に項目の中身の手順を足したもので, 受け持ちと正解は同じ形). 版 1 と知らない版では止まる |
| 受け持ち, 出力の形, 正解の種類と出どころの値の名前 | ML-Quest と ML-EDA の名前を写す | 無し |
| モデルの定義 (ML-ModelControl の定義の YAML. config の `model.spec`) | 中身を読まずに実験に写す (`model_spec.yaml`. ml_modelcontrol には頼らない) | 無し (ファイルのまま写す) |
| 追加の依存の登録簿 (`describe_requirements`, `environment_packages`) と, Optimizer と Scheduler の登録簿 (`describe_optimizers`, `describe_schedulers`) | 持つ (ML-Dashboard と ML-Quest が読む. ML-Quest の骨組みは, 学習のスクリプトが import する段階のライブラリ (`eda`, `modelcontrol`, `inference`) だけを config の `requires` に書く) | 無し (関数の戻り値) |
| 画面が使う確かめと読み込みの関数 (下の一覧) | 持つ (ML-Dashboard が使う. 公開するのは各パッケージの `__all__` にある名前だけで, 中のモジュールの名前と場所は約束しない) | 無し (関数の戻り値) |

画面 (ML-Dashboard) は, 学習を始める前の確かめ, config の読み方, 実験のディレクトリの読み込みに, 次の公開の名前を使う. 画面が同じ決まりを自分で持たず, ML-Found の中を変えても画面が壊れないため.

- `ml_found.experiment`: `start_experiment` が出力先を作る前にする確かめのうち, 実験の名前 (`check_experiment_name`), 学習の設定 (`check_trainer_config`), 計算の精度をデバイスで使えるか (`check_precision(cfg, device)`), 再開元 (`check_resume_path(cfg, project_dir)`), 始める前に読む重み (`check_init_weights(cfg, project_dir)`). 見本の一覧と形式の版の読み込み (`read_samples_index`). 評価のファイルの大きさの上限 (`MAX_EVALUATION_BYTES`. 画面がファイルごとの理由を出すために評価を自分で読む場合も, `load_evaluations` と同じ上限を使う)
- `ml_found.training`: Optimizer と Scheduler の確かめ (`check_optim_config`, `check_optim_params`, 引数の値の範囲の `check_param_values(cls, params)`, 仮のパラメータで作れるかの `try_build_optim(cfg, device)`, 学習を始めるときに補う Scheduler の引数の `missing_data_size_params`). クラスの解決 (`resolve_optimizer_class`, `resolve_scheduler_class`). 名前付きの Optimizer の扱い (`uses_named_optimizers`, 名前ごとの config を返す `named_views`)
- `ml_found.config`: 「項目=値」の上書きの読み方 (`parse_overrides`) と, torch のクラスに渡す引数を持つ項目の名前 (`params_sections`. 名前付きの Optimizer の項目も含む)
- `ml_found.rl`: 方策の重みのハッシュ (`file_sha256`). 評価の軌跡の説明の版と大きさの上限 (`TRAJECTORY_FORMAT_VERSION`, `MAX_TRAJECTORY_INFO_BYTES`. 画面が軌跡の一覧のために説明を自分で読む場合も, 同じ上限を使う). 記録した軌跡からの方策の見積もり (`estimate_policy_value`)

始める前の確かめは 1 つの関数にまとめず, 1 つずつ出す. 画面は必須の引数の不足 (学習のスクリプトが `start_experiment` の前に補う場合がある) を止めずに注意にし, デバイスの誤りも残りを確かめてから返し, 仮の Optimizer は cpu で作るので, 確かめの順と扱いが `start_experiment` (最初の誤りで止まる) と違うため.

## 3. 用語

| 用語 | 意味 |
| --- | --- |
| プロジェクト (project) | config と学習のスクリプトを持つディレクトリ. ML-Found を submodule として取り込む. 形は [FORMAT.md](FORMAT.md) の「プロジェクトの形」 |
| config | 学習の設定. 既定の `configs/default/train.yaml` と, そこからの差分の `configs/experiment/train.yaml` を重ね, 「項目=値」で上書きする |
| 実験 (experiment) | 1 回の学習. 出力先 `<output_dir>/<name>/<日付>/<時刻>/` (実験のディレクトリ) に config, 記録, チェックポイント, 状態を残す (下の「出力」) |
| step | 1 バッチの計算 (順伝播, 損失, 指標) をする関数 `step(model, batch, phase)`. プロジェクトが書く (下の「学習スクリプトと step の決まり」) |
| チェックポイント (checkpoint) | 再開に使う `last.pt` と, ベストモデルの `best.pt`. モデル, Optimizer, Scheduler, 記録, 乱数, loader (強化学習の環境の乱数と状態) の状態を持つ |
| ベストモデル (best) | 監視する指標 (`trainer.monitor`) が最も良いエポックの重み. 学習の後の `final_model.pt` は既定でこの重み |
| 方策の一式 (policy) | 強化学習で学習した方策の重み, 組み立て直しの引数, 守りの書き出し (`policy/`. 下の「方策の書き出しと守り」) |
| 追加の依存 (requirement) | 任意の機能に要るライブラリと, その確かめ方と入れ方の登録簿の 1 項目 (下の「追加の依存」) |

## 4. コードの置き場所

```
ML-Found/
├── src/ml_found/
│   ├── config/             # config の型定義, 読み込み, 差分, 起動のしかた
│   ├── experiment/         # 出力先の作成, ロガー, シード, 環境の情報, 処理されなかったエラーの記録, 何の実験か, 評価と見本
│   ├── training/           # Trainer, Optimizer と Scheduler とその登録簿, Early Stopping, 重みの指数移動平均 (EMA), 微調整の補助 (混合精度, 始める前に読む重み, 学習するパラメータだけのチェックポイント), チェックポイント, 記録, 実験の状態, 生成の画像の FID, 異常の検知の指標, 効果を見積もるの指標, 値段を変える判断の指標, 推薦の指標, 時間をまたぐ量の判断の指標
│   ├── rl/                 # 強化学習の経験の記録, 環境とのやり取りの loader, REINFORCE, Actor-Critic, 方策の一式と守り, 守り付きの評価, 評価の軌跡の表, 記録した軌跡からの方策の見積もり
│   └── utils/              # デバイス, バッチの転送, 学習曲線, パラメータの一覧, 追加の依存の登録簿, 出力先のファイルをリンクを辿らずに読む関数
├── examples/
│   ├── minibatch/          # wine データセットの分類, DataLoader で学習する例
│   ├── fullbatch/          # グラフのノード分類, マスクで学習と検証を分ける例
│   ├── cartpole/           # CartPole の方策の学習と, 守りと一緒の書き出し (追加の依存 rl)
│   └── pendulum/           # Pendulum の連続の行動の方策を Actor-Critic で学習し, 守りと一緒に書き出して守り付きで評価 (追加の依存 rl)
├── tests/
├── CODING_STYLE.md         # コーディングスタイルの基準
└── pyproject.toml
```

## 5. 追加の依存

任意の機能に要るライブラリは本体の依存に含めず, 追加の依存 (pip の extras) として要るときだけ入れる. ML-Found は機能ごとに「要るもの, 入っているかの確かめ方, 入れ方」の一覧 (登録簿) を持ち, 足りないまま使うと (入っていないか, 版が下限より古い), 次の一手 (pip のコマンドか公式の手順) を案内して止まる.

| 機能 | 要るもの (確かめ方) | 入れ方 |
| --- | --- | --- |
| `rl` | `gymnasium` 1.0 以上 (モジュールが見つかるかと版) | `pip install -e "./ML-Found[rl]"` (追加の依存 `rl`) |
| `mujoco` | `gymnasium` 1.0 以上と `mujoco` (モジュールが見つかるか. `gymnasium` は版も) | `pip install -e "./ML-Found[mujoco]"`. 描画には OpenGL が要る ([公式の手順](https://mujoco.readthedocs.io/en/stable/python.html)) |
| `ros2` | `rclpy` か `ros2` のコマンドのどちらか | pip では入らない. [公式の手順](https://docs.ros.org/) で入れ, setup のスクリプトを source してから Python を起動する |
| `cuda` | `torch.cuda.is_available()` | NVIDIA のドライバと, [PyTorch の手順](https://pytorch.org/get-started/locally/) で CUDA 付きの PyTorch |
| `fid` | `pytorch_fid` (モジュールが見つかるか. 生成の画像の FID の特徴を取り出す Inception. 下の「生成の画像の FID」) | `pip install -e "./ML-Found[fid]"` (追加の依存 `fid`. torchvision と scipy も入る). 学習済みの重みは初めて使うときに取る |
| `eda` | `ml_eda` と `pyarrow` (ML-EDA. データセットの読み込みと前処理. 書き出したデータセットは parquet) | `pip install "ml-eda[parquet] @ git+https://github.com/victor-von-pooh/ML-EDA.git"` (追加の依存 `parquet` で pyarrow も入る) |
| `modelcontrol` | `ml_modelcontrol` (ML-ModelControl. モデルの定義から作る) | `pip install "ml-modelcontrol @ git+https://github.com/victor-von-pooh/ML-ModelControl.git"` |
| `inference` | `ml_inference` (ML-Inference. 学習したモデルで予測し, 判断に使う) と `pyarrow` (データセットの parquet を読む) | `pip install "ml-inference @ git+https://github.com/victor-von-pooh/ML-Inference.git"` (ML-EDA (pyarrow を含む) と ML-ModelControl も一緒に入る) |

```python
from ml_found import MissingRequirementError, describe_requirements, require

try:
    require("rl")  # 無いか古ければ入れ方を示す MissingRequirementError (ImportError の子) を送出
except MissingRequirementError as error:
    print(error)  # 入れ方の案内
entries = describe_requirements()  # 全ての機能の一覧 (JSON にできる)
```

- `describe_requirements(names=None, check=True)` は, 機能ごとに `name`, `description`, `checks` (確かめ方ごとの `kind`, `name`, `ok`, `error`, `version`, `min_version`), `match` (`all` か `any`), `available`, `install` (`kind` が `pip` か `guide`, `extra`, `packages`, `url`, `command`, `extra_command`), `note` を返す. 画面 (ML-Dashboard) のカードや, 問題の設定の受け渡し (ML-Quest) で, 足りないものを早めに知らせるのに使う. `check=False` では確かめない (`ok` と `available` が `None`. JSON では null)
- 確かめ方は import せずにモジュールを探す (`importlib.util.find_spec`) ので, 一覧を作るだけで重いライブラリを読み込まない. ただし `a.b` のような名前では親のパッケージ `a` を import する
- 版の下限 (`min_version`) を持つ確かめ方は, 見つかったモジュールの版を import せずにパッケージの情報から読み (`importlib.metadata.version`. 読むパッケージの名前は `distribution`, 無ければ `name`), 下限より古ければ満たしていない (`ok` が `false`) とし, `version` に入っている版を, `error` に「版 0.29.1 は古い. 1.0 以上が要る」のような説明を入れる. `require` は入っていない時と同じ入れ方を添え, 1 行目を「要るものの版が足りません: gymnasium (0.29.1. 1.0 以上が要る)」にして止まる (`MissingRequirementError` の `outdated` に版が足りないもの, `missing` に入っていないもの). pip のコマンドは版の指定を含むので, そのまま古い版を入れ替える
- 下限は pyproject.toml の任意の依存の下限と同じにする (テストで pip の指定と一緒に確かめる) (今は `gymnasium>=1.0` だけ. `mujoco` と段階ごとのライブラリは下限を持たない). 版の文字は PEP 440 の順で比べ (`1.0.dev1 < 1.0a1 < 1.0rc1 < 1.0 < 1.0.post1`. `1.0` と `1.0.0` は同じ), `+` の後のローカルの版 (`2.11.0+cpu` の `cpu`) とエポック (`1!`) は比べない (`1!0.1` は 1.0 以上と見ず, 止まる側に倒れる). 比べ方は ML-Found が持ち, `packaging` には依存しない. 版を読めない場合 (パッケージの情報が無いか, 読めない形の版) は止めず, `ok` を `true` にして `error` に確かめていないことを残す
- プロジェクトの機能は `register_requirement(name, description, checks, install, match, note)` で加えられる (`ml_found.utils`. ML-Found の機能の名前は上書きできない). `module` の確かめ方には `min_version` と `distribution` も書ける (他の種類に書くか, 読めない版の文字か, `distribution` が pip で入れるときのパッケージの名前の形 (`[` や版の指定や空白を含まない) でなければエラー)
- `environment_packages(names, include_found=True)` は, プロジェクトの仮想環境を作るときに 1 回の `pip install` に渡す指定の一覧を返す. 先頭が ML-Found 自身 (`ml-found @ git+https://github.com/victor-von-pooh/ML-Found.git`. `include_found=False` で除く) で, 続けて機能の pip の指定を書いた順に並べる (同じ指定は 1 回). pip で入らない機能 (`ros2`, `cuda`) は含めないので, `describe_requirements` の入れ方で案内する. 登録簿に無い名前はエラー. 画面 (ML-Dashboard) がプロジェクトの環境を作るときに使う
- 段階ごとのライブラリ (`eda`, `modelcontrol`, `inference`) は PyPI に公開していないので, git のリポジトリから入れる. ML-Found の追加の依存 (`extra`) にはしない (ML-Found が他の段階のライブラリに頼らないため)

学習に要る機能は, config の一番上の `requires` に機能の名前のリストで書ける (例: `requires: [rl]`). `start_experiment` は出力先を作る前に, 書いた順に `require` で確かめ, 足りなければ入れ方を示す `MissingRequirementError` で止まる. 画面 (ML-Dashboard) は同じ項目を読み, 足りないまま学習を起動しないので, 画面から始めても端末から始めても同じ決まりで止まる.

```yaml
# configs/default/train.yaml
requires: [rl]
```

- 値は文字のリスト (1 つだけでもリスト). 1 つの文字だけ, 同じ名前が 2 回, 登録簿に無い名前はエラーになる (登録簿の名前を添える). 書かないか `null` の場合は何も確かめない. 値の参照 (`${...}`) は解決して読む
- default に書くとそのプロジェクトの全ての学習に効く. experiment の config は default に無い項目を書けないので, 学習ごとに変える場合は default に `requires: []` などを書き, experiment で置き換える (リストは置き換わり, 足し合わない. 例: default の `requires: [rl]` を experiment の `requires: []` で外す)
- プロジェクトの機能 (`register_requirement`) の名前も書ける. ただし `start_experiment` の前に登録する. 画面は自分のプロセスで登録されない名前を注意にするだけで止めないので, 足りなければ `start_experiment` が止める
- 書いた名前だけを確かめるので, 書き忘れは見つけられない (そのときは学習のスクリプトの `require` や `make_env` が止める)

## 6. config の決まり

default には ML-Found の項目とプロジェクト固有の項目を書く.
ML-Found の項目は, 書かなかったものが既定値になる.

experiment で書けるのは default にある項目の値だけで, 項目の追加や構造の変更はエラーになる.
例外は `optimizer.params` と `scheduler.params` で, クラスを変えると必要な引数も変わるので自由に書ける.
クラス名を変えた場合は default の引数は捨てられる.

**参照 (`${...}`) の確かめ**: `start_experiment` は他の確かめより先に (出力先を作る前に), config 全体の参照を解決できるかを確かめる. 読み込み関数 (`load_config` など) も差分を求める前に同じ確かめをするので, 読み込みの後に書き込んだ値も, 読み込み関数を通らない config も同じ決まりで止まる. 解決できない参照は全て, 項目の場所 (`model.dim`, `paths[1]` など), 書いた参照, 理由を並べた `ValueError` になる. 次の場合は誤りにする.

- 無い項目を指す参照 (`${data.dim}` で `data.dim` が無い)
- 無い環境変数の参照 (`${oc.env:NAME}` で既定値を書かず, 環境変数が無い). 既定値 (`${oc.env:NAME,既定値}`) を書いた場合と, 環境変数がある場合は止めない
- 登録されていない resolver の参照. プロジェクトで登録した resolver は, `start_experiment` の前に (読み込み関数を使う場合はその前に) 登録すれば使え, 確かめるときにも呼ばれる (誤りを出した場合は, 誤りの種類を添えて止まる)
- 循環する参照 (`a: ${b}` と `b: ${a}` など)
- 値の決まっていない項目 (`???`) を指す参照. `???` そのものは誤りにしない

誤りの文には解決した値 (環境変数の値など) と OmegaConf の文 (値を含み得る) を入れず, 項目の場所と書いた参照と理由だけを入れる (「25. 安全」). 確かめでは参照を解決するので, 参照の resolver を 1 回呼ぶ. `use_cache` の無い resolver (乱数や時刻など) は, 確かめで得た値と `config.yaml` に保存する値や学習で読む値が別になりうる. 呼ぶと副作用のある resolver は, 呼ぶ回数が 1 回増える.

`start_experiment` は出力先を作る前に, `optimizer.params` と `scheduler.params` の名前がクラスの引数にあるか (typo の確認), 必須の引数 (`StepLR` の `step_size` など) があるか, 値が登録簿の範囲 (下の `describe_optimizers()` などで返すもの. 例: `lr` は 0 以上, `ReduceLROnPlateau` の `factor` は 0 以上 1 未満) に入るかを確かめる. 範囲は意味のある値の範囲で, PyTorch の確認より狭いものもある (`momentum` は 1 以下など). 補間 (`${...}`) は解決した値で確かめ, 範囲を持つ引数に引用符で囲んだ数や真偽値を書いた場合もエラーにする. 値の範囲は torch のクラスだけを確かめ, 自作のクラスは確かめない. 任意の名前の引数 (`**kwargs`) や位置だけの引数を受け取る自作のクラスは確かめない. 最後に, 学習と同じデバイスの仮のパラメータで Optimizer と Scheduler を実際に作ってみて (データの量で決まる引数は仮の値で補う), 範囲を持たない引数の値や自作のクラスの値の誤り (PyTorch やクラスが作るときに出すエラー) も見つける. 値の決まっていない引数 (`???`) がある場合は, 作ってみる確認はしない (解決できない補間は, その前の確認でエラーになる). 自分で作った Optimizer や Scheduler を `Trainer` に渡す場合も, config の引数は確かめるので, 使わない引数は消しておく.

選べる Optimizer と Scheduler とその引数 (種類, 既定値, 必須か, 説明, 範囲. 範囲の端を含まない場合は `exclusive_minimum` と `exclusive_maximum` が true. 実装の都合で使う引数は `advanced` が true) と, そのまま使えるか (`supported`, 使えない理由は `note`), closure が必要か (`requires_closure`, Optimizer のみ), 勾配を使うか (`uses_gradients`, Optimizer のみ) は `describe_optimizers()` と `describe_schedulers()` で取得できる. ダッシュボードの入力フォームなどに使う. Muon と SparseAdam は, そのままでは使えないので `supported` が false で理由を返す. 他の Scheduler を組み合わせるものと関数を受け取るもの (`SequentialLR`, `ChainedScheduler`, `LambdaLR`, `MultiplicativeLR`) は返さない.

プロジェクトで作った Optimizer は, `register_optimizer` で登録すると, config の `optimizer.name` に名前だけで書け, `describe_optimizers()` にも加わる (ダッシュボードの登録簿は, ダッシュボードのプロセスで登録しないと出ない). 範囲を渡した引数は, torch のクラスと同じく出力先を作る前に確かめる. `start_experiment` の前に登録する (再開するスクリプトでも登録する). 登録できるのはプロジェクトのクラスだけで, torch のクラスや torch の名前と同じ名前, 同じクラスの別の名前は使えない. `param_notes` は引数ごとに `description`, `minimum`, `maximum`, `exclusive_minimum`, `exclusive_maximum` を項目に持つ辞書で, 任意の名前の引数 (`**kwargs`) を受け取るクラスには範囲を渡せない. `unregister_optimizer(name)` で外せる. ダッシュボードから起動する場合は, ダッシュボードのプロセスでは登録されないので, config の名前は `custom_optim.Lion` のようにモジュールのパスで書く.

```python
from ml_found.training import register_optimizer

register_optimizer(
    Lion, description="符号だけを使う Optimizer",
    param_notes={"lr": {"description": "学習率", "minimum": 0.0}}
)
```

```python
from ml_found.training import describe_optimizers

adam = next(spec for spec in describe_optimizers() if spec["name"] == "Adam")
print([(param["name"], param["default"]) for param in adam["params"]])
```

## 7. 学習スクリプトと step の決まり

上書きは experiment の後に重なり, experiment と同じく default にある項目だけを書けて, `diff.yaml` にも残る. 上書きでクラス名を変えると, default と experiment に書いた前のクラスの引数は捨てられる. 値は YAML として読むので, `scheduler.name=null` で使わない設定にできる. 項目の削除 (`~`) や追加 (`+`) の書き方には対応していない.
独自の引数 (`--seed` など) を持つ学習スクリプトは, 自分の parser に `add_config_arguments(parser)` で `--experiment` と上書きを加え, 読み込んだ引数を `load_config_from_args(project_dir, args)` に渡す. パスや上書きを自分で決める場合は `load_config(default_path, experiment_path, overrides)` を使う.

`start_experiment` が返す `Experiment` は, 学習に使う config (`cfg`), プロジェクトのディレクトリ (`project_dir`), 実験の出力先 (`run_dir`. `<output_dir>/<name>/<日付>/<時刻>/` の `Path`), ロガー (`logger`), デバイス (`device`), モデルの定義の写しの場所 (`model_spec_path`. 写した場合は `run_dir / "model_spec.yaml"`), 写した元の定義のファイル (`model_spec_source`. プロジェクトの中の絶対パス) を持つ (`model_spec_path` と `model_spec_source` は, 写さなかった場合と `model.spec` が無い場合は `None`). モデルを作るのに使ったファイルを残す場合は `run_dir` に写す. モデルの定義 (ML-ModelControl) は, config の `model.spec` に場所を書けば `start_experiment` が写す (下の「モデルの定義の写し」).

**モデルの定義の写し**: config の `model.spec` (相対パスはプロジェクトから辿る) にファイルがあれば, `start_experiment` が出力先を作る前に読み (学習を始めるときの中身), 出力先を作った後に中身を変えずに `model_spec.yaml` として写し, 場所と中身の sha256 を `source.yaml` の `model_spec` に書く (出力先だけで何で学習したかが分かり, 同じモデルで起動し直せるように). ML-Found は定義の中身を読まない (ml_modelcontrol に頼らない) ので, 定義の誤りはモデルを組み立てるときに分かる.

- `model.spec` が無い, 空 (`null` か空白だけの文字) 場合は何もしない (`source.yaml` の `model_spec` は null). 文字でない値 (`123` など) も写さずに null とし, そのことを `train.log` に 1 行書く. 解決できない参照 (`${nope}`, 無い環境変数の `${oc.env:NAME}` など) も, 定義を読む段 (`read_model_spec`) では null とするが, `start_experiment` は先に参照の確かめ (「6. config の決まり」) で止まる. `model` はプロジェクトの項目なので, ML-Found は `model.spec` の他は読まない
- 写せない場合は写さずに理由を `status` に残して `train.log` に書き, 学習は続ける (例外で止めない. 定義を組む前に仮のモデルで学習するプロジェクトがあり, 写しは学習の補いなので). ファイルが無い場合と, `..` の直前が無いかディレクトリでない場合 (`nosuch/../m.yaml`. OS では開けない) は `missing`, 場所として読めない書き方 (存在しないユーザーの `~`, ヌル文字など) は `invalid`, 写しを書けなかった場合 (出力先に書けない, 同じ名前のディレクトリがあるなど) は `write_failed`
- プロジェクトの外を指す場所 (`outside`), 場所か途中のディレクトリがシンボリックリンクのもの (`link`), 普通のファイルでないもの (`not_file`. 末尾が区切りの `m.yaml/` と `m.yaml/.` も), 読めないもの (`unreadable`), 16 MiB を超えるもの (`too_large`. 重みのファイルなどを誤って指した場合に丸ごと読まないため) は写さない (実験に思わぬファイルを写さないため). リンクには Windows のジャンクションも含む. リンクの後の `..` (`extlink/../m.yaml` など) も `link` とし, そのときだけ `path` は正規化せずに書いた文字のまま残す (リンクと関係の無い `..` は正規化する) (POSIX では, `..` を消して正規化した場所と OS が開く場所が違うため). 開くときは最後の要素のリンクを辿らず (`O_NOFOLLOW`), 開いたものがプロジェクトの中の同じファイルかを確かめてから読む (確かめた後に途中のディレクトリを入れ替えられた場合も外を写さないため). プロジェクトに至るまでのリンク (macOS の `/tmp` など) は問わず, リンクを通した場所から書いた絶対パスもプロジェクトの中として扱う
- `trainer.resume_from` で再開する学習でも写す (再開でも今の定義から組み立て, 再開元の重みを strict に読むので, 学習が動いた時点で今の定義がその実験の定義). 再開かどうかは `config.yaml` の `trainer.resume_from` で分かる. 同じ `Trainer` で `fit` を呼び直す場合は同じ実験なので, 最初に写したものが残る
- 写したファイルは `config.yaml` と同じ権限 (umask に従う) で作る. 学習のスクリプトが学習の後に同じ名前で写し直しても上書きできる (そのときの中身は `sha256` と合わないことがある)

**モデルは定義の写しから組み立てる (推奨)**: 学習のスクリプトは, `model.spec` の場所でなく `Experiment` の `model_spec_path` (写した `model_spec.yaml`) からモデルを組み立てる. 写しの中身は `source.yaml` の `sha256` と同じなので, 記録した定義と学習に使った定義がそろう. `sha256` は `start_experiment` が読んだ時点の中身なので, `model.spec` の場所 (か別の場所) から組み立てると, 別の時点 (`start_experiment` の前や後) に書き換えた定義を使い, 記録と一致しないことがある. `None` の場合 (写さなかった場合. 理由は `source.yaml` の `status` と `train.log`) は, 記録に `sha256` が無いので, プロジェクトの定義から組み立てても記録との食い違いは起きない.

- 定義の中の相対の参照 (外部の重みのファイルなど) は, 写しのある `run_dir` でなく元の定義の場所から辿る. 組み立てる関数には, 相対の参照の起点として `model_spec_source.parent` を渡す (渡さずに写しのディレクトリから辿ると, 参照したファイルが見つからない). ML-Found が写すのは定義のファイルだけで, 定義が参照するファイルは写さない
- 定義の誤りで空の出力先が残らないように, 元の定義の確かめ (組み立てずに形を確かめる関数など) は `start_experiment` の前にする. 組み立ては写しからする

```python
# 元の定義を確かめてから実験を始める (誤りなら出力先を作る前に止まる)
check_spec(project_dir / cfg.model.spec)
experiment = start_experiment(cfg, diff, str(project_dir))

# 写しがあれば写しから (相対の参照は元の定義の場所から辿る),
# 無ければ (仮のモデルなど) プロジェクトで組み立てる
if experiment.model_spec_path is not None:
    model = build_model(
        experiment.model_spec_path,
        base_dir=experiment.model_spec_source.parent
    )
else:
    model = build_fallback_model(cfg)
```

`check_spec`, `build_model`, `build_fallback_model` はプロジェクトで用意するもの. 例えば ML-ModelControl では, 定義の確かめに `infer`, 組み立てに `build_model(path, base_dir=...)` を使える (`base_dir` を省くと定義のファイルのディレクトリから辿る).

`step(model, batch, phase)` の決まりは以下の通り.

- `batch` は ML-Found がデバイスに移してから渡す
- `phase` は学習時に `"train"`, 検証時に `"valid"` になる (`trainer.eval_train: true` の場合は, 学習用データの評価し直しで `"train_eval"`)
- 戻り値は辞書で, `loss` は必須. その他の値は `train/acc` や `valid/acc` のように指標として記録され, 学習曲線も描かれる
- 指標は 1 つの値にし, 名前には `/` を使わない (学習曲線の画像のファイル名に使うため)
- 指標はバッチのサンプル数で重み付けして平均する. サンプル数は自動で推定するが, 戻り値に `batch_size` を入れると, そちらを使う. 推定できない場合 (テンソルを含まないバッチなど) は 1 として扱い, `train.log` に警告を 1 回出す
- 重み付き平均してよいのは, 損失, MAE, 正解率のように「サンプルごとの値の平均」になる指標だけ. RMSE, AUC, F1 のようにエポック全体から求める指標は, 下の `on_epoch_end` で求める
- 平均せずに記録だけしたい値 (KL の重みのようなスケジュールの値など) は, 戻り値の `extra` に「名前: 値」の辞書で入れる. `train/` や `valid/` を付けずに, エポックの最後に返した値を `history.csv` の列に記録する (学習率の `lr` と同じ扱い. `epoch`, `lr` と, `optimizers` の場合の `lr_<名前>` は使えない). 検証でも返した場合は検証の値が残る (`trainer.eval_train` の評価し直しで返した値は記録しない). 名前は空でない文字列で, `/`, カンマ, `=`, `epoch`, `lr` と, 指標と同じ名前 (`loss` など, 学習曲線で混ざるため) は使えない. 値は数か 1 つの値のテンソル. `extra` という名前の指標は返せない

## 8. フルバッチ学習

グラフ全体のように 1 度に全データを流す場合は, 要素が 1 つのリストを渡す.
学習用と検証用で同じデータを使い, `phase` でマスクを使い分ける.

```python
def step(model: nn.Module, batch: dict, phase: str) -> dict:
    mask = batch[f"{phase}_mask"]
    ...


history = trainer.fit([graph], [graph])
```

`trainer.eval_train: true` と使う場合は, `phase` に `train_eval` も渡るので, `train` のマスクに読み替える (`"train" if phase == "train_eval" else phase`).

## 9. 継承して細かく変える

`Trainer` を継承すると, 次のメソッドを上書きできる.

| メソッド | 呼ばれるタイミング | 用途 |
| --- | --- | --- |
| `step(batch, phase)` | バッチごと | `step_fn` を渡す代わりに書く |
| `on_epoch_start(phase)` | エポックの学習または検証の前 | 予測を集める入れ物を用意する |
| `on_epoch_end(phase)` | エポックの学習または検証の後 | AUC など, バッチの平均では求められない指標を辞書で返す. `train/` や `valid/` (評価し直しでは `train_eval/`) を付けて記録される. `valid` では検証するエポックだけ呼ばれるので, 重い評価は `phase == "valid"` のときに書く (「検証の間隔」) |

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
| `update_parameters(loss)` | 更新の手順を変える. `super().update_parameters(loss)` の後に自分で持つ平均の重みを更新する, など (`trainer.ema.decay` の EMA は `super().update_parameters(loss)` の中で更新される). 学習のバッチごとに呼ばれ, 勾配を貯める設定 (`trainer.accumulate_grad_batches`) では更新した回だけ `self.global_step` が増える (自分の処理を更新の回だけにする場合は, 呼ぶ前と後の `global_step` を比べる). バッチ数を持たない loader の端数の更新はエポックの終わりに `update_parameters` の外で行う |
| `extra_state_dict()` | チェックポイントに加える状態を辞書で返す (EMA の重み, ML-Found の外で作った Optimizer など). `load_extra_state_dict(state)` も合わせて上書きすると, 再開時に戻る |
| `eval_model()` | 検証, ベストモデルの判定, `final_model.pt` に使うモデルを返す (既定は `self.model`. `trainer.ema.decay` を使う場合は EMA のモデル). 自分で持つ平均の重みで検証する場合はそのモデルを返す |

EMA は `trainer.ema.decay` の設定だけで使える (「重みの指数移動平均 (EMA)」). 下は, 設定に無い平均の取り方 (更新の間隔を空ける, 減衰率を学習の進み具合で変えるなど) を自分で書く場合の決まりと例.

`eval_model` を上書きした場合の動作.

- 検証と学習用データの評価し直しでは, `step_fn` にこのモデルが渡され, 推論モードになる. `step` を上書きしている場合は, 学習以外 (`phase` が `valid` か `train_eval`) で自分で `self.eval_model()` を使う
- ベストモデルの重みはこのモデルから取り, 学習後に `self.model` に戻す. ベストモデルに戻さない設定 (`restore_best: false` か `select_best: false`) では, 最後のこのモデルの重みを `self.model` に読み込んで `final_model.pt` に保存する. どちらも `self.model` に読み込むので, `self.model` と同じ名前と形の重みを持つ必要がある (違う場合は学習を始める前にエラー)
- `last.pt` と `best.pt` の `model` は学習している `self.model` の重みのまま (再開に使うため). ベストモデルの重みは `best.pt` でも省かずに `checkpoint` の `best_model` に入れる. 重みだけの `best.pt` (`best_weights_only: true`) の `model` は, このモデルのベストモデルの重み
- このモデルの状態 (EMA の重みなど) は自分で `extra_state_dict` と `load_extra_state_dict` で保存して戻す. 戻さないと, 再開した学習ではこのモデルが作り直した状態から始まる
- 再開で `last.pt` にベストモデルの重みが無い場合 (`last_with_best: false`) は, 再開元のエポックがベストでも `best.pt` から戻す
- エポックの途中で止めた場合, このモデルの状態は `extra_state_dict` で保存していれば前のエポックのものに戻る. 保存していない場合は戻せないので, `final_model.pt` は途中までの重み (`final_weights` は `partial`) になり, 警告が出る
- 検証しない場合は `train/loss` (学習している `self.model` で計算した値) でベストモデルを判定し, 記録するのはこのモデルの重みになる. EMA の重みの良し悪しで選ぶには検証用のデータを渡す
- `torch.optim.swa_utils.AveragedModel` は既定で BatchNorm の統計量を平均せず, 学習しているモデルの値を写す (`use_buffers=True` で平均する)

```python
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from ml_found import Experiment, Trainer


class EMATrainer(Trainer):
    def __init__(self, model: nn.Module, experiment: Experiment, step_fn, decay: float):
        super(EMATrainer, self).__init__(model, experiment, step_fn)
        self.ema = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(decay))

    def update_parameters(self, loss: torch.Tensor) -> None:
        super(EMATrainer, self).update_parameters(loss)
        self.ema.update_parameters(self.model)

    def extra_state_dict(self) -> dict:
        return {"ema": self.ema.state_dict()}

    def load_extra_state_dict(self, state: dict) -> None:
        # extra の無いチェックポイントからの再開では空の辞書が渡される
        if "ema" in state:
            self.ema.load_state_dict(state["ema"])

    def eval_model(self) -> nn.Module:
        return self.ema.module
```

`on_epoch_end` は勾配の計算を止める範囲の外で呼ばれる (検証の後でも勾配の計算が有効). 中で推論する場合は `torch.no_grad()` を書く.
指標を返さない場合 (画像を保存するだけ, など) は空の辞書を返す. `step` の指標と同じ名前は使えない (エラー).
Optimizer や Scheduler を自分で作る場合は `Trainer(model, experiment, step, optimizer, scheduler)` のように渡す.
渡した Scheduler にも, config で作った場合と同じく `scheduler.interval` と ReduceLROnPlateau の確認が行われる.
自分で作った Scheduler は `config.yaml` に残らない. クラス名と主な引数は `run.json` の `optimizers` (`scheduler` と `scheduler_params`) に, クラス名と状態は `train.log` に残る. `scheduler_params` の形は [FORMAT.md](FORMAT.md) の「run.json」の節にあり, config から作った Scheduler も同じ形で書く.

## 10. Scheduler の総ステップ数と warmup

`OneCycleLR` の総ステップ数 (`total_steps`, または `epochs` と `steps_per_epoch`) は, 書かなければ `fit` が学習を始めるときに補う. `steps_per_epoch` は `scheduler.interval: step` なら学習用データのバッチ数 (`len(train_loader)`), `epoch` なら 1 (1 エポックに 1 回進めるため), `epochs` は `trainer.epochs` にする. 補った値は `config.yaml` と `train.log` に残る. そのため config から作る `OneCycleLR` は `fit` を呼ぶまで作られない (`trainer.scheduler` は `None`). `scheduler.interval: step` で `len()` を持たない `train_loader` では補えないのでエラーになり, `total_steps` か `steps_per_epoch` を書く. `OneCycleLR` は, 補ったかどうかに関係なく, 総ステップ数が `trainer.epochs` とバッチ数から求めた予定より少ないと学習を始める前にエラーに, 多いと警告にする (学習率が下がりきらない). 再開した場合は再開元の総ステップ数が使われるので, `trainer.epochs` やバッチ数を変えて再開した場合や, 補った値の入った `config.yaml` を default にして条件を変えた場合も, この確認で分かる. エポック数を増やして再開する場合は, `trainer.resume_scheduler: config` にすると, 補い直した総ステップ数で作り直して続きを学習する (下の「学習の再開」を参照).

学習全体で決まる他の Scheduler の引数 (`CosineAnnealingLR` の `T_max`, `LinearLR`, `ConstantLR`, `PolynomialLR` の `total_iters`) は, 値に `auto` と書くと, `fit` が学習を始めるときに学習全体のステップ数で補う. `scheduler.interval: step` ならエポック数 × 学習用データのバッチ数, `epoch` ならエポック数. 補った値は `config.yaml` と `train.log` に残る. `auto` と書けるのはこれらのクラス (と継承したクラス) の引数だけで, 他の引数 (範囲を持たない `eta_min` などや, Optimizer の引数も) に書くと出力先を作る前にエラーになる. warmup のように学習の一部の長さを表す値には使えない (`auto` は学習全体の長さ). `OneCycleLR` と同じく, Scheduler は `fit` を呼ぶまで作られない (`trainer.scheduler` は `None`). エポック数などを変えて再開すると補う値が再開元と変わるので, 既定の `trainer.resume_scheduler: checkpoint` ではエラーになり, `config` にすると補い直した値で作り直す. 補った値の入った `config.yaml` を default にすると, `auto` ではなく数値になるので, エポック数を変える場合は `auto` に書き直す.

学習の始めに学習率を少しずつ上げる warmup は, `scheduler.warmup_steps` (回数, `interval` の単位) か `scheduler.warmup_ratio` (学習全体のステップ数に対する割合) で書く. 学習率は最初の値の `scheduler.warmup_start_factor` (既定 0.01) 倍から線形に上げ, その後は `scheduler.name` の Scheduler に続ける (`SequentialLR` で組み合わせる). `auto` と書いた引数は, warmup の後の残りのステップ数で補う. `warmup_ratio` は `fit` が学習を始めるときに回数に直し, `config.yaml` の `warmup_steps` に残す. `scheduler.name` が無い場合は warmup だけで, その後の学習率は一定. ReduceLROnPlateau, OneCycleLR, CyclicLR とは組み合わせられない (エラー). warmup の他に複数の Scheduler を組み合わせる設定 (config に `SequentialLR` を書く形) は, warmup と 1 つの Scheduler で実用の大半に足りるので対応しない. `warmup_ratio` を使う場合も, Scheduler は `fit` を呼ぶまで作られない. warmup が学習全体のステップ数以上の場合と, 割合が小さく 0 回になった場合は警告する. 再開では, warmup の有無を変えた場合 (Scheduler の組み立てが変わる) と, 割合から補った回数が再開元と変わった場合は, 既定ではエラーになり, `trainer.resume_scheduler: config` にすると作り直す.

```yaml
scheduler:
  name: CosineAnnealingLR
  interval: step
  params:
    T_max: auto        # warmup の後の残りのステップ数
  warmup_ratio: 0.1    # 学習全体の 1 割で学習率を上げる
```
自分で値を決める場合は, `load_config` の後, `start_experiment` の前に config に書き込むと `config.yaml` にも残る.

```python
cfg, diff = load_project_config(str(project_dir))
train_loader = make_loader(cfg)
cfg.scheduler.params.steps_per_epoch = len(train_loader)
experiment = start_experiment(cfg, diff, str(project_dir))
```

## 11. 更新の手順

- 勾配法の Optimizer は `zero_grad`, `loss.backward()`, 勾配のクリッピング, `optimizer.step()` の順に更新する. 勾配を貯める設定 (`trainer.accumulate_grad_batches`) では, 決めた数のバッチの勾配を貯めてから 1 回クリップして更新する (「大きな学習済みのモデルの微調整」)
- LBFGS のように `step` に closure が必要な Optimizer は自動で見分け, 同じバッチで損失と勾配を計算し直す closure を渡す. 見分ける順は, クラスの属性 `requires_closure` (自作の Optimizer が `True` か `False` で宣言), torch の Optimizer の仕様 (LBFGS. `describe_optimizers()` の `requires_closure` と同じ), `step` の `closure` が既定値の無い引数かどうか. 自作の Optimizer で `closure` に既定値があるのに実際には必要な場合は, クラスの本体に `requires_closure = True` と書く (property や `__init__` の中で付けた属性は見ない). closure は `step` を何度も呼ぶので, `step` で予測を貯めるなどの処理は, closure の中では `self.in_closure` が `True` になることを使って飛ばす
- Nelder-Mead 法のような勾配を使わない Optimizer は, クラスの本体に `uses_gradients = False` と書く. ML-Found は学習でも勾配を計算せず (逆伝播しない), 今のパラメータの損失を求める closure を `step` に渡す (`step` は勾配の記録を止めて呼ぶので, パラメータを直接書き換えてよい). `trainer.grad_clip_norm` と `log_grad_norm` とは一緒に使えない (config で指定した場合は出力先を作る前にエラー). `trainer.valid_grad` は検証だけに効く. `manual_optimization: true` の場合はそちらが優先され, 学習でも勾配を記録して `step` の中で自分で更新する. 登録簿の仕様にも `uses_gradients` を返す
- 複数の Optimizer を使う場合は, 下の「複数の Optimizer」のように config の `optimizers` に名前付きで書く. 更新の順番が込み入る場合 (識別器を更新した後の値で生成器の損失を求めるなど) は, `trainer.manual_optimization: true` にして `step` の中で更新する. ML-Found は逆伝播, 勾配のクリッピング, 更新を行わず (`grad_clip_norm` と `log_grad_norm` と同時に使うとエラー), 更新回数と Scheduler (`interval` に応じて step ごとかエポックごと) だけを進める. `step` は勾配の流れない損失 (`loss.detach()`) を返してよい
- パラメータを 1 つも持たないモデル (`model.parameters()` が空. 学習用の正常の行の中心と精度行列をバッファに持ち, 新しい行の近さから度合いを出す距離に基づく異常の度合いなど) は, 勾配で学習しないので, `Trainer` は Optimizer と Scheduler を作らない. config の `optimizer`, `scheduler`, `optimizers` は書いてあっても誤りにせず, 値は他の実験と同じに `start_experiment` で確かめる (クラス名の誤りなどは出力先を作る前に止まる) が, 学習には使わない. 使わないことを `train.log` に 1 行書く. 自分で作った Optimizer と Scheduler は渡せない (`Trainer` を作るときにエラー). バッファは `fit` の前にプロジェクトで求めておく
    - **fit**: 学習のフェーズ (`train`) を回さず (更新は 0 回, `global_step` は 0 のまま), 判定するエポックごとに検証 (`step`, `on_epoch_start`, `on_epoch_end` に `phase` は `valid`) と, `trainer.eval_train: true` の場合は評価し直しだけを行う. 検証用のデータが要る. 監視する指標は `valid/` (`eval_train: true` の場合は `train_eval/` も) で始まるものにする (`train/` は記録しないため). 検証用のデータを渡さない場合と, それ以外の監視する指標を書いた場合は, 学習を始める前に `fit` がエラーで止まり, `run.json` は `failed`
    - **エポック数**: 毎エポック同じ検証になるので `trainer.epochs: 1` にする. 1 でない場合も誤りにせず毎エポック同じ検証をし, `train.log` に警告を 1 行書く
    - **残すもの**: `run.json` の状態, `history.csv`, ベストモデルの判定, `best.pt`, `last.pt`, `final_model.pt` (バッファを含む), 評価は他の実験と同じ形. 違うのは, `history.csv` に `train/` と `lr` の列が無いことと, `run.json` の `optimizers` が空のリストであること. 再開は, パラメータの無いモデルの `last.pt` からパラメータの無いモデルでだけできる (逆も同じ)
    - 凍結したパラメータだけを持つモデル (全てが `requires_grad` 偽) は, パラメータを持つので Optimizer を作り, 学習のフェーズを回す (途中で凍結を解ける)

## 12. 複数の Optimizer

GAN (生成器と識別器) や強化学習の Actor-Critic のように, モデルの部分ごとに別の損失と別の Optimizer で更新する場合は, `optimizer` と `scheduler` の代わりに `optimizers` に名前付きで書く.

```yaml
# configs/default/train.yaml
optimizers:
  discriminator:
    name: Adam
    params: {lr: 0.0002, betas: [0.5, 0.999]}
    modules: [discriminator]      # 受け持つ部分 (model の属性の名前, . で区切って入れ子も可)
  generator:
    name: Adam
    params: {lr: 0.0002, betas: [0.5, 0.999]}
    modules: [generator]
    scheduler:                    # 書き方は一番上の scheduler と同じ (省略可)
      name: StepLR
      params: {step_size: 10}
```

`step` は `loss` に, Optimizer の名前ごとの損失の辞書を返す (`Trainer` に渡す関数でも, 継承したクラスの `step` でもよい).

```python
bce = nn.BCEWithLogitsLoss()


def step(model, batch, phase):
    x, _ = batch
    fake = model.generator(torch.randn(len(x), 64, device=x.device))
    real_logits = model.discriminator(x)
    fake_logits = model.discriminator(fake)
    d_loss = (
        bce(real_logits, torch.ones_like(real_logits))
        + bce(fake_logits, torch.zeros_like(fake_logits))
    )
    g_loss = bce(fake_logits, torch.ones_like(fake_logits))
    return {"loss": {"discriminator": d_loss, "generator": g_loss}}
```

- **更新**: ML-Found は, 損失ごとに受け持つ部分のパラメータだけの勾配を求め (`torch.autograd.grad`), Optimizer ごとに勾配をクリップしてから, config に書いた順に更新する. 全ての損失の勾配は更新の前の同じパラメータで求める (同時の更新). 例えば生成器の損失の勾配は識別器のパラメータに入らないので, 上の例の `d_loss` に `fake.detach()` を使わなくてもよい. 損失が届かないパラメータの勾配は `None` になり, その更新では Optimizer が飛ばす (weight decay や momentum も進まない). クリップは Optimizer ごとのパラメータを, それぞれ `trainer.grad_clip_norm` に収める
- **受け持つ部分**: `modules` は `model.get_submodule` で探す名前のリスト (空のリストとモデル全体の `''` は不可). 中のパラメータを受け持つ (凍結したものも受け持ち, 学習の途中で凍結を解くと更新する. 途中で凍結したものは勾配を求めない). 学習するパラメータ (`requires_grad`) は, どれかの Optimizer が受け持つ. どれにも入らない学習するパラメータ, 2 つ以上に入るパラメータ, 名前がモデルに無いもの, パラメータを 1 つも受け持たない Optimizer は, `Trainer` を作るときにエラーになる (出力先を作った後)
- **記録**: 名前ごとの損失 (`train/loss_<名前>`, 検証では `valid/loss_<名前>`), その合計 (`train/loss`, `valid/loss`), 名前ごとの学習率 (`lr_<名前>`, `lr_steps.csv` の列も同じ), `log_grad_norm` の場合は名前ごとのクリップ前の勾配のノルム (`train/grad_norm_<名前>`). 合計の `loss` は既定の監視する指標になるが, GAN のように損失の合計が良し悪しを表さない場合は `trainer.monitor` を書くか `trainer.select_best: false` にする. `step` は `loss_<名前>` と `grad_norm_<名前>` を, `extra` は `lr_<名前>` を返せない. 損失は名前ごとに `torch.Tensor` で, 勾配を持つものにする
- **Scheduler**: `interval` は, Scheduler を付けた全ての Optimizer で同じにする. `auto`, warmup, `OneCycleLR` の補う引数は, Scheduler ごとに補う (補う前と, Scheduler を付けない名前は `self.scheduler["名前"]` が `None`)
- **保存と再開**: Optimizer と Scheduler の状態は名前ごとにチェックポイントに保存し, 再開できる. 名前と, 名前ごとの Scheduler の組み立て (有無, クラス, warmup の有無) は再開元と同じにする. 1 つの Optimizer の学習の `last.pt` から名前付きで再開することと, その逆はできない. `auto` や `warmup_ratio` で補った値が再開元と違う場合 (エポック数を変えた場合など) はエラー. それ以外の Scheduler の引数の違いは確かめず, 再開元の状態を使う (`trainer.resume_scheduler: config` は使えない)
- **config の書き方**: 名前は `.`, `/`, 空白を含まない文字列にする. `optimizers` は default の config に書き, experiment と上書きでは既にある名前の項目だけを変えられる (名前の追加と削除はできない). `optimizer` と `optimizers` は一方だけ書く (default に両方書いた場合と, `optimizers` を使う default に experiment で `optimizer` を書いた場合はエラー). 一番上の `scheduler` は既定値のまま使わない (変えるとエラー). 使わない `optimizer` の既定値は `config.yaml` に残る. 名前ごとの `name` と `params` を省略すると `Adam` と `{lr: 0.001}` になる
- **ログ**: `train.log` に `Optimizer:` (名前ごとの Optimizer), `Scheduler: discriminator=なし, generator=StepLR`, `Optimizer <名前> が受け持つ部分: [...] (学習するパラメータ数: ...)`, 名前ごとの Scheduler の状態を書く
- **継承したクラス**: `self.optimizer["generator"]` で名前ごとの Optimizer を, `self.scheduler["generator"]` で Scheduler を取り出せる (`self.optimizer` と `self.scheduler` は名前ごとのものをまとめたもの). `update_parameters` は合計の損失を受け取り, 名前ごとの損失は `self.named_losses` にある
- **使えないもの**: closure が必要な Optimizer (LBFGS など), 勾配を使わない Optimizer (`manual_optimization` でも不可), `Trainer` に渡す自分で作った Optimizer と Scheduler. 確かめるのは出力先を作る前 (`modules` の書き方, 一番上の `scheduler`, `interval` の違い, `resume_scheduler: config` も同じ)
- **`trainer.manual_optimization: true`**: ML-Found は更新を行わない. `step` は名前ごとの損失の辞書 (`loss_<名前>` と合計を記録) か, 1 つの損失を返す. 更新の順番が込み入る場合 (識別器を更新した後の値で生成器の損失を求めるなど) に使う

## 13. 検証の設定

- PINN のように検証でも入力に対する微分を使う場合は, `trainer.valid_grad: true` にする
- 拡散モデルのように損失に乱数を含む場合は, `trainer.valid_seed` を決めると毎エポック同じ乱数で検証し, ベストモデルの判定が乱数で揺れない. 検証の後は学習の乱数の状態に戻すので, 学習には影響しない. 固定するのは Python, NumPy, torch の全体の乱数なので, 自分の `generator` を渡した DataLoader や, ワーカーを使い回す DataLoader のワーカーの乱数には届かない. `register_rng` で登録した乱数 (下の「外部ライブラリの乱数」) は固定の対象になる
- Dropout などで学習時と推論時の振る舞いが違うモデルでは, `train/` の指標は学習中の振る舞いの値になる. `trainer.eval_train: true` にすると, 各エポックの学習の後に学習用データを推論モード (勾配の計算なし, `valid_grad` と `valid_seed` は検証と同じに効く) で評価し直し, `train_eval/` を付けて記録する (`train_eval/loss` など). 学習用と検証用の指標を同じ条件で比べられる. 評価し直しの前後で乱数の状態 (全体の乱数, `register_rng` で登録した乱数, DataLoader と sampler に渡した `generator`) を戻すので, 並べ替える DataLoader でも学習そのものは変わらない. ただし `persistent_workers=True` の DataLoader では, 使い回すワーカーの中の乱数 (データ拡張など) が進むので学習の結果が変わる (学習を始めるときに警告が出る). `step` が評価し直しで返した `extra` は記録しない (学習中の値が残る). DataLoader の中のデータ拡張 (`Dataset` の変換など) は評価し直しにもかかるので, 拡張をかけない値が欲しい場合は, 拡張の無い学習用データの DataLoader で別に評価する. 時間は学習用データ 1 周分の推論だけ増える. `step` と `on_epoch_start`, `on_epoch_end` には `phase` に `train_eval` が渡るので, `phase == "train"` で学習かどうかを見分ける書き方にする. 監視する指標にも使える (`trainer.monitor: train_eval/loss`)

## 14. 検証の間隔

FID, Chamfer 距離, 検索の再現率のように重い評価を毎エポック行わない場合は, `trainer.valid_every_n_epochs` で検証の間隔を空ける. 検証するのは, エポックの番号 (1 から数えた通しの番号) が間隔で割り切れるエポックと, 予定の最後のエポック (`trainer.epochs`. `fit` を呼び直した場合はその `fit` の最後) だけ. 例えば 7 エポックで間隔が 3 なら, エポック 3, 6, 7 を検証する. 既定の 1 では毎エポック検証する.

```yaml
trainer:
  epochs: 100
  valid_every_n_epochs: 5
  monitor: valid/fid
  early_stopping:
    use: true
    patience: 3   # 検証 3 回 (15 エポック) 改善しなければ止める
```

- 検証するエポック (判定するエポック) だけ, 検証, `eval_train` の評価し直し, 監視する指標の確認 (無いか nan や inf ならエラー), ベストモデルの判定, Early Stopping, ReduceLROnPlateau を行う. ベストモデルは判定したエポックの中から選ぶ. 検証用のデータを渡さない場合 (`train/loss` を監視する場合) も同じエポックだけ判定する
- Early Stopping の `patience` と ReduceLROnPlateau の `patience` は, エポック数ではなく判定の回数で数える. 間隔が 5 で `patience: 3` なら, 15 エポック改善しなければ止まる
- 判定しないエポックは, 学習の指標 (`train/`), 学習率, `extra` だけを記録する. `history.csv` の `valid/` と `train_eval/` の列は空になり, `History.get` では `None` になる. 学習曲線では記録のあるエポックだけを点で描く
- `step` が `extra` で返した値は, 判定するエポックでは検証の値, 判定しないエポックでは学習の値が残る. 学習と検証の両方で返す値は `history.csv` で学習と検証の値が入れ替わって並ぶので, どちらか一方のフェーズだけで返すか, フェーズごとに別の名前にする
- `extra` の名前が指標と同じ場合のエラーも, その指標が検証だけのもの (`valid/` だけにある名前) なら, 最初に判定するエポックまで出ない
- ReduceLROnPlateau 以外の Scheduler (`scheduler.interval: epoch` の StepLR など) は, 判定しないエポックも毎エポック進める
- 重い評価は `on_epoch_end` を継承して `phase == "valid"` のときに書く. `on_epoch_start` と `on_epoch_end` はフェーズごとに呼ばれるので, 検証しないエポックでは `valid` で呼ばれず, 重い評価も行われない. 評価の間は `progress.json` の段階が `epoch_end` になり, 画面に評価中と出る
- 判定しないエポックで `STOP` や `add_callback` の関数で止めた場合は, 検証せずに止める. ベストモデルは最後に判定したエポックのまま (1 回も判定しないうちに止めた場合はベストモデルが無く, `final_model.pt` は最後のエポックの重み). 止めたエポックの `last.pt` は保存する
- 監視する指標が無いか nan や inf のエラーは, 最初に判定するエポック (間隔が 5 ならエポック 5) まで出ない. 名前の誤りに早く気付くには, 初めは `trainer.epochs` を小さくして試す
- エポックの番号は通しなので, 再開と `fit` の呼び直しでも同じエポックを判定する. 再開で間隔を変えると警告が出る (待つ回数は判定の回数のまま続く). ただし予定の最後のエポックは必ず判定するので, 途中のエポック数で止めた学習 (例えば `trainer.epochs: 4` で学習して 7 まで再開) では, その最後のエポック (4) も判定し, 通しで学習した場合より判定が 1 回多くなる (間隔で割り切れるエポック数で区切ると同じになる)

### 生成の画像の FID

生成した画像と本物の画像の近さは, `FrechetInceptionDistance` で FID (Fréchet Inception Distance) として数え, `on_epoch_end` の指標として記録する. 学習済みの Inception (pytorch-fid) で画像ごとの特徴 (既定は 2048 次元) を取り出し, 本物と生成の特徴の平均と共分散から 2 つの正規分布のフレシェ距離を求める. 小さいほど本物に近い. 重い評価なので `trainer.valid_every_n_epochs` で間隔を空けて, 検証のときだけ数える. 監視する指標にすると, ベストモデルを FID で選べる (`trainer.monitor: valid/fid`, `trainer.mode: min`).

```python
import torch

from ml_found import FrechetInceptionDistance, Trainer


class GenerativeTrainer(Trainer):
    def __init__(self, model, experiment, step_fn, real_loader):
        super(GenerativeTrainer, self).__init__(model, experiment, step_fn)
        # 追加の依存 fid が無ければ入れ方を示す MissingRequirementError
        self.fid = FrechetInceptionDistance(device=self.device)
        self.real_loader = real_loader

    def on_epoch_end(self, phase: str) -> dict:
        if phase != "valid":
            return {}
        with torch.no_grad():
            # 本物の特徴は 1 回だけ足して使い回す
            if not self.fid.has_real:
                for images, _ in self.real_loader:
                    self.fid.update_real(images)
            # 生成はエポックごとに作り直す
            self.fid.reset_fake()
            for _ in range(10):
                self.fid.update_fake(self.eval_model().sample(500))
        return {"fid": self.fid.compute()}
```

- 画像は `(N, 3, H, W)` か `(N, 1, H, W)` (白黒は 3 つのチャンネルに写す) のテンソルで, 値は `[0, 1]`. 範囲の外か nan を含む画像はエラーになる (`[-1, 1]` で生成するモデルは `(x + 1) / 2` にして渡す). 大きさは Inception が 299 x 299 に直すので問わない. 特徴は推論モードで勾配を計算せずに取り出す
- 特徴は 1 枚ずつ持たずに, 数と和と外積の和を float64 で足し合わせる (mps でも CPU で足す). バッチに分けて足しても, まとめて足した場合と同じ値になる. 共分散を求めるので, 本物と生成は 2 枚以上ずつ要る (足りなければエラー). FID は枚数で偏り, 特徴の次元 (2048) より少ないと共分散が退化するので, 比べる FID は同じ枚数で数え, 枚数は特徴の次元より多くする (よく使われるのは 1 万から 5 万枚)
- `reset_fake` は生成の特徴だけを捨て, 本物の特徴は残す. `reset` は両方を捨てる. 本物の特徴はチェックポイントに入れないので, 再開した学習では最初の検証で数え直す
- 生成に乱数を使う場合は `trainer.valid_seed` を決めると, 毎回同じ乱数で生成し, FID が乱数で揺れにくい
- `dims` (64, 192, 768, 2048) で Inception の特徴を取るブロックを選べる. 論文などの FID と比べるのは既定の 2048 だけ
- `feature_model` に自分の特徴のモデル (`(N, 3, H, W)` の `[0, 1]` の画像から `(N, D)` の特徴を返す `nn.Module`) を渡すと, pytorch-fid も重みも使わない (テストや, 画像でない領域の特徴の距離)
- `frechet_distance(mean1, cov1, mean2, cov2)` は, 平均と共分散からフレシェ距離だけを求める. 行列の平方根は対称な `cov1^(1/2) cov2 cov1^(1/2)` の固有値から求めるので, scipy に依存しない
- 学習済みの重み (`pt_inception-2015-12-05-6726825d.pth`, 約 91 MB) は, 特徴のモデルを渡さずに `FrechetInceptionDistance` を作るとき (pytorch-fid の Inception を作るとき) に pytorch-fid の GitHub のリリースから取り, PyTorch のキャッシュ (`TORCH_HOME`, 既定は `~/.cache/torch/hub/checkpoints`) に置く. 2 回目からはキャッシュを使う. ネットワークの無い計算機では, 一度ネットワークのある所で作ってキャッシュを写しておく
- 使用の条件: pytorch-fid は Apache License 2.0. 重みは TensorFlow の FID の公式の実装 (TTUR, Apache License 2.0) が使う Inception (TensorFlow が配った `inception-2015-12-05`) を PyTorch に移したもので, 重みに別の条件は書かれていない. ImageNet で学習した重みなので, 公開や商用に使う場合は ImageNet の利用条件も確かめる. 重みは ML-Found に含めず, 使う人の計算機に取る

### 異常の検知の上位 K 件の適合率と PR-AUC

異常の検知の評価は, `precision_at_k` (上位 K 件の適合率) と `average_precision` (PR-AUC. 平均の適合率) で数える. 入力は行ごとの異常の度合い (大きいほど異常) と, 異常か正常かの正解. 学習の段階では `on_epoch_end` の指標に, 学習の後は `save_evaluation` の表の値に使う. 戻り値は Python の `float` で, torch だけで数える (scikit-learn と scipy に依存しない). 推論の段階 (ML-Inference) は ML-Found に依存せず, 下の見本で同じ値になることを自分のテストで確かめる.

```python
from ml_found import average_precision, precision_at_k, save_evaluation

scores = errors_per_row     # 行ごとの再構成の誤差など (N,)
labels = is_anomaly         # 異常なら 1 か True (N,)

# on_epoch_end の中で, 監視する指標に
metrics = {
    "precision_at_10": precision_at_k(scores, labels, 10),
    "pr_auc": average_precision(scores, labels)
}

# 学習の後の評価の表に
save_evaluation(
    experiment, "anomaly", metrics, "test",
    better={"precision_at_10": "higher", "pr_auc": "higher"}
)
```

入力の決まりは次の通り.

- 度合いと正解は 1 次元の同じ長さのテンソル, NumPy の配列, 数の並びのどれか. 勾配の付いたテンソルと GPU のテンソルも受け, CPU の float64 に写して数える. 長さが 0 か, 形が違う場合は誤り
- 正解は真偽値か, 0 と 1 だけの数. 他の値 (2, -1, 0.5, nan) は誤り
- 度合いの nan は誤り (欠けた度合いを正常とも異常とも決めないため. 欠けた行は渡す前に除く). 無限大は使え, 同じ符号の無限大どうしは同点

PR-AUC は scikit-learn の `average_precision_score` と同じ形で数える. ただし無限大の度合いを受ける点だけは, scikit-learn (無限大を誤りにする) より広い.

- 度合いの高い順に並べ, 度合いの同じ行はまとめて 1 つのしきい値とする. しきい値 t ごとに, 度合いが t 以上の行の適合率 P(t) と再現率 R(t) を求める
- PR-AUC = Σ (R(t) - R(前のしきい値)) × P(t). 最初のしきい値の前の再現率は 0. 曲線を補間せず, 同点の行を並べる順にも依らない
- 正解の異常が 1 件も無い場合は 0 (`average_precision_score` と同じ値. 警告は出さない)

上位 K 件の適合率の決まりは次の通り.

- 度合いの高い順の上位 K 件のうち, 異常の行の割合. `k` は 1 以上の整数 (真偽値は誤り)
- K 件目の境に同点がある場合は, 境より上の行を全て数え, 境の同点の行からは残りの枠の数だけを, 同点の行の中の異常の割合で数える (同点を無作為に並べた場合の期待値). 同点の行を並べる順に依らず, 異常の数が小数になる場合がある
- K が行の数より多い場合は, K を行の数とする (全ての行の中の異常の割合)
- 正解の異常が 1 件も無い場合は 0

照合の見本 (手で数えた値. ML-Inference も同じ値を確かめる) は次の通り.

| 見本 | 度合い | 正解 | 値 |
| --- | --- | --- | --- |
| 同点を含む | `[0.9, 0.8, 0.8, 0.5, 0.3, 0.3]` | `[1, 0, 1, 0, 1, 0]` | PR-AUC = 13/18 (1/3 × 1 + 1/3 × 2/3 + 0 × 1/2 + 1/3 × 1/2). K = 1 は 1, K = 2 は 0.75 (境の同点の 0.8 の 2 行から 1 枠, 異常の割合 1/2), K = 3 は 2/3, K = 5 は 0.5 (境の同点の 0.3 の 2 行から 1 枠), K = 10 は 0.5 (K を 6 とする) |
| 上位が正常 | `[0.1, 0.4, 0.35, 0.8, 0.4]` | `[0, 1, 1, 0, 0]` | PR-AUC = 5/12 (0 × 0 + 1/2 × 1/3 + 1/2 × 1/2). K = 2 は 0.25 (0.8 の 1 行と, 境の同点の 0.4 の 2 行から 1 枠), K = 3 は 1/3 |
| 全て同点 | `[0.5, 0.5, 0.5, 0.5]` | `[1, 0, 0, 1]` | PR-AUC = 0.5. K = 1 は 0.5, K = 3 は 0.5 |
| 異常が無い | `[0.3, 0.2, 0.1]` | `[0, 0, 0]` | PR-AUC = 0. K = 2 は 0 |

### 効果を見積もるの Qini の曲線と係数, 上位 K 割の効果の差

効果を見積もる (行ごとの処置の効果 (uplift) を見積もる. 例はクーポンを配ったときに買う確率がどれだけ上がるか) の評価は, `qini_curve` (Qini の曲線), `qini_coefficient` (Qini の係数), `uplift_at_k` (上位 K 割の効果の差) で数える. 入力は行ごとの効果の見積もり (大きいほど処置する価値がある), 処置 (処置の群か対照の群か), 結果 (買ったかの 0 と 1 か, 買った額). 正解の効果は行ごとには見えないので, 見積もりの高い順に並べた上位の行の中で, 処置の群と対照の群の結果を比べて数える. 学習の段階では `on_epoch_end` の指標に, 学習の後は `save_evaluation` の表と曲線 (`kind="curve"`) に使う. 係数と効果の差は Python の `float` で, 曲線は列の名前から値の並びの辞書. 異常の検知の指標と同じく torch だけで数え (scikit-uplift と scikit-learn に依存しない), 推論の段階 (ML-Inference) は ML-Found に依存せず, 下の見本で同じ値になることを自分のテストで確かめる.

```python
from ml_found import (
    qini_coefficient, qini_curve, save_evaluation, uplift_at_k
)

scores = predicted_uplift   # 行ごとの効果の見積もり (N,)
treatment = is_treated      # 処置の群なら 1 か True (N,)
outcome = purchased         # 結果の 0 と 1 か額 (N,)

# on_epoch_end の中で, 監視する指標に (trainer.monitor: valid/qini)
metrics = {
    "qini": qini_coefficient(scores, treatment, outcome),
    "uplift_at_30": uplift_at_k(scores, treatment, outcome, 0.3)
}

# 学習の後の評価の表と曲線に
save_evaluation(
    experiment, "uplift", metrics, "test",
    better={"qini": "higher", "uplift_at_30": "higher"}
)
save_evaluation(
    experiment, "qini_curve", qini_curve(scores, treatment, outcome),
    "test", kind="curve", x="fraction"
)
```

入力の決まりは次の通り.

- 見積もり, 処置, 結果は 1 次元の同じ長さのテンソル, NumPy の配列, 数の並びのどれか. 勾配の付いたテンソルと GPU のテンソルも受け, CPU の float64 に写して数える. 長さが 0 か, 形が違う場合は誤り
- 処置は真偽値か, 0 と 1 だけの数. 他の値 (2, -1, 0.5, nan) は誤り. 全体に処置の群か対照の群の行が 1 行も無い場合も誤り (効果を比べられないため)
- 結果は有限の数 (0 と 1 でも額でもよい. 負の額も使える). nan と無限大は誤り
- 見積もりの nan は誤り (欠けた行は渡す前に除く). 無限大は使え, 同じ符号の無限大どうしは同点

Qini の曲線と係数は scikit-uplift (0.5.1) の `qini_curve` と `qini_auc_score` (既定の `negative_effect=True`) に合わせる. 違うのは, 結果の額を受ける点 (scikit-uplift は 0 と 1 だけを受ける. 額にも同じ式を使う), 無限大の見積もりを受ける点, 割る面積が 0 の場合に 0 を返す点 (scikit-uplift は 0 で割る誤り) だけ.

- 見積もりの高い順に並べ, 見積もりの同じ行はまとめて 1 つの点とする (同点の行を並べる順に依らない). 上位の n 行のうち処置の群の行の数を n_t, 対照の群の行の数を n_c, それぞれの結果の和を Y_t, Y_c として, Qini の値は Y_t - Y_c × n_t / n_c (対照の群の数を処置の群に合わせた結果の和の差). n_c が 0 の間は Y_t
- 曲線は 0 行の (0, 0) から始め, 同点のまとまりの終わりごとに 1 点を置く. 横軸は件数の `rows` (上位の行の数) と割合の `fraction` (`rows` / 全ての行の数) の両方を持つ. 縦軸の `qini` は件数 (結果が 0 と 1 なら, 処置で増えた結果の数. 額なら増えた額) で, scikit-uplift の `qini_curve` と同じ値. `random` は無作為に処置した場合の直線 (0 から全ての行での Qini の値まで) の値
- 係数 = (A - B) / (P - B). A は曲線の下の面積 (台形で数える), B は無作為の直線の下の面積 (全ての行の数 × 全ての行での Qini の値 / 2), P は理想の並びの曲線の下の面積. 理想の並びは, 見積もりの代わりに「処置の群の行は結果, 対照の群の行は結果の符号を変えた値」を使った並び (効果の出る処置の行を先に, 効果の出る対照の行を後に置く). 係数は理想の並びで 1, 無作為の並びで 0 に近く, 負の場合は無作為より悪い
- 係数は割合で書くか件数で書くかに依らない (横軸と縦軸を全ての行の数で割ると A, B, P が同じ比で縮むため). 件数の曲線から数える
- P - B が 0 の場合 (結果が全て 0 の場合など) は 0
- 曲線の点の数は見積もりの種類の数 + 1. `points` に 2 以上の整数を渡すと, 先頭と最後の点と, `rows` が全ての行の数の j / (`points` - 1) (j = 1, ..., `points` - 2) に初めて届く点だけを残す (値は間引く前の点と同じ. 重なる点は 1 つにする). 評価のファイルの上限 (20 MiB) に収めるため, 1 点はおよそ 100 バイトなので, 10 万行を超える場合は `points` を渡す

上位 K 割の効果の差の決まりは次の通り.

- 見積もりの高い順の上位 K 割の行のうち, 処置の群の結果の平均から対照の群の結果の平均を引いた値. `k` は 0 より大きく 1 以下の割合 (0.3 で上位 3 割. 1 で全ての行. 真偽値と nan は誤り)
- 上位の行の数は K × 全ての行の数の切り捨て. K は 10 進の表記どおりに数える (0.3 は 3/10. NumPy の float32 と `Decimal` も表記どおり. 2 進の丸めで 0.29 × 100 が 28 にならないため). scikit-uplift の `uplift_at_k` (`strategy="overall"`) は float のまま切り捨てるので, 同点が無くても K × 全ての行の数が 2 進で整数より僅かに小さくなる K では 1 行違う (例: 100 行で K = 0.29 は scikit-uplift が 28 行, ここは 29 行. 0.57 と 0.58 も同じ). そうでない K で同点が無ければ同じ値. scikit-uplift は K = 1 を誤りにするが, ここは全ての行として受ける. scikit-uplift の値と照合する場合は, 0.3 や 0.5 のように 2 進の丸めで切り捨てが変わらない K を使う
- 境に同点がある場合は, 境より上の行を全て数え, 境の同点のまとまりの行は全てを (残りの枠の数 / まとまりの行の数) の重みで数える (重み付きの平均. 同点の行を並べる順に依らない. scikit-uplift は並べる順で決める)
- 上位の行に処置の群か対照の群が無い場合 (上位の行の数が 0 の場合も) は nan (scikit-uplift と同じ. 平均を決められないため). `save_evaluation` では null になり, 監視する指標にすると誤りになるので, K を大きくする

照合の見本 (手で数えた値. ML-Inference も同じ値を確かめる) は次の通り. 曲線は (`rows`, `qini`) の点の並び.

| 見本 | 見積もり | 処置 | 結果 | 値 |
| --- | --- | --- | --- | --- |
| 同点を含む | `[0.9, 0.8, 0.8, 0.6, 0.5, 0.5, 0.3, 0.2]` | `[1, 1, 0, 1, 0, 1, 0, 0]` | `[1, 0, 1, 1, 0, 0, 0, 1]` | 曲線は (0, 0), (1, 1), (3, -1), (4, -1), (6, 0), (7, 2/3), (8, 0) (例えば 3 行では Y_t = 1, Y_c = 1, n_t / n_c = 2 で 1 - 2). 係数 = -5/72 (A = -5/6, B = 0, 理想の曲線 (0, 0), (2, 2), (6, 2), (8, 0) の P = 12). K = 0.25 は -1/3 (2 行. 0.9 の 1 行と, 境の同点の 0.8 の 2 行を重み 1/2 で. 処置の群は 1 / 1.5 = 2/3, 対照の群は 0.5 / 0.5 = 1), K = 0.3 は -1/3 (2.4 を切り捨てて 2 行), K = 0.75 は 0, K = 1 は 0 |
| 上位が処置の群だけ | `[0.9, 0.7, 0.7, 0.4, 0.4, 0.1]` | `[1, 0, 1, 1, 0, 0]` | `[1, 0, 1, 0, 0, 1]` | 曲線は (0, 0), (1, 1), (3, 2), (5, 2), (6, 1) (1 行では対照の群が無いので Y_t). 係数 = 12/13 (A = 9, B = 3, 理想の曲線 (0, 0), (2, 2), (5, 2), (6, 1) の P = 9.5). K = 0.25 は nan (1 行で対照の群が無い), K = 0.5 は 1, K = 1 は 1/3 |
| 結果が額 | `[3.0, 2.0, 2.0, 1.0, 0.5, 0.5]` | `[1, 1, 0, 1, 0, 0]` | `[500, 0, 200, 300, 0, 100]` | 曲線は (0, 0), (1, 500), (3, 100), (4, 200), (6, 500). 係数 = 2/23 (A = 1700, B = 1500, 理想の曲線 (0, 0), (1, 500), (2, 800), (4, 800), (5, 650), (6, 500) の P = 3800). K = 0.25 は nan, K = 0.5 は 50, K = 1 は 500/3 |
| 全て同点 | `[0.5, 0.5, 0.5, 0.5]` | `[1, 0, 1, 0]` | `[1, 0, 0, 1]` | 曲線は (0, 0), (4, 0). 係数 = 0 (A = B = 0, 理想の曲線 (0, 0), (1, 1), (3, 1), (4, 0) の P = 3). K = 0.25 は 0 (1 行の枠を 4 行に重み 1/4 で), K = 0.5 は 0 |
| 結果が全て 0 | `[0.3, 0.2, 0.1]` | `[1, 0, 1]` | `[0, 0, 0]` | 曲線は (0, 0), (1, 0), (2, 0), (3, 0). 係数 = 0 (P - B = 0). K = 0.5 は nan (1 行), K = 1 は 0 |

### 値段を変える判断の弾力性と利益の比べ

値段を変える判断 (試しに値段を変えた売れ行きの記録から, 値段を入力に入れて売れた数を予測し, 決めた値段の候補の中で利益が最も大きい値段を商品ごとに選ぶ) の評価は, `price_elasticity` (値段の弾力性) と `profit_lift` (選んだ値段と今の値段の見込みの利益の比べ) で数える. どちらも学習したモデルの予測した数を受け, 予測そのものはしない (候補の値段ごとに値段の列だけを変えて予測するのは, 呼ぶ側の骨組みと推論の段階). 売れた数の予測の誤差は今の回帰の指標 (損失の `valid/loss` など) で数える. 学習の段階では `on_epoch_end` の指標と `save_evaluation` の表に, 推論の段階 (ML-Inference) では結論の値に使う. 効果を見積もるの指標と同じく torch だけで数え, ML-Inference は ML-Found に依存せず, 下の見本で同じ値になることを自分のテストで確かめる.

```python
import torch

from ml_found import price_elasticity, profit_lift, save_evaluation

price = current_price               # 商品ごとの今の値段 (N,)
candidates = [-0.1, 0, 0.1, 0.2]    # 今の値段に対する変える割合 (0 を含む)

# 今の値段と 1 % 上げた値段での予測した数 (N,)
elasticity = price_elasticity(predict(price), predict(price * 1.01))

# 候補の値段 price × (1 + c) ごとの予測した数 (N, 候補の数)
units = torch.stack(
    [predict(price * (1 + c)) for c in candidates], dim=1
)
lift = profit_lift(price, candidates, units, cost=cost, max_change=0.1)

# 学習の後の評価の表と, 商品ごとの選んだ値段の表に
metrics = {
    "elasticity": elasticity["mean"], "profit_ratio": lift["ratio"],
    "profit_gain": lift["gain"]
}
save_evaluation(
    experiment, "price", metrics, "test",
    better={"profit_ratio": "higher", "profit_gain": "higher"}
)
save_evaluation(experiment, "price_items", lift["items"], "test")
```

入力の決まりは次の通り.

- 値段, 予測した数, 原価は, テンソル, NumPy の配列, 数の並びのどれか. 勾配の付いたテンソルと GPU のテンソルも受け, CPU の float64 に写して数える. 商品の数が 0 か, 形が合わない場合は誤り
- 予測した数は有限の 0 以上の数. 負の数, nan, 無限大は誤り
- 今の値段は有限の正の数. 原価は有限の 0 以上の数で, `None` の場合は全ての商品で 0 (利益は売上になる)
- 値段の候補は, 今の値段に対する変える割合の 1 次元の並び (0.1 で 1 割上げる. 候補の値段は 今の値段 × (1 + 割合)). 割合は -1 より大きい有限の数で, 同じ割合が 2 つある場合と, 0 (今の値段) を含まない場合は誤り. 真偽値も誤り. 数の並び, NumPy の配列, テンソル, `Decimal` は 10 進の表記どおりに読む (上位 K 割の効果の差の K と同じ. float32 のテンソルの 0.1 も 1/10)
- 変える幅の上限 `max_change` は 0 以上の有限の数か `None` (上限無し). 割合の絶対値が上限を超える候補は選ばない. 上限と割合は 10 進の表記どおりに比べる (上限 0.1 で割合 -0.1 と 0.1 は選べる). 今の値段 (割合 0) はいつも選べる

弾力性の決まりは次の通り.

- 商品ごとの弾力性 = 100 × (1 % 上げた値段での予測した数 - 今の値段での予測した数) / 今の値段での予測した数. 値段を 1 % 上げたときに, 予測した数が何 % 変わるか (-2 なら 2 % 減る). 1 % 上げた点との差で数える (中心の差は 1 % 下げた予測も要り, 対数の比は 1 % 上げた予測が 0 のときに無限大になるため. 1 % 上げた予測が 0 なら -100)
- 今の値段での予測した数が 0 の商品は nan (変わる割合を決められないため) で, 平均に数えない
- 戻り値は `items` (列 `elasticity` の, 商品ごとの弾力性の並び), `mean` (nan でない商品の弾力性の平均. 数える商品が無い場合は nan), `counted` (平均に数えた商品の数) の辞書

利益の比べの決まりは次の通り.

- 商品ごと候補ごとの見込みの利益 = (候補の値段 - 原価) × 予測した数. 原価より安い候補では負になる
- 商品ごとに, 上限の中の候補から見込みの利益が最も大きい候補を選ぶ. 最も大きい利益との差が 1e-9 × max(1, |最も大きい利益|) 以下の候補は同じ利益とみなし (候補の値段の 2 進の丸めで選ぶ候補が揺れないため), その中から割合の絶対値の小さい候補 (今の値段に近い候補. 同じなら割合の小さい候補) を選ぶ. 今の値段がいつも選べるので, 選んだ値段の見込みの利益は今の値段の見込みの利益より小さくならない
- 戻り値は次の辞書. `items` は列の名前から商品ごとの値の並びの辞書 (`save_evaluation` の表にそのまま渡せる)
    - `items`: `change` (選んだ割合), `price` (選んだ値段), `profit` (選んだ値段での見込みの利益), `current_profit` (今の値段での見込みの利益)
    - `total`: 選んだ値段での見込みの利益の和
    - `current_total`: 今の値段での見込みの利益の和
    - `gain`: `total` - `current_total` (0 以上)
    - `ratio`: `total` / `current_total`. `current_total` が 0 以下の場合は nan (比が利益の増減を表さないため. `gain` で比べる). `save_evaluation` では null になり, 監視する指標にすると誤りになるので, その場合は `gain` を監視する
    - `changed`: 選んだ割合が 0 でない商品の数

照合の見本 (手で数えた値. ML-Inference も同じ値を確かめる) は次の通り. まず弾力性の見本.

| 見本 | 今の値段での予測した数 | 1 % 上げた値段での予測した数 | 値 |
| --- | --- | --- | --- |
| 0 を含む | `[100, 50, 0, 20, 0]` | `[98, 50.5, 3, 0, 0]` | 弾力性 `[-2, 1, nan, -100, nan]`, `mean` = -101/3, `counted` = 3 |
| 全て 0 | `[0, 0]` | `[1, 0]` | 弾力性 `[nan, nan]`, `mean` = nan, `counted` = 0 |

利益の比べの見本は次の通り. 予測した数は商品ごとの候補の順の並び.

| 見本 | 今の値段, 原価, 候補, 上限 | 予測した数 | 値 |
| --- | --- | --- | --- |
| 上限と同じ利益を含む | 値段 `[10, 20, 5]`, 原価 `[6, 12, 5]`, 候補 `[-0.1, 0, 0.1, 0.2]`, 上限 0.1 | `[[120, 100, 85, 80], [60, 50, 40, 10], [30, 20, 10, 0]]` | 1 つ目は利益 360, 400, 425 (0.2 の 480 は上限を超えるので選ばない) で 0.1 (値段 11, 利益 425). 2 つ目は 0 と 0.1 が同じ 400 なので 0 (値段 20, 利益 400). 3 つ目は利益 -15, 0, 5 で 0.1 (値段 5.5, 利益 5). `current_profit` は `[400, 400, 0]`, `total` = 830, `current_total` = 800, `gain` = 30, `ratio` = 83/80, `changed` = 2 |
| 原価が無い | 値段 `[10, 8, 5]`, 原価 `None`, 候補 `[-0.2, 0, 0.2]`, 上限 `None` | `[[130, 100, 70], [10, 10, 9], [20, 18, 10]]` | 売上は 1 つ目が 1040, 1000, 840 で -0.2 (値段 8), 2 つ目が 64, 80, 86.4 で 0.2 (値段 9.6), 3 つ目が 80, 90, 60 で 0 (値段 5). `profit` は `[1040, 86.4, 90]`, `current_profit` は `[1000, 80, 90]`, `total` = 1216.4, `current_total` = 1170, `gain` = 46.4, `ratio` = 3041/2925, `changed` = 2 |
| 今の利益が 0 以下 | 値段 `[10, 10]`, 原価 `[12, 10]`, 候補 `[0, 0.5]`, 上限 `None` | `[[5, 2], [0, 0]]` | 1 つ目は利益 -10, 6 で 0.5 (値段 15). 2 つ目は 0 と 0 で 0 (値段 10). `profit` は `[6, 0]`, `current_profit` は `[-10, 0]`, `total` = 6, `current_total` = -10, `gain` = 16, `ratio` = nan, `changed` = 1 |

### 推薦の上位 K 件の Recall@K と NDCG

推薦 (利用者ごとに品物を点数の高い順に並べ, 上位 K 件を出す) の評価は, `recall_at_k` (Recall@K) と `ndcg_at_k` (NDCG@K) で数える. 入力は利用者ごとの並べた品物 (順位の順) と, 利用者ごとの正解の品物 (並べた後に反応した品物など). 利用者ごとの値を数え, 数える利用者で平均する. 学習の段階では `on_epoch_end` の指標に, 学習の後は `save_evaluation` の表の値に使う. 戻り値は Python の `float` で, 品物を Python の値に写して数える (scikit-learn などに依存しない). ML-Quest の並べるの骨組みの評価と ML-Inference の `rank_metrics` も同じ数え方で, 推論の段階 (ML-Inference) は ML-Found に依存せず, 下の見本で同じ値になることを自分のテストで確かめる.

```python
from ml_found import ndcg_at_k, recall_at_k, save_evaluation

top = ranked_items          # 利用者ごとの上位の品物の並び (順位の順)
relevant = reacted_items    # 利用者ごとの正解の品物の集合 (同じ利用者の順)

# on_epoch_end の中で, 監視する指標に (trainer.monitor: valid/ndcg_at_10)
metrics = {
    "recall_at_10": recall_at_k(top, relevant, 10),
    "ndcg_at_10": ndcg_at_k(top, relevant, 10)
}

# 学習の後の評価の表に
save_evaluation(
    experiment, "ranking", metrics, "test",
    better={"recall_at_10": "higher", "ndcg_at_10": "higher"}
)
```

入力の決まりは次の通り.

- 並べた品物と正解は, 利用者ごとの値の並び (リストかタプル. 同じ長さで, 同じ位置が同じ利用者) か, 利用者から値への辞書のどちらかで, 2 つとも同じ形にする (片方だけが辞書の場合と, 並びの長さが違う場合は誤り). 辞書の場合は並べた品物の辞書にいる利用者を数え, 正解の辞書にいない利用者は正解が無いとみなす. 正解の辞書だけにいる利用者は, 並べていないので数えない
- 利用者ごとの並べた品物は, 順位の順のリスト, タプル, 1 次元の NumPy の配列かテンソル (集合は順が無いので誤り). 正解は集合, リスト, タプル, 1 次元の NumPy の配列かテンソル (同じ品物が何度あっても 1 つ). 文字列は誤り (1 文字ずつの品物と取り違えないため)
- 品物は番号でも名前でもよく, 等しいかどうかだけで比べる. NumPy の配列とテンソルの値は Python の数に写して比べる
- 上位 K 件の中に同じ品物が 2 度ある場合は誤り (同じ正解を 2 度数えないため)
- `k` は 1 以上の整数 (真偽値は誤り. NumPy の整数も使える)

数え方の決まりは次の通り.

- 上位 K 件は, 利用者ごとの並べた品物の先頭から K 件. 並べた品物の数が K より少ない場合 (K が候補の品物の数より多い場合と, 前に反応した品物を除くと残りが少ない場合) は, 並べた全ての品物を数え, 割る数は変えない. 上位が空の利用者も数え, 値は 0
- 同点の品物は, 渡した順位のまま数える. 点数は受けないので, 点数の同じ品物の順は並べる側が決める
- Recall@K = 上位 K 件に入った正解の数 / 正解の全数. 正解が K より多い利用者は 1 に届かない (割る数を min(K, 正解の数) にしない)
- NDCG@K = DCG / 理想の DCG. DCG は上位 K 件の正解の順位 r (1 から) ごとの 1 / log2(r + 1) の和 (正解の重みは全て 1), 理想の DCG は正解を上から min(K, 正解の数) 件並べたときの和. 正解が K より多い利用者も, 上位 K 件が全て正解なら 1
- 正解の無い利用者は数えない (割る数が 0 になるため). 数える利用者がいない場合 (利用者が 0 人の場合も) は 0
- 2 つの指標とも, 数えた利用者の値の単純な平均

照合の見本 (手で数えた値. ML-Inference も同じ値を確かめる) は次の通り. 並べた品物と正解は利用者の順の並び.

| 見本 | 並べた品物 | 正解 | 値 |
| --- | --- | --- | --- |
| 利用者が混ざる | `[[10, 11, 12], [20, 21, 22], [], [40, 41, 42]]` | `[{11, 13, 14}, set(), {30}, {40}]` | 数えるのは 1, 3, 4 人目 (2 人目は正解が無い. 3 人目は上位が空で 0, 4 人目は 1 位が正解で 1). K = 1 は Recall 1/3, NDCG 1/3. K = 2 は Recall 4/9 ((1/3 + 0 + 1) / 3. 1 人目は 2 位の 1 件で 1/3), NDCG (1 / log2 6 + 0 + 1) / 3 (1 人目は (1 / log2 3) / (1 + 1 / log2 3) = 1 / log2 6). K = 3 は Recall 4/9, NDCG (1 / (1 + 1.5 × log2 3) + 0 + 1) / 3 (1 人目の理想は 1 + 1 / log2 3 + 1/2). K = 5 (並べた 3 件より多い) は K = 3 と同じ |
| 正解が K より多い | `[[3, 1, 2, 7]]` | `[{1, 2, 3, 4, 5}]` | K = 2 は Recall 2/5, NDCG 1 (上位 2 件が全て正解). K = 3 は Recall 3/5, NDCG 1. K = 4 は Recall 3/5, NDCG (1 + 1 / log2 3 + 1/2) / (1 + 1 / log2 3 + 1/2 + 1 / log2 5). K = 6 (並べた 4 件より多い) は Recall 3/5, NDCG (1 + 1 / log2 3 + 1/2) / (1 + 1 / log2 3 + 1/2 + 1 / log2 5 + 1 / log2 6) (理想は min(6, 5) = 5 件) |
| 数える利用者がいない | `[[1, 2], []]` | `[set(), set()]` | K = 1 と K = 2 の Recall と NDCG は 0 |

### 時間をまたぐ量の判断の分位点の損失と見込みの損

時間をまたぐ量の判断 (商品ごとの日ごとの売れた数の記録から先の期間の需要を分位点で予測し, 品切れの損と売れ残りの損の比から決まる分位点で発注の数を決める) の評価は, `pinball_loss` (分位点の損失) と `order_cost` (発注の数の見込みの損) で数える. どちらも学習したモデルの予測と, 呼ぶ側が決めた発注の数を受け, 予測と発注の数の決め方そのものはしない (水準 τ = `shortage_cost` / (`shortage_cost` + `holding_cost`) の分位点を選び, 在庫の数を引いて 0 から上限の間に収めるのは, 呼ぶ側の骨組みと推論の段階). 需要の予測の点の誤差は今の回帰の指標 (mae など) で数える. 学習の段階では `on_epoch_end` の指標と `save_evaluation` の表に, 推論の段階 (ML-Inference) では結論の値に使う. 値段を変える判断の指標と同じく torch だけで数え, ML-Inference は ML-Found に依存せず, 下の見本で同じ値になることを自分のテストで確かめる.

```python
from ml_found import order_cost, pinball_loss, save_evaluation

levels = [0.1, 0.5, 0.9]    # 学んだ分位点の水準

# 先の期間の実際の需要 (N,) と, 水準ごとの予測 (N, 水準の数)
pinball = pinball_loss(actual, quantiles, levels)

# 選んだ発注の数と, 前の期間と同じ数の発注 (どちらも在庫の数を足した, 売れる数)
cost = order_cost(
    actual, order, shortage_cost=3, holding_cost=1, baseline=last_period
)

# 学習の後の評価の表と, 水準ごとと行ごとの表に
metrics = {
    "pinball_loss": pinball["mean"], "order_cost": cost["total"],
    "order_cost_ratio": cost["ratio"], "order_cost_saving": cost["saving"]
}
save_evaluation(
    experiment, "demand", metrics, "test",
    better={
        "pinball_loss": "lower", "order_cost": "lower",
        "order_cost_ratio": "lower", "order_cost_saving": "higher"
    }
)
save_evaluation(experiment, "pinball_levels", pinball["levels"], "test")
save_evaluation(experiment, "order_items", cost["items"], "test")
```

入力の決まりは次の通り.

- 実際の値 (需要), 分位点の予測, 発注の数, 比べる発注の数は, テンソル, NumPy の配列, 数の並びのどれか. 勾配の付いたテンソルと GPU のテンソルも受け, CPU の float64 に写して数える. 行の数が 0 か, 形が合わない場合は誤り. 整数でない数も受ける (先の期間の需要の分位点は整数とは限らないため)
- 欠けた値 (nan) と無限大は誤り. 欠けた行は渡す前に除く
- 実際の値, 発注の数, 比べる発注の数は 0 以上の数 (需要と発注の数は負にならないため). 負の数は誤り
- 分位点の予測は有限の数なら負の数も受ける (モデルの出力を 0 で切る前の値でも損は決まるため). 水準の順に大きくなっていなくてもよい (分位点の交差もそのまま数える)
- 分位点の水準は 1 次元の並びで, 0 より大きく 1 より小さい有限の数. 0 と 1, 範囲の外, 同じ水準が 2 つある場合, 空の場合, 真偽値と文字列は誤り. 予測の列は水準と同じ順で, 水準を大きさの順に並べなくてもよい. 数の並び, NumPy の配列, テンソル, `Decimal` は 10 進の表記どおりに読む (値段を変える判断の候補と同じ. float32 の 0.1 も 0.1)
- 品切れの損 `shortage_cost` と売れ残りの損 `holding_cost` は商品 1 つあたりの同じお金の単位の有限の正の数 (Python と NumPy の数か `Decimal`. 10 進の表記どおりに読む). 0 以下は誤り (どちらかが 0 だと選ぶ分位点の水準 τ が 0 か 1 になり, 分位点で決まらないため). 全ての行で同じ値
- 比べる発注の数 `baseline` は任意 (`None` で比べない). 呼ぶ側が前の期間の実際の需要などを渡す

分位点の損失の決まりは次の通り.

- 行ごと水準ごとの損 = max(q (y - ŷ), (q - 1) (y - ŷ)). y は実際の値, ŷ は水準 q の予測. 予測が実際より小さいと q 倍, 大きいと 1 - q 倍の差が損になる
- 戻り値は `levels` (列 `level` (渡した順の水準) と `loss` (水準ごとの全ての行の損の平均) の辞書. `save_evaluation` の表にそのまま渡せる) と `mean` (全ての行と水準の損の平均. 水準ごとの平均の平均と同じ) の辞書. 小さいほど良い

学習の損失には `PinballLoss` を使う. `pinball_loss` は評価のために勾配を外して Python の数を返すので, step の損失には使えない. `PinballLoss` は `torch.nn.Module` で, 水準を作るときに受け, 実際の値と分位点の予測から, 勾配の付いたスカラーのテンソルを返す. 数え方は `pinball_loss` と同じで, 値は `pinball_loss` の `mean` と同じになる (下の見本). 評価の値と学習の損失の数え方がずれないよう, 骨組みは損失を自分で書かずにこれを使う.

```python
from ml_found import PinballLoss

criterion = PinballLoss([0.1, 0.5, 0.9])


def step(model, batch, phase):
    inputs, actual = batch    # actual は先の期間の実際の需要 (N,)
    quantiles = model(inputs)["quantiles"]    # (N, 水準の数)
    return {"loss": criterion(actual, quantiles)}
```

- 水準の決まり (範囲, 重なり, 空, 10 進の表記どおりに読む) は `pinball_loss` と同じで, 作るときに確かめる. 水準は `levels` (渡した順の数の並び) に持ち, 重みではないので `state_dict` とチェックポイントには入らない
- 分位点の予測は (行の数, 水準の数) の浮動小数点数のテンソル (勾配を流すため). 数の並び, NumPy の配列, 整数のテンソルは誤り. 損失は予測と同じ装置と型で数え, 予測と同じ装置と型のスカラーを返す (float32 の予測は float32 のまま, GPU の予測は GPU のまま. CPU の float64 に写さない)
- 実際の値はテンソル, NumPy の配列, 数の並びのどれかで, 予測と同じ装置と型に写して数える. 1 次元で行の数が予測と同じこと, 0 以上の有限の数であることは `pinball_loss` と同じ決まり. 実際の値のテンソルが勾配を持つ場合は, 実際の値にも勾配が流れる
- 分位点の予測の nan と無限大は誤り (`pinball_loss` と同じ). 値の確認のため, 呼ぶたびに GPU から真偽を 1 つ読む
- 予測の勾配は, 予測が実際より大きい要素で (1 - q), 小さい要素で -q を, 行の数 × 水準の数で割った値. 差が 0 の要素は 2 つの傾きの平均 (0.5 - q) を同じ数で割った値になる

見込みの損の決まりは次の通り.

- 行ごとの足りない数 = max(需要 - 発注の数, 0), 余った数 = max(発注の数 - 需要, 0), 損 = `shortage_cost` × 足りない数 + `holding_cost` × 余った数. 発注の数の引数には, 在庫の数と発注の数を足した, 売れる数を渡す (足りない数と余った数は売れる数と需要の差のため). 比べる発注の数も同じく, 在庫の数を足した数を渡す
- 戻り値は次の辞書. `items` は列の名前から行ごとの値の並びの辞書 (`save_evaluation` の表にそのまま渡せる)
    - `items`: `shortage` (足りない数), `excess` (余った数), `cost` (損). 比べる発注の数を渡した場合は `baseline_cost` (比べる発注の数での損) も
    - `total`: 損の和 (小さいほど良い)
    - `baseline_total`: 比べる発注の数での損の和. 比べない場合は nan
    - `saving`: `baseline_total` - `total` (正なら比べる発注の数より損が小さい). 比べない場合は nan
    - `ratio`: `total` / `baseline_total` (1 より小さいと比べる発注の数より良い). 比べない場合と, `baseline_total` が 0 の場合は nan (比が損の増減を表さないため. `saving` で比べる). `save_evaluation` では null になり, 監視する指標にすると誤りになるので, その場合は `total` を監視する

照合の見本 (手で数えた値. ML-Inference も同じ値を確かめる) は次の通り. まず分位点の損失の見本. 予測は行ごとの水準の順の並び. `PinballLoss` の戻り値も, 予測の型 (float64 と float32) と装置によらず `mean` と同じ値になる.

| 見本 | 実際の値 | 予測 | 水準 | 値 |
| --- | --- | --- | --- | --- |
| 水準が 1 つ | `[10, 4, 7]` | `[[8], [6], [7]]` | `[0.5]` | 差 y - ŷ は 2, -2, 0 で損は 1, 1, 0. `loss` = `[2/3]`, `mean` = 2/3 |
| 水準が複数 | `[10, 4, 0]` | `[[6, 9, 13], [2, 5, 4], [-1, 0, 2]]` | `[0.1, 0.5, 0.9]` | 1 行目の損は 0.4, 0.5, 0.3 (0.9 は差 -3 で 0.1 × 3). 2 行目は 0.2, 0.5, 0 (0.5 と 0.9 の予測が交差). 3 行目は 0.1, 0, 0.2 (予測 -1 も受ける). `loss` = `[7/30, 1/3, 1/6]`, `mean` = 11/45 |

見込みの損の見本は次の通り.

| 見本 | 需要, 発注の数, 損 | 比べる発注の数 | 値 |
| --- | --- | --- | --- |
| 足りる行と余る行が混ざる | 需要 `[10, 5, 8, 0]`, 発注 `[12, 3, 8, 2]`, `shortage_cost` 3, `holding_cost` 1 | `[7, 9, 8, 0]` | `shortage` = `[0, 2, 0, 0]`, `excess` = `[2, 0, 0, 2]`, `cost` = `[2, 6, 0, 2]`, `baseline_cost` = `[9, 4, 0, 0]`. `total` = 10, `baseline_total` = 13, `saving` = 3, `ratio` = 10/13 |
| 比べない | 需要 `[4, 6]`, 発注 `[6, 2.5]`, `shortage_cost` 2, `holding_cost` 0.5 | `None` | `shortage` = `[0, 3.5]`, `excess` = `[2, 0]`, `cost` = `[1, 7]` (`baseline_cost` の列は無い). `total` = 8, `baseline_total`, `saving`, `ratio` は nan |
| 比べる損が 0 | 需要 `[3, 0]`, 発注 `[3, 1]`, `shortage_cost` 4, `holding_cost` 1 | `[3, 0]` | `cost` = `[0, 1]`, `baseline_cost` = `[0, 0]`. `total` = 1, `baseline_total` = 0, `saving` = -1, `ratio` = nan |

## 15. 外部ライブラリの乱数

ML-Found が固定と保存をするのは Python, NumPy, torch の全体の乱数だけなので, 独自の乱数生成器を持つライブラリ (qiskit など) は, `register_rng` で登録すると同じ扱いになる.

```python
from qiskit_machine_learning.utils import algorithm_globals

from ml_found import register_rng, start_experiment


def seed_qiskit(seed: int) -> None:
    algorithm_globals.random_seed = seed


def get_qiskit_state() -> dict:
    return algorithm_globals.random.bit_generator.state


def set_qiskit_state(state: dict) -> None:
    algorithm_globals.random.bit_generator.state = state


# シードを固定する start_experiment の前に登録
register_rng("qiskit", seed_qiskit, get_qiskit_state, set_qiskit_state)
experiment = start_experiment(cfg, diff, str(project_dir))
```

- `start_experiment` で `experiment.seed` を渡して固定し, `trainer.valid_seed` でも固定する (検証の後は状態を戻す). 登録した名前は `train.log` に残る. `start_experiment` の後に登録すると固定されないので, `Trainer` の作成で警告が出る
- チェックポイントの乱数の状態 (`rng` の `extra`) に名前ごとに入り, 再開で戻る. 状態はテンソル, 数値, 文字列, `None`, NumPy の配列と, それらのリスト, タプル, 辞書だけで作る (NumPy の配列は安全に読み込める形に変えて保存し, 戻すときに配列に戻す). それ以外の型は `register_rng` の時点でエラーになる
- `get_state` は今の状態の写しを返す (検証の前に取り出して後で戻すので, 後から変わる値を返すと戻らない). `seed_fn` で乱数生成器を作り直す場合は, ライブラリが参照しているもの自体を置き換える (古い生成器を持ち続けるコードには効かない)
- 登録は Python のプロセスの中で続く. 同じプロセスで別の実験をする場合は `unregister_rng(name)` で外す (`register_rng` と同じく `ml_found` から import できる)
- 再開元と今で登録した乱数が違う場合は, 戻せない乱数を警告し, `train.log` にも残す. 再開元に状態が無い乱数は, `start_experiment` でシードから固定した最初の状態から使い直すので, 続けて学習した場合とは一致しない
- 外部ライブラリの乱数から作ったテンソルは, `step` の中でバッチと同じデバイスに移す (`.to(x.device)` など)

## 16. 学習の再開

`trainer.resume_from` に前回の `last.pt` を指定すると, モデル, Optimizer, Scheduler, 記録, Early Stopping, 乱数 (`register_rng` で登録したものを含む), loader (強化学習の環境の乱数, 環境の層の状態 (正規化の統計と `state_dict` の値), 探索の設定. 下) の状態を戻して続きのエポックから学習する.

```yaml
# configs/experiment/train.yaml
trainer:
  resume_from: outputs/wine/2026-11-08/16-18-27/last.pt
```

- 相対パスは, `experiment.output_dir` と同じくプロジェクトのディレクトリから辿る
- `last.pt` にはベストモデルの重みも含まれるので, `last.pt` だけで再開できる. `trainer.checkpoint.last_with_best: false` の場合は, 同じディレクトリの `best.pt` からベストモデルの重みを戻す. `best.pt` の学習しているモデルの重み (`model`) と, 別のモデルから取ったベストモデルの重み (`checkpoint` の `best_model`) のどちらを使うかは, 再開元のチェックポイントに残した `separate_best_model` (ベストモデルの重みを `trainer.ema.decay` の EMA か `eval_model` の上書きの返す別のモデルから取ったか) で決めるので, EMA の有無や `eval_model` の上書きを変えて再開しても再開元のベストモデルの重みが戻る (`separate_best_model` を持たないチェックポイントでは, 再開元の `ema` の有無と今の `eval_model` の上書きで決める). `best.pt` が無いか, `every_n_epochs` で `last.pt` の後に `best.pt` が更新されていてエポックが合わない場合は警告が出て, 再開後に改善しなければ最後のエポックの重みが残る (`run.json` の `final_weights` は `last`). `best_weights_only: true` の `best.pt` と `final_model.pt` は学習の状態を含まないので, 再開に指定すると出力先を作る前にエラーになる
- `trainer.monitor` や `trainer.mode` を再開元から変えるとエラーになる
- Early Stopping で止めた学習を再開しても学習しない
- Scheduler の状態は引数 (`T_max` など) ごと再開元から戻る. config で引数を変えて再開すると効かず, 警告が出る. 例えば `T_max: ${trainer.epochs}` は再開時に `epochs` を増やしても変わらない. `trainer.resume_scheduler: config` にすると, 引数が違う場合は config の引数で Scheduler を作り直し, 再開元と同じステップ数まで進める (再開元で進めた回数が, config の総ステップ数を超える場合はエラー). 再開した後の学習率は, config の引数で通しで学習した場合と同じになる (再開元までのエポックは元の引数で学習したまま). 引数を書き換えるだけでは, 例えば `T_max` を過ぎて 0 になった学習率は戻らないため. 指標で学習率を変える ReduceLROnPlateau は進め直せないので, 警告を出して再開元の状態を使う. 自分で作って渡した Scheduler は対象外. 比べるのは, config から作った直後の Scheduler の状態 (学習の進み具合で変わる値を除く) と, Scheduler が Optimizer に書き込む値 (`OneCycleLR` の `max_lr` など) なので, 名前を変えて持つ引数 (`OneCycleLR` の `epochs` は `total_steps`, `pct_start` は区切りのステップ, `CyclicLR` の `max_lr` は `max_lrs`, `base_lr` は `base_lrs`) の違いも分かる. Optimizer の引数 (`lr` など) の違いは対象外で, 再開元の値が使われる
- `last.pt` の保存の間隔を空けていて, 再開元の `best.pt` が `last.pt` より後のエポックのベストモデルの場合は, 再開で捨てる履歴のものなので新しい出力先に写さない (`train.log` に警告が出る)
- `persistent_workers=True` の DataLoader では, ワーカーの乱数の種を最初の 1 回だけ引くので, 再開した学習は通しで学習した場合と一致しない (警告が出る). 一致させたい場合は `num_workers: 0` にする
- 存在しない `resume_from` は, 出力先を作る前にエラーになる
- `trainer.ema.decay` の EMA の重みと更新回数も戻る. 再開元と EMA の設定が違う場合は「重みの指数移動平均 (EMA)」の「再開」
- 学習と検証の loader が `checkpoint_state()` (チェックポイントに入れる状態を返す) と `load_checkpoint_state(state)` (戻す) の両方を持つ場合は, その状態を `last.pt` の `loaders` (`train` と `valid`) に入れ, 再開とエポックの途中で止めた場合に戻す. `RolloutLoader` と `EvaluationLoader` が持つ (「強化学習」の「再開」). 自分で書く loader も, この 2 つを持てば同じく戻る. `load_checkpoint_state` は, 戻せなかった状態や再開元の値で上書きした設定を伝える文のリストを返してよく, `Trainer` が警告して `train.log` に残す. 状態は `register_rng` の状態と同じ型だけで作る (チェックポイントを安全に読み込むため). `state_dict` の名前は使わない (DataLoader の仲間が持つ `state_dict` を意図せず保存しないため). 再開元に loader の状態が無い (この仕組みの前のチェックポイントなど) か, 状態を受け取る loader が無い場合は警告し, `train.log` にも残す

同じ `Trainer` で `fit` を呼び直すと, 前の `fit` の続きとして, 最後に終えたエポックの次から `trainer.epochs` の分だけ学習する (3 エポックの後なら 4 から 6). 記録と `last.pt` は同じ実験のディレクトリに続けて書く.

```python
trainer = Trainer(model, experiment, step)
trainer.fit(train_loader, valid_loader)   # エポック 1 から 3
trainer.fit(train_loader, valid_loader)   # エポック 4 から 6 (続き)
```

- モデルの重みは, 前の `fit` の最後のエポックのものから続ける (学習の後にベストモデルや `eval_model` の重みに置き換えていても, 置き換える前に戻す). 前の `fit` がエポックの途中でエラーか割り込み (Ctrl-C) で止まった場合は, エポックの途中で止める場合 (「学習の外からの確認と中断」) と同じく, そのエポックを捨てて最後に終えたエポックの状態に戻してから続ける. Optimizer, Scheduler, 更新回数, 乱数, ベストモデルもそのまま続く. 置き換える前の重みは, 呼び直しに備えて `fit` の後も cpu に持つ
- Early Stopping は, 呼び直すたびに止めた印と我慢の回数を戻す (前の `fit` で止めていても学習する)
- `resume_from` は最初の `fit` だけで使う
- 監視する指標は前の `fit` と同じにする (検証用データを渡すかどうかを変えて指標が変わる場合はエラー)
- `auto` で補った Scheduler の引数 (`T_max` など) は最初の `fit` の長さなので, 続きの学習では予定を超えて進む (警告が出る). `OneCycleLR` は総ステップ数を超えられないので, 学習用データのバッチ数が分かる場合は, 前の `fit` までを含めた予定の総ステップ数が超える時点で, 始める前にエラーになる
- `run.json` の開始の時刻, 終えたエポック数, ベストモデルは前の `fit` から引き継ぎ, 所要時間 (`duration_sec`) は全ての `fit` の合計. 前の `fit` で置いた `STOP` のファイルは消す
- 学習の前の確認 (`OneCycleLR` の総ステップ数など) でエラーになった場合は, モデルの重みなどを変えない. 前の `fit` が学習を始める前に止まった場合は, 続きではなく最初の `fit` として扱う (`resume_from` も使う)
- Optimizer を差し替えたい場合 (一部を凍結した後に全体を学習するなど) は, 新しい `Trainer` を作る

## 17. 学習後のモデル

`fit` の後の `trainer.model` はベストモデルの重みになっている (`trainer.restore_best: true` かつ `select_best: true` の場合). `trainer.ema.decay` を使った場合と `eval_model` を上書きした場合は, EMA のモデルか上書きで返すモデルの重みになる.
後から使う場合は `final_model.pt` を読み込む. `final_model.pt` は cpu のテンソルで保存するので, 学習したデバイスに関わらず読み込める.

GAN のように損失が良し悪しの指標にならない場合は, `trainer.select_best: false` にするとベストモデルを選ばず (`best.pt` も作らない), 最後のエポックの重みが残る. 学習の途中で損失の定義が変わる場合 (β-VAE など) のように, ベストモデルは記録しておきたいが最後の重みも残したい場合は `trainer.restore_best: false` にする.

```python
import torch

model.load_state_dict(torch.load("outputs/wine/.../final_model.pt"))
```

## 18. 重みの指数移動平均 (EMA)

`trainer.ema.decay` に減衰率を書くと, ML-Found が学習するモデルの重みの指数移動平均 (EMA) を持ち, 検証, ベストモデル, `final_model.pt` に使う. 拡散のモデルや画像の生成でよく使う. 既定の `null` では使わない.

```yaml
# configs/default/train.yaml
trainer:
  ema:
    decay: 0.999         # 更新ごとに EMA = decay × EMA + (1 - decay) × 今の重み
    use_buffers: false   # BatchNorm の統計量などのバッファは平均せずに写す
```

- **更新**: パラメータを更新するたびに 1 回, Optimizer の更新の後に更新する (`update_parameters` の中. `torch.optim.swa_utils.AveragedModel` と `get_ema_multi_avg_fn` を使う). 最初の更新では今の重みを写す. 勾配を使わない Optimizer, closure を使う Optimizer, 名前付きの複数の Optimizer でも同じ. `manual_optimization: true` では, `step` の中で更新した後に 1 回更新する. `update_parameters` を上書きする場合は, `super().update_parameters(loss)` を呼ぶと EMA も更新される (呼ばないと更新されない)
- **バッファ**: `trainer.ema.use_buffers: false` (既定) では, バッファ (BatchNorm の統計量など) を平均せず, 更新のたびに学習しているモデルの値を写す. `true` ではパラメータと同じく平均する
- **検証と保存に使うモデル**: 検証, 学習用データの評価し直し (`eval_train`), ベストモデルの判定に使うモデルとその重みは EMA のモデル. `final_model.pt` と `fit` の後の `trainer.model` は, EMA のベストモデルの重み (`restore_best: false` か `select_best: false` では最後の EMA の重み). `last.pt` と `best.pt` の `model` は学習しているモデルの重みのまま (再開に使うため) で, EMA の状態 (重み, 更新回数, 減衰率, `use_buffers`) はチェックポイントの `ema` に入れる. EMA のベストモデルの重みは `best.pt` でも `checkpoint` の `best_model` に入れ, 重みだけの `best.pt` (`best_weights_only: true`) の `model` は EMA のベストモデルの重み. どれも「継承して細かく変える」の `eval_model` を上書きした場合と同じ動作
- **ベストモデルの選び方**: 監視する指標は EMA のモデルで検証した値. 減衰率が大きいと, 学習の始めの EMA は最初の重みを多く含むので, 始めのエポックの検証の値は学習しているモデルの値より悪い (ベストモデルは EMA の値どうしで比べるので, 選び方は変わらない). 検証しない場合は `train/loss` (学習しているモデルで計算した値) で判定し, 記録するのは EMA の重み. EMA の重みの良し悪しで選ぶには検証用のデータを渡す
- **再開**: `last.pt` の `ema` から EMA の重みと更新回数を戻すので, 再開した学習は通しの学習と一致する (他の状態と同じく「学習の再開」の条件で). エポックの途中で止めた場合 (「学習の外からの確認と中断」) と, 同じ `Trainer` で `fit` を呼び直した場合も, EMA は他の状態と一緒に戻って続く. 再開元と設定が違う場合は次のとおりで, どれも警告を出して `train.log` に残す
    - 再開元に EMA の状態が無い (EMA を使わなかった実験, `ema` を持たないチェックポイント): 再開した重みから EMA を始める. 再開元のベストモデルは, 値も重みも再開元の学習しているモデルのもの (`last_with_best: false` では再開元の `best.pt` の `model` から戻す). 再開した後は EMA で検証した値と比べる
    - 再開元が EMA を使い, 今回は使わない: 再開元の EMA の状態は使わない. 再開元のベストモデルは, 値も重みも EMA のもの (`last_with_best: false` では再開元の `best.pt` の `checkpoint` の `best_model` から戻す). 再開した後は学習しているモデルで検証した値と比べる
    - 減衰率か `use_buffers` が再開元と違う: 戻した EMA の重みから今回の値で続ける
- **継承の口との関係**: EMA は `trainer.ema_model` (`AveragedModel`. 使わない場合は `None`) が持ち, 既定の `eval_model()` は中のモデル (`trainer.ema_model.module`) を返す. 学習中に生成の見本を作る場合などは `self.eval_model()` を使う. `fit` の後の `trainer.eval_model()` は最後の EMA の重みのままで, EMA のベストモデルの重みを持つのは `trainer.model` (と `final_model.pt`) だけ. 継承したクラスでは `ema_model` という名前の属性を使わない (EMA の更新がその属性を使うので, 自分の EMA を置くと黙って 2 回更新される). `eval_model` を上書きした場合は上書きした方を使う (EMA を持ったまま学習しているモデルで検証する場合など. このとき EMA の重みは `final_model.pt` にならず, チェックポイントの `ema` にだけ残る). EMA の状態は ML-Found がチェックポイントの `ema` に入れるので, `extra_state_dict` に足さなくてよい (`extra` は継承したクラスの状態だけのまま). 継承して EMA を自分で持つ書き方 (「継承して細かく変える」の例) も動くが, `trainer.ema.decay` と両方を使うと 2 つの EMA を持つので, どちらかにする
- **メモリ**: モデルと同じ大きさの写しをデバイスに持つ. 写しは `Trainer` を作るときに作る (`copy.deepcopy`) ので, 写せないモデルでは `Trainer` の作成でエラーになる (出力先が作られた後なので `run.json` は `failed`)
- **確かめ**: `trainer.ema.decay` は `null` か, 0 より大きく 1 より小さい数 (真偽値は不可). 誤りは出力先を作る前に見つかる (`check_trainer_config`)

### 他のリポジトリへの影響

- **出力の形**: 版は変えない. `config.yaml` と `diff.yaml` に `trainer.ema` の項目が増える (項目を足す変更). `final_model.pt` は EMA を使っても学習するモデルと同じ名前と形の `state_dict` で, 読み方は変わらない (重みが EMA のものになるだけ). `run.json` の `final_weights` の値 (`best`, `last`, `partial`) の意味も同じで, EMA を使った実験ではどのエポックの EMA の重みかを表す. チェックポイントの中の形は約束の外 (FORMAT.md). 読む側は, `config.yaml` の `trainer.ema.decay` が `null` でなければ `final_model.pt` が EMA の重みと分かる
- **画面 (ML-Dashboard) の起動のフォームと config の確かめ**: 画面は config の確かめに `check_trainer_config` を使うので, 減衰率の誤りも起動の前に見つかる. 起動のフォームに欄が無くても, experiment の config に `trainer.ema.decay` を書けば使える (画面に欄を足すかは画面が決める. 足す場合は, この項目を知らない版の ML-Found の config では出さない). 画面の submodule の ML-Found がこの項目を知らない版だと, `trainer.ema` を書いた config は「default に無い項目」の誤りになるので, 画面とプロジェクトの ML-Found の版をそろえる

## 19. 大きな学習済みのモデルの微調整

学習済みのモデルの一部を凍結して微調整する場合に, 学習のスクリプトか継承したクラスで毎回書く手間を, config の項目にする. 勾配の蓄積, 混合精度 (bf16), 始める前に読む重み, 学習するパラメータだけのチェックポイントの 4 つで, どれも既定では使わない.

```yaml
# configs/default/train.yaml
trainer:
  accumulate_grad_batches: 4     # 4 バッチの勾配を貯めて 1 回更新
  precision: bf16                # 順伝播と損失を bfloat16 の自動の混合精度で計算
  init_weights:
    path: weights/pretrained.pt  # 始める前に読む重み (プロジェクトからの相対パス)
    source_prefix: encoder.      # ファイルのキーのうちこの前置きで始まるものを
    target_prefix: backbone.     # この前置きに付け替えて読む
    strict: false
  checkpoint:
    trainable_only: true         # last.pt と best.pt から凍結した重みを省く
```

**勾配の蓄積** (`trainer.accumulate_grad_batches`. 1 以上の整数, 既定 1): 決めた数のバッチの勾配を貯めてから 1 回更新する. メモリに載る小さなバッチで, 大きなバッチと同じ更新にするため.

- 各バッチの損失を貯める数で割ってから逆伝播するので, 貯めた勾配は損失の平均の勾配になる (同じ大きさのバッチなら, 合わせた大きなバッチの勾配と同じ). 勾配のクリッピングと `train/grad_norm` は貯めた勾配で 1 回
- エポックの最後の端数 (バッチ数が割り切れない場合) も 1 回更新し, 勾配は貯めた数の平均に直す. 勾配はエポックをまたがない (再開と, エポックの途中で止めた場合がエポックの区切りで決まるので, 通しの学習と一致する). バッチ数を持たない loader では, エポックの終わりに端数を更新する
- 更新回数 (`global_step`), `scheduler.interval: step` の Scheduler, EMA, 更新ごとの学習率の記録は, 更新した回だけ進む. 学習全体のステップ数 (`auto`, `warmup_ratio`, `OneCycleLR` の総ステップ数) は, エポックごとの更新の回数 (バッチ数を貯める数で割って切り上げた数) で補う. 指標はバッチごとのまま記録する
- 使えるのは, 勾配を使う 1 つの Optimizer を ML-Found が更新する場合だけ. closure を使う Optimizer (LBFGS など), 勾配を使わない Optimizer, `optimizers`, `manual_optimization` と一緒に書くと, 出力先を作る前にエラー (自分で作って渡した Optimizer は `Trainer` の作成でエラー)
- BatchNorm の統計はバッチごとに更新されるので, 大きなバッチの学習とは一致しない

**混合精度** (`trainer.precision`. `fp32` か `bf16`, 既定 `fp32`): `bf16` では, 学習, 検証, 学習用データの評価し直しの `step` (closure の中も) を `torch.autocast(device_type, dtype=torch.bfloat16)` の中で呼ぶ. 重み, 勾配, Optimizer の状態, チェックポイント, `final_model.pt` はモデルの重みの型 (普通は float32) のまま. bfloat16 は float32 と同じ指数の幅を持つので, 損失の倍率 (`GradScaler`) は使わない. 逆伝播は autocast の外で行う (PyTorch の勧め). 損失を float32 で求めたい場合は, `step` の中で `pred.float()` にしてから損失を求める.

- 学習のデバイスで使えるかは, 出力先を作る前に小さな行列の積を実際に計算して, 結果が bfloat16 になるかで確かめる (`check_precision`). bfloat16 に対応しない GPU と, 自動の混合精度を持たないデバイスではエラー. cpu, cuda (対応する GPU), mps で使える
- 再開と同じ `Trainer` での `fit` の呼び直しは, 同じ精度なら通しの学習と一致する. 精度を変えて再開してもよい (止めない)
- mps での効果の目安 (Apple M3, PyTorch 2.15, `nn.TransformerEncoder` の 6 層, 幅 512, 長さ 256, AdamW): バッチ 32 の 1 回の更新は fp32 が 655 ms, bf16 が 589 ms (1 割ほど速い). 確保したメモリ (`torch.mps.driver_allocated_memory`) は fp32 と bf16 でほぼ同じ (3.3 GB). バッチ 8 を 4 回貯めると, 確保したメモリは 1.2 GB に減り, 1 回の更新は 3% から 8% 遅い. mps では, メモリを減らすのは勾配の蓄積で, bf16 は速さが少し上がるだけ

**始める前に読む重み** (`trainer.init_weights`): `Trainer` を作るとき (EMA の写しと Optimizer を作る前) に, ファイルの重みの一部をモデルに読む. 学習済みの骨格を自分のモデルの一部に読む場合や, 学習するパラメータだけのチェックポイントから新しい学習を始める場合に使う.

- `path` のファイルは, `state_dict` を保存したもの (`final_model.pt` など) か, ML-Found のチェックポイント (`last.pt`, `best.pt`. 中の `model` を読む). 相対パスはプロジェクトのディレクトリから辿る. `torch.load(weights_only=True)` で, テンソルを読み込まずに参照して (`mmap`) 読む. 無いファイルと, 重みの辞書でないファイルは出力先を作る前にエラー (`check_init_weights`)
- `source_prefix` で始まるキーだけを選び, その前置きを `target_prefix` に付け替えて, モデルにある名前の重みを読む. 形が違う重みと, 1 つも読めない場合 (前置きの誤りなど) はエラー. `strict: true` では, モデルの全ての重みがそろい, 使わない重みも無いことが要る. 読んだ数, ファイルに無いモデルの重み, 使わなかったファイルの重み (10 個まで名前も) は `train.log` に書き, `Trainer` の属性 `init_weights_report` (`loaded`, `missing`, `unused` のキーのリスト) でも読める
- `trainer.resume_from` で再開する場合も読む (再開元の重みで上書きされ, 学習するパラメータだけのチェックポイントに無い凍結した重みはこの値になる)
- 凍結 (`requires_grad`) は変えない. 凍結はモデルの組み立て (ML-ModelControl の `train: false` など) か学習のスクリプトで行う. ML-ModelControl の定義の外部の重み (`weights`) と同じことは, 定義に書く方を勧める (付け替えの書き方と報告が多い). この項目は定義を使わないモデルと, ML-Found のチェックポイントを読む場合のため

**学習するパラメータだけのチェックポイント** (`trainer.checkpoint.trainable_only`. 既定 `false`): `true` では, `last.pt` と `best.pt` (重みだけの `best.pt` も) のモデル, ベストモデル, EMA の重みから, 保存するときに学習しない (`requires_grad` が偽の) パラメータを省き, 省いた名前と形を `trainable_only` に残す. バッファ (BatchNorm の統計など) は省かない (学習で変わるため). 凍結した重みの分だけ軽くなる. `final_model.pt` は全ての重みのまま (読む側の約束を変えないため).

- 再開では, 省いた重みを今のモデルの値で補ってから読む. そのため学習のスクリプトは, 再開する場合も同じ凍結した重みを持つモデルを作る (同じ学習済みの重みを読むか, `init_weights` で読む). 省いた名前が今のモデルに無いか形が違う場合と, 省いた名前の他に足りない重みがある場合はエラー. 凍結した重みの値が同じかは確かめない. 再開元では凍結していて今は学習する (凍結を外した) 重みも今のモデルの値で補い, その名前 (10 個まで) を `train.log` に警告として書く
- 再開元が学習するパラメータだけのチェックポイントかは, 再開元の `trainable_only` で決めるので, 今回の設定に関わらず再開できる. `last_with_best: false` で再開元の `best.pt` からベストモデルの重みを戻す場合も同じく補う
- エポックの途中で止めた場合と `fit` の呼び直しで戻す重みは, メモリの上の全ての重みなので変わらない

### 他のリポジトリへの影響

- **出力の形**: 版は変えない. `config.yaml` と `diff.yaml` に `trainer.accumulate_grad_batches`, `trainer.precision`, `trainer.init_weights`, `trainer.checkpoint.trainable_only` の項目が増える (項目を足す変更). `final_model.pt` は bf16 でも, 学習するパラメータだけのチェックポイントでも, モデルの重みの型のままの全ての重み. `last.pt` と `best.pt` の中の形は約束の外 (FORMAT.md)
- **画面 (ML-Dashboard) の起動のフォームと config の確かめ**: 画面が欄を出す項目は `trainer.precision` (文字. 選べる値は `fp32` と `bf16`) と `trainer.accumulate_grad_batches` (1 以上の整数). 値の誤りと組み合わせの誤りは `check_trainer_config`, デバイスで bf16 を使えるかは `check_precision(cfg, device)` (デバイスは `get_device(cfg.experiment.device)`) で, 起動の前に分かる. 欄を出すのは, 画面の submodule の ML-Found がこの項目を知る版の場合だけにする (知らない版では「default に無い項目」の誤りになる). `trainer.init_weights.path` を書いた config は `check_init_weights(cfg, project_dir)` で確かめる

## 20. 何の実験か, データ全体の評価, 見本

実験のディレクトリに, 任意で次の 3 つを書ける (形は [docs/FORMAT.md](FORMAT.md)). 書かなくても学習は変わらず, 書かなかった実験も同じく読める.

- **`task.yaml`** (何の実験か): 受け持ち (`role`. `predict`, `generate`, `rank`, `group`, `effect`), 出力の形と意味 (`output`. 方策は `action`. `what`), 正解の種類と出どころ (`target.kind`, `target.origin`), 作り方 (`method`. `scratch`, `finetune` など) と元にした実験や重み (`base`), 学習に使ったデータセット (`dataset`). 値の名前は ML-Quest と ML-EDA のものを写す. 正解が今ある処理の出力 (`target.origin.type: process`) なら, 指標はその処理との一致で, モデルはその処理を超えられない. 環境の報酬で学習した方策 (`environment`) なら, 指標はシミュレータなどの環境の報酬で, 実機での結果は別に確かめる. このことを指標と一緒に読めるようにするためのファイル
- **`evaluations/<名前>.json`** (データ全体の評価): グループごとの表, 曲線, 比べる相手との差, 守る指標を, 名前と分割の付いた表として書く
- **`samples/`** (見本): 画像, 文, 音, 点群の見本を, エポックと名前ごとに書き, `samples/index.json` に一覧を残す

`task.yaml` は default の config に `task` を書くと, `start_experiment` が書く. `task.dataset` に ML-EDA が書き出したデータセット (`dataset.yaml` の版 2 か 3 かそのディレクトリ. 相対パスはプロジェクトから辿る) を書くと, 受け持ち, 正解の種類と出どころを埋める. 書いた項目は埋めた値より優先する (`target.origin` は欄ごとでなく丸ごと置き換え. null と書いた項目は書かなかったものとして扱い, 埋めた値を残す). 誤り (知らない項目, 選択肢に無い値, 版 2 と 3 でない `dataset.yaml`) は出力先を作る前に見つかる. ML-Found は ml_eda に頼らずに `dataset.yaml` を読み, 版 1 (列の正解. 正解の出どころを持たない) と知らない版では止まる. 辞書でない `task` と, `task.yaml` の項目 (`role`, `output`, `what`, `target`, `method`, `base`, `dataset`) を 1 つも持たない `task` は, プロジェクトが別の意味で使う項目として警告して使わない. `task.yaml` の項目と他の項目を混ぜた `task` は誤りになる (今まで `task` に自分の辞書を書いていたプロジェクトは, 名前を変える).

```yaml
# configs/default/train.yaml
task:
  dataset: outputs/dataset          # 受け持ち, 正解の種類と出どころを埋める
  output: shape
  what: 撮影ごとの物の表面の点群
  method: scratch
```

スクリプトから書く場合は `write_task` を使う (後から書いた方で置き換える). 微調整では元にした実験 (`base.experiment`) か重み (`base.weights`. Hugging Face のモデルの名前など) を書く.

強化学習の方策は, 受け持ちを `effect` (問題の設定 (ML-Quest) では効果の見積もりの扱い), 出力の形を `action`, 正解の種類を `reward`, 正解の出どころを `environment` (環境の報酬) にする.

```python
write_task(experiment, {
    "role": "effect",
    "output": "action",
    "what": "観測から押す向きを選ぶ方策",
    "target": {
        "kind": "reward",
        "origin": {"type": "environment", "name": "CartPole-v1"}
    },
    "method": "scratch"
})
```

```python
from ml_found import save_evaluation, save_sample, task_from_dataset, write_task

experiment = start_experiment(cfg, diff, PROJECT_DIR)
task = task_from_dataset("outputs/dataset", project_dir=PROJECT_DIR)
task.update({
    "what": "文ごとの感情",
    "method": "finetune",
    "base": {"weights": "bert-base-uncased"}
})
write_task(experiment, task)
```

評価は学習の後に `save_evaluation(experiment, 名前, 表, 分割)` で書く. 表は行の辞書の並び, 列の名前から値の並びの辞書, 1 行の辞書, pandas の DataFrame (索引は使わないので, 要るなら `reset_index()` で列にする) のどれか. 値は数, 文字, 真偽値, None (テンソルと NumPy の数も可. nan と無限大は null). `description` (何の評価か), `kind="curve"` と `x` (曲線の横軸の列), `baseline` (比べる相手の名前), `guard=True` (守る指標の表), `epoch` (評価した重みのエポック), `better` (列ごとの指標の向き. 列の名前から `"lower"` か `"higher"` の辞書) を添えられる. 実験の代わりに実験のディレクトリのパスを渡すと, 終えた実験にも後から書ける.

```python
save_evaluation(
    experiment, "per_group",
    {
        "group": ["a", "b"],
        "iou": [0.81, 0.64],
        "iou_baseline": [0.78, 0.70],
        "iou_diff": [0.03, -0.06]
    },
    "test", baseline="今の処理", guard=True, better={"iou": "higher"}
)
```

`better` (列ごとの指標の向き) の決まりは次の通り.

- 書かなくてもよい. 書かない列は, 画面が指標の名前から向きを推し量るか, 分からないとする. 名前から向きの決まらない指標 (`neg_log_loss`, 独自の指標など) と守る指標の表では書くのを推奨する (守れたかの判定を画面が言い切れるように)
- 表に無い列と, `lower` と `higher` でない値は誤り. グループや曲線の横軸の列にも書けるが, 指標でない列には書かなくてよい. DictConfig などの Mapping も受け, NumPy の文字などは `str` にそろえる
- 比べる相手との表の形の `<指標>`, `<指標>_baseline`, `<指標>_diff` は同じ向きなので, どれか 1 つに書けば JSON には同じ指標の列に広げて書く (上の例では `iou_baseline` と `iou_diff` も `higher`). 終わりは 1 段ずつ外し, 広げた列からさらに変化が無くなるまで広げる (`acc_baseline_diff` に書くと `acc_baseline` に, そこから `acc` と `acc_diff` にも届く). 書いた JSON の `better` は, もう広げるところの無い形. 広げるのは終わりを外す向きだけなので, `<指標>` に書いても `<指標>_baseline_diff` には届かない (`acc` に書くと `acc_baseline` と `acc_diff` まで. `acc_baseline_diff` にも向きが要るならそこにも書く)
- 1 つの列に違う向きが届く書き方は誤り (`acc` と `acc_baseline_diff` に違う向きを書くと `acc_baseline` に 2 つが届く)
- `_diff` や `_baseline` で終わる本物の指標の名前 (`score_diff` という独自の指標など) も, 終わりを外した名前の仲間と見なす. 仲間と違う向きにする場合は, その表で列の名前を変える
- `load_evaluations` は各評価の `better` を辞書で返す. 書かなかった評価と, `better` を持たない前のファイルは空の辞書. 表に無い列と知らない値の欄は除き, 欠けた同じ指標の列には同じく広げる. 違う向きが届く列は埋めない

比べる相手との表は, 上のように 1 行を 1 つのグループにし, 指標ごとに `<指標>` (この実験), `<指標>_baseline` (比べる相手), `<指標>_diff` (この実験 - 比べる相手) を並べる形を推奨する (画面が差の列を見分けられる). 相手が 2 つ以上ある場合などは, 1 行を (グループ, 答えた側) にし, 答えた側を `answer` の列 (`model` か相手の名前) に書く形も使える (docs/FORMAT.md の「evaluations」).

評価と見本の名前は英数字と `_`, `.`, `@`, `-` だけで, 先頭は `.` 以外 (見本は `index.json` と, `samples/` の直下のファイルと同じ名前も不可).

見本は `save_sample(experiment, 名前, 値, epoch=...)` で書く. 文字は文 (`.txt`), `pathlib.Path` はファイルをそのまま写す (種類は拡張子から), 配列とテンソルは `kind` を渡す. `image` は (高さ, 幅), (高さ, 幅, チャンネル), (チャンネル, 高さ, 幅) で, 小数は 0 から 1, 整数は 0 から 255 の値 (PNG). `audio` は (サンプル数,) か (チャンネル, サンプル数) で -1 から 1 の値と `sample_rate` (16 bit の WAV. チャンネルは 8 まで. (サンプル数, チャンネル) の配列は `.T` で入れ替える). `point_cloud` は (点の数, 3) で nan の無い座標 (PLY). `mask` は (高さ, 幅) の領域の番号 (0 から 255 の整数か真偽値) で, 決まった色の表 (PASCAL VOC と同じ. 0 は黒) で色分けした PNG を書き, 番号ごとの名前と色の表を一覧の `classes` に残す (名前は `classes` に名前のリストか番号から名前の辞書で渡す). 小数の配列は確率の取り違えとして誤り (argmax で番号にする). 画像と音の nan と無限大, 整数の音 (型の最大値で割って小数にする) は誤り. 学習の途中では `on_epoch_end` の中で `epoch=self.epoch` を渡し (`samples/<名前>/epoch_0005.png` に置く), 学習の後はエポックを省く (`samples/<名前>/final.png`). 同じエポックと名前で書き直すと置き換える. `split` と `caption` (どちらも文字) も添えられる. 値の決まり (点群の座標の単位, 正規化で足した量など) は, `caption` の文でなく `convention` (JSON にできる辞書) に書く (画面が元の座標に戻せるように). 点群の `shift` は ML-EDA の書き出しの `shift` と同じ向き (正規化した座標 = 元の座標 + shift) で, ML-EDA の値をそのまま写せる. `scale` は正規化の後に割った大きさ. 同じプロセスのスレッドからは同時に書けるが, 複数のプロセス (DDP のランクなど) から書く場合はランク 0 だけが書く. 前からプロジェクトが `samples/` の直下に置いていた番号付きの画像 (`samples/epoch_010.png`) はそのまま使え, `load_samples` も見本として返す (`legacy: true`).

```python
save_sample(experiment, "seg", pred_mask, kind="mask",
            classes=["背景", "物", "縁"], caption="予測")
save_sample(experiment, "cloud", points, kind="point_cloud",
            convention={"units": "m", "shift": [0.1, 0.0, 0.5], "scale": 1.0})
```

`original_coordinates(点群, 項目["convention"])` は, `load_samples` が返した点群の見本の座標を元の座標 (`値 × scale - shift`) に戻す (画面が符号を取り違えないため). `samples/index.json` は形式の版 (`format_version`. 今は 2) を持ち, 版の無い一覧 (版 1. `shift` が逆の向き) は `load_samples` が今の意味に写して返す. 新しい版の一覧の見本は, `convention` を null にし, `newer_version: true` を付けて返す (docs/FORMAT.md の「samples」).

読む側は `load_task`, `load_evaluations`, `load_samples` で読める (無い実験では `None` か空のリスト). `load_evaluations` は, シンボリックリンク, 普通でないファイル (FIFO など), 20 MiB (公開の `MAX_EVALUATION_BYTES`. 画面も同じ上限を使う) を超えるファイルを読まずに飛ばし, `evaluations/` 自体がリンクの実験は空とする (実験のディレクトリの外を読まず, 止まらないため). 開く前にリンクを辿らずに調べ, 開くときもリンクを辿らず (`O_NOFOLLOW`) 待たずに (`O_NONBLOCK`) 開き, 開いたものが調べたものと同じ普通のファイルかを確かめ, 上限の 1 バイト先までだけ読む. `dir_fd` を使える環境では `evaluations/` をリンクを辿らずに開いた番号から調べて開く (確かめた後に `evaluations/` を入れ替えられても外を読まないため). 実験のディレクトリより上の入れ替わりは確かめない. `load_task` (1 MiB まで) と, `load_samples` と `save_sample` が読む `samples/index.json` (20 MiB まで) も同じ読み方で読む. 読まなかった `task.yaml` は無いものとし (`None`), 読まなかった一覧は読めない一覧とする (`load_samples` は直下の画像だけを返し, `save_sample` は誤りで止まる). `samples/` 自体をリンクにすると, ML-Found は見本を読み書きしない (`load_samples` は一覧も直下の画像も読まずに空を返し, `save_sample` は外にディレクトリを作る前に止まる). `describe_origin(task["target"]["origin"])` は, 指標が何との一致かの説明の文を返す (画面が指標の横に出すため). 新しい版の ML-Found が書いた `task.yaml` は, `load_task` が正解の出どころを分からないもの (`unknown`) に置き換え, `newer_version: true` を付けて返す (意味の変わったかもしれない欄から「処理を超えられない」と言い切らないため). 版が整数でない `task.yaml` は誤り.

## 21. 強化学習

Gymnasium の形の環境 (`reset(seed=...)` が `(観測, info)` を, `step(行動)` が `(観測, 報酬, 終了, 打ち切り, info)` を返すもの) とやり取りして, 行動を選ぶ方策を学習できる. 別の仕組みは作らず, 環境とのやり取りを「バッチを返す loader」にして, `Trainer` の step とエポックの決まりに乗せる. 使うには追加の依存 `rl` (Gymnasium) を入れる (上の「追加の依存」). 例は `examples/cartpole/` (CartPole を REINFORCE で学習し, 守りと一緒に書き出す) と `examples/pendulum/` (Pendulum の連続の行動の方策を `ActorCriticStep` で学習し, トルクの範囲と角速度の止める条件の守りと一緒に書き出して `evaluate_policy` で確かめる).

```python
from ml_found import Trainer, load_project_config, start_experiment
from ml_found.rl import (
    EvaluationLoader, Exploration, ReinforceStep, RolloutLoader, make_env
)

# 学習用と評価用の環境は別に作る (Gymnasium が無ければ入れ方を示して止まる)
train_env = make_env("CartPole-v1")
valid_env = make_env("CartPole-v1")
model = PolicyNet(...)  # 観測から行動ごとのスコアを返すモデル

# 1 エポックは 20 エピソード (5 エピソードで 1 回更新), 評価は別の種の 10 エピソード
train_loader = RolloutLoader(
    train_env, model, episodes=20, batch_episodes=5, gamma=0.99,
    exploration=Exploration(temperature=1.0, epsilon=0.0), seed=42
)
valid_loader = EvaluationLoader(valid_env, model, episodes=10, seed=10000)
trainer = Trainer(model, experiment, ReinforceStep(entropy_coef=0.01))
history = trainer.fit(train_loader, valid_loader)
```

連続の行動 (Box) の方策のモデルは, 観測から行動の平均を返し, 対数の標準偏差 `log_std` を持つ (下の「連続の行動」).

```python
import torch
from torch import nn


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        # 親クラスのコンストラクタを呼び出す
        super(GaussianPolicy, self).__init__()

        # 平均の線形層と, 観測によらない対数の標準偏差
        self.mean = nn.Linear(obs_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 行動の平均
        mean = self.mean(x)

        return mean
```

- **経験の記録**: `ExperienceBuffer` が 1 歩ごとの観測, 行動, 報酬, 実際にその行動を選んだ確率 (連続の行動では確率密度. 対数で持つ) と安全の違反を貯め, エピソードを終えるときに割引した報酬の和を求める. `to_batch()` がテンソルの辞書にする. 1 歩ごとの `observations`, `actions`, `rewards`, `returns` (割引した報酬の和), `behavior_log_probs`, `behavior_probs`, エピソードごとの `episode_rewards`, `episode_lengths`, `episode_violations`, `episode_terminated` (環境の終了なら 1, 打ち切りなら 0), `final_observations` (最後の歩の後の観測. 打ち切りで `end_episode(terminated=False)` に `final_observation` を渡さない場合はエラー. 行動の前の観測で見積もらないため. 終了で渡さない場合は最後の歩の観測を入れ, 価値は 0 にする), `episode_success` (環境が成功を知らせた場合) と, 割引率 `gamma` (0 次元) を入れる. 自分で loader を書いて `ReinforceStep` に渡す場合は `observations`, `actions`, `returns`, `behavior_probs` (連続の行動では `behavior_log_probs`) とエピソードごとの 3 つの集計が, `ActorCriticStep` に渡す場合はさらに `rewards`, `episode_terminated`, `final_observations`, `gamma` が要る
- **エポック**: `RolloutLoader` は今の方策でエピソードを集め, `batch_episodes` ごとに 1 つのバッチを返す (バッチごとに 1 回更新するので, 次のバッチは更新した方策で集める). エポックの長さは `episodes` (エピソードの数) か `steps` (歩数. 達したエピソードの終わりまで) のどちらかで決める. 環境の乱数の種は最初の `reset` だけに渡し, 後は環境の乱数を続けて使う. `max_episode_steps` で 1 エピソードの歩数を切れる. 環境の打ち切り (TimeLimit など) と `max_episode_steps` で切ったエピソードも, 終わったもの (最後の歩の後の報酬を 0) として報酬の和を求める (モンテカルロの REINFORCE. 打ち切りの後の価値を足すのは, 下の `ActorCriticStep`)
- **探索**: `Exploration(temperature, epsilon)` で, 方策の分布を温度で平らにし, `epsilon` の割合で一様に選ぶ. 実際に選んだ確率を記録し, `ReinforceStep` は方策の確率との比 (1 で打ち切る) で重みを補正する. 比の方策の確率は, 行動を選んだときと同じ推論モードで求める (Dropout があっても, 既定の温度 1, `epsilon` 0 では更新の前の比は 1. 損失の勾配は学習モードで求める). 途中で変える場合は `add_callback` の関数で `train_loader.exploration` を差し替える
- **評価**: `EvaluationLoader` は学習とは別の環境で, 毎回 `seed`, `seed + 1`, ... の種で始めたエピソードを, 最もスコアの高い行動で動かす (エポックの間で比べられる). 全てのエピソードを 1 つのバッチにする. `record_trajectory=True` で 1 歩ごとの軌跡も記録する (下の「評価の軌跡の表」)
- **指標**: エピソードごとの報酬の和 (`reward`), 歩数 (`length`), 安全の違反の数 (`violations`. 環境が `info` の `safety_key` (既定 `cost`) に 0 より大きい値を入れた歩の数) を, エピソードの数で重み付けして平均する. `success_key` (`is_success` など) を指定すると, 最後の歩の値から成功の割合 (`success`) も記録する. 学習と検証では方策の損失 (`loss`) とエントロピー (`entropy`) も記録する
- **損失**: `ReinforceStep` は, バッチの平均を基準 (baseline) にした REINFORCE. 学習の `loss` は方策の勾配の損失, 検証の `loss` は評価の報酬の和の平均の符号を反転した値 (意味が違うので, 監視する指標は `trainer.monitor: valid/reward` と `trainer.mode: max` にする). 検証では, 評価で通った観測ごとの方策のエントロピーの平均 (`valid/entropy`) も記録する
- **報酬が上限で並ぶ場合**: CartPole のように報酬の和に上限がある環境では, 評価の報酬が上限に達すると後のエポックは同じ値になり, ベストモデルは最初に達したエポックのままになる. `valid/loss` と `valid/length` は `valid/reward` と一緒に動くので, `trainer.monitor_tiebreak` に指定しても比べられない. 報酬と別に動く指標を指定する. 推奨は `trainer.monitor_tiebreak: train/reward` と `trainer.monitor_tiebreak_mode: max` (探索を含む学習のエピソードでも高い報酬を取る方策を選ぶ. 学習の環境の乱数で揺れる). 迷いの少ない方策を選ぶなら `valid/entropy` と `min`. 安全の違反が起きる環境では `valid/violations` と `min` も使える (違反が 0 で並ぶ場合は比べられない)
- **行動の空間**: 行動を番号で選ぶ環境 (Discrete. 番号は 0 から) と, 連続の行動の環境 (Box) に対応する. 他の空間 (MultiBinary など) はエラー
- **連続の行動**: 方策は正規分布にする. モデルは観測から行動の平均 (要素の数は行動の空間の要素の数) を返し, 観測によらない対数の標準偏差 `log_std` (要素ごとか 1 つの `nn.Parameter`) を持つ (無ければ `RolloutLoader` と step がエラー). 標準偏差は `log_std` の値を -20 以上 2 以下に制限して求める (発散して標準偏差が 0 か無限大になり, 損失が inf や nan になるのを避けるため). 制限の外に出た `log_std` も勾配は 0 にならず (値だけを制限し, 勾配はそのまま通す), 更新で範囲の中に戻れる. ただし範囲の外で外向きの勾配 (標準偏差を広げるか狭める向き) が続くと, `log_std` の生の値は動き続ける (使う値は範囲の端のまま). 戻るのに掛かる更新の回数は, 外に出た量で増える. 無限大になった `log_std` は使う値を範囲の端にし, 勾配は 0 にする (戻せないので, モデルを作り直す). 形が () の Box は未対応 (形を (1,) にする). 学習では平均と標準偏差 (`Exploration` の温度を掛ける) の正規分布から選び, 評価 (`EvaluationLoader`) では平均を選ぶ. `epsilon` は番号で選ぶ行動だけに使え, 連続の行動ではエラーにする. バッチの `actions` は (歩数, 要素の数) の float32 で, 選んだ確率密度の対数を `behavior_log_probs` に入れる (`behavior_probs` は確率密度で, 小さい値は 0 にアンダーフローするので連続の行動では使わない) (`ReinforceStep` は要素ごとの対数の確率密度の和で方策の勾配を求め, 比は対数の差から求める)
- **行動の範囲**: 環境には, 選んだ行動を行動の空間の範囲で clip して渡す. 経験には clip する前の値を記録する (選んだ確率密度と合わせるため). 書き出した方策 (下の「方策の書き出しと守り」) は平均を行動とし (`head` は `deterministic`), 守りの `action_low` と `action_high` (空間の範囲との狭い方) で clip する. 学習の clip は空間の範囲だけなので, 守りで空間より狭くした範囲は学習の間は当たらない. 狭い範囲の中で学ばせる場合は, 環境の行動の空間をその範囲にする (Gymnasium の `RescaleAction` など)
- 行動を選ぶときは, モデルを一時的に推論モードにする (1 件の観測の BatchNorm と, Dropout で確率がずれるのを避けるため)
- `RolloutLoader` は学習だけに使う. `trainer.eval_train: true` では, 評価し直しで学習の環境の乱数が進んで結果が変わるので, エラーにする (評価は `EvaluationLoader`)
- 再開 (`resume_from`) とエポックの途中で止めた場合の戻し方は, 下の「再開」
- `eval_model` を上書きした場合は, `EvaluationLoader` に渡すモデルもそのモデルにする (loader は渡されたモデルで行動を選ぶ)

### 再開

`resume_from` で再開した学習は, 環境の乱数と状態も戻り, 通しで学習した場合と一致する. `RolloutLoader` と `EvaluationLoader` の状態は `last.pt` に入る (「学習の再開」).

- **エピソードの途中では保存しない**: `RolloutLoader` はエピソードを途中で止めずにエポックを区切り, チェックポイントはエポックの終わりだけに書く. 保存するときの環境は必ずエピソードの間 (次は `reset`) なので, 環境の中の物理の状態 (CartPole の `state` など) は保存しない. エポックの途中で止めた場合は, 前のエポックの終わり (か学習を始めた時点) の状態に戻し, 次の `reset` から続ける
- **環境の乱数**: Gymnasium の環境は `reset` で環境の乱数 (`np_random`) から初めの状態を決めるので, 乱数を戻せば続きのエピソードが通しと同じになる. `RolloutLoader` は `env.unwrapped.np_random` の状態 (`bit_generator.state`) を保存して戻す (NumPy の MT19937, PCG64, PCG64DXSM, Philox, SFC64). 最初の `reset` の前は保存しない. 自分で作った Wrapper は `unwrapped` で中の環境を返す (返さないと乱数が見つからず, `state_dict` も無ければ戻すときに警告が出る)
- **loader の中の状態**: 最初の `reset` だけ種を渡した印と, 探索の設定 (`exploration`) を戻す. 印を戻さないと, 再開した loader が最初の `reset` で種を渡し直して同じエピソードを繰り返す. 探索の設定は, 登録した関数で差し替えた値も続くように再開元の値で続ける (今の設定と違う場合は警告). 行動を選ぶ乱数は torch の全体の乱数なので, `Trainer` が戻す
- **エピソードを越えて続く状態**: `reset` で初めに戻らない状態は, 環境の層 (外側の Wrapper から `.env` で辿った各層と `unwrapped`) ごとに保存して戻す. 層の深さ (外側から 0), クラスの名前, 種類で再開元の層と対応させ, 合わない層 (違う Wrapper で包んだ場合など) は戻さずに警告する
    - Gymnasium の観測と報酬の正規化 (`NormalizeObservation` の `obs_rms`, `NormalizeReward` の `return_rms` と `discounted_reward`) の統計は, 何もしなくても保存して戻す
    - 自分で作った環境と Wrapper は, `state_dict()` (状態を返す) と `load_state_dict(state)` (戻す) を持たせる. 両方を持つ層はこの 2 つを使う (正規化の統計より優先). Gymnasium には状態の取り出しと戻しの決まりが無く, MuJoCo の環境の `set_state(qpos, qvel)` と名前がぶつかるので, PyTorch と同じ名前にする
    - `state_dict` の値は `register_rng` の状態と同じ型 (テンソル, 数値, 文字列, `None`, NumPy の配列と, それらのリスト, タプル, 辞書. 辞書のキーは文字列か整数) だけで作る. それ以外の型は保存のときにエラーになる
    - `RecordEpisodeStatistics` のエピソードの統計 (`info` の `episode` と数) は戻せないので, 再開で警告する (学習には使わない)
    - **制限**: `state_dict` を持たない自作の環境と Wrapper の, エピソードを越えて続く状態は戻らず, 外から見つけられないので警告も出ない. その場合, 再開した学習は通しと一致しない. 普通の Gymnasium の環境 (CartPole, Pendulum など) は乱数だけで一致する
- **評価の環境**: `EvaluationLoader` はエピソードごとに決まった種 (`seed`, `seed + 1`, ...) で `reset` するので, 環境の乱数は保存しない. 環境の層の状態 (上の正規化の統計と `state_dict` の値) だけを保存して戻す

エピソードを越えて続く状態を持つ自作の環境の例 (エピソードの数を数え, 観測に出す). `state_dict` と `load_state_dict` を持つので, 再開した学習は通しと一致する (同じ形の環境 `CarryEnv` で `tests/test_rl_resume.py` が確かめる).

```python
class CountingEnv:
    def reset(self, seed=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.episodes += 1
        obs = np.array([self.episodes % 10], dtype=np.float32)
        return obs, {}

    def state_dict(self) -> dict:
        return {"episodes": self.episodes}

    def load_state_dict(self, state: dict) -> None:
        self.episodes = int(state["episodes"])
```

### 価値の関数を使う方法 (ActorCriticStep)

`ActorCriticStep(value_coef=0.5, entropy_coef=0.0, gae_lambda=0.95, normalize=True)` は, 状態の価値の関数で利得を見積もる Actor-Critic (集めた経験で 1 回だけ更新する形. 利得は GAE). 番号で選ぶ行動と連続の行動のどちらにも使える. PPO のように同じ経験で何度も更新する方法は, 自分で loader と step を書けば今の仕組みで動かせる (名前付きの複数の Optimizer も使える).

```python
import torch
from torch import nn

from ml_found.rl import ActorCriticStep


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        # 親クラスのコンストラクタを呼び出す
        super(ActorCritic, self).__init__()

        # 方策 (平均と log_std) と価値の関数
        self.mean = nn.Linear(obs_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(obs_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 行動の平均 (書き出した方策もこの出力だけを使う)
        mean = self.mean(x)

        return mean

    def value(self, x: torch.Tensor) -> torch.Tensor:
        # 状態の価値
        values = self.critic(x)

        return values


model = ActorCritic(obs_dim, action_dim)
train_loader = RolloutLoader(train_env, model, episodes=10, gamma=0.95)
valid_loader = EvaluationLoader(valid_env, model, episodes=5)
step = ActorCriticStep(value_coef=0.5, gae_lambda=0.95)
trainer = Trainer(model, experiment, step)
history = trainer.fit(train_loader, valid_loader)
```

- **モデルの決まり**: モデルは方策の出力 (`forward`. 番号の行動ならスコア, 連続の行動なら平均) に加えて, 観測から状態の価値を返す `value(観測)` (形は (バッチ,) か (バッチ, 1)) を持つ (無ければエラー). 方策と価値の関数を 1 つのモデルにまとめ, 1 つの Optimizer で学習する. 勾配を流す価値は学習モードで, 利得と目標に使う価値は推論モードで勾配なしに求める (行動を選んだときと同じモード. BatchNorm の統計を進めず, Dropout の雑音を入れないため)
- **利得**: エピソードの後ろから, 1 歩の TD の誤差 (`報酬 + gamma × 次の状態の価値 - 状態の価値`) を `gamma × gae_lambda` で重ねて求める (`gae_lambda` が 0 なら 1 歩の TD, 1 ならモンテカルロ). 環境が終了したエピソードは最後の歩の後の価値を 0, 打ち切り (TimeLimit と `max_episode_steps`) のエピソードは最後の歩の後の観測の価値にする. `gamma` は `RolloutLoader` の値をバッチの `gamma` から使う. 足し合わせは, 報酬と価値の勾配を切って CPU の 64 ビットの配列にまとめ, バッチの全ての歩を後ろから 1 回で重ねて (エピソードの最後の歩で重ねを切る) から, 報酬と同じデバイスと型のテンソルに戻す (GPU で 1 歩ずつの小さな処理を並べないため)
- **損失と指標**: 学習の `loss` は, 方策の勾配の損失 (利得は `normalize` で標準化) + `value_coef ×` 価値の関数の損失 (利得と価値の和を目標にした平均 2 乗誤差) - `entropy_coef ×` エントロピー. 検証は `ReinforceStep` と同じ. 記録する指標は `train/loss`, `train/value_loss`, `train/entropy`, `train/reward`, `train/length`, `train/violations`, `valid/loss` (評価の報酬の和の平均の符号を反転した値), `valid/entropy`, `valid/reward`, `valid/length`, `valid/violations` (`success_key` を指定した場合は `success` も). 監視する指標は `trainer.monitor: valid/reward` と `trainer.mode: max` にする
- **報酬の尺度**: 価値の関数の損失は報酬の尺度の 2 乗で大きくなる. 1 歩の報酬が大きい環境 (Pendulum は -16 から 0) では, 学習の環境の報酬を縮める (Gymnasium の `TransformReward` など. 検証の環境は元の報酬のままにすると `valid/reward` が元の尺度になる) か, `value_coef` と `trainer.grad_clip_norm` で価値の関数の勾配の大きさを調える

## 22. 方策の書き出しと守り

学習した方策は, 重み, 観測と行動の空間, 守り (行動の範囲, 止める条件, 外れたときの動き) を 1 つのディレクトリ (方策の一式) に書き出す. 置き場は実験のディレクトリの `policy/` を推奨する (形と版は [docs/FORMAT.md](FORMAT.md) の「policy/」).

```python
from ml_found.rl import export_policy, load_policy, make_guards

# 台車の位置と棒の角度が範囲を外れたら, 元の制御に戻す
guards = make_guards(
    env.observation_space, env.action_space,
    observation_low=[-2.4, None, -0.21, None],
    observation_high=[2.4, None, 0.21, None],
    fallback="handover"
)
# 組み立て直すための引数を metadata に (下の「組み立て直し」)
model_args = {"obs_dim": 4, "hidden_dim": 64, "n_actions": 2}
export_policy(
    experiment.run_dir / "policy", model, env.observation_space,
    env.action_space, guards,
    metadata={
        "env_id": "CartPole-v1", "algorithm": "REINFORCE",
        "model": {"class": "model.PolicyNet", "args": model_args}
    }
)

# 読み直して, 観測から守りを当てた行動を得る
policy = load_policy(experiment.run_dir / "policy", PolicyNet(**model_args))
result = policy.act(observation)
# {"action": 0, "status": "ok", "reasons": []}
```

- **行動の範囲**: 連続の行動 (Box) は `action_low` と `action_high` に clip する (空間の範囲との狭い方). 番号で選ぶ行動 (Discrete) は `allowed_actions` の中の最もスコアの高い行動にする. どちらも `status` が `clipped` になる
- **止める条件**: 観測の要素ごとの範囲 (`observation_low`, `observation_high`. `None` は制限なし). 範囲の外の観測, 観測の形の違いと有限でない値, 数にできない観測とモデルの出力 (float にできない桁の大きな整数も), モデルの出力の大きさの違いと有限でない値, モデルのエラーは, 外れたときの動きにする
- 範囲は有限の数か `None` (制限なし) で書く. nan は守りを黙って外すことになるので, 書き出しでも読み込みでもエラーにする. 守り, 空間, 外れたときの動きの知らない項目 (`observation_hihg` のような書き損じ) もエラーにする
- 連続の行動は float32 で返す. 0.1 のような範囲は float32 に丸めると少し外に出るので, 外に出た要素は内側の隣の float32 に寄せる. float32 で無限大になる行動と観測 (1e39 など) は, 外れたときの動きにする
- 書き出しでは `policy.pt` の sha256 を `policy.json` に書き, 読むときに確かめる (違えばエラー)
- `policy.pt` はモデルの `state_dict` 全体なので, 連続の行動の方策の `log_std` と, `ActorCriticStep` で学んだ価値の関数 (`value`) の重みも入る. `head` が読むのはモデルの `forward` の出力だけで, `log_std` と `value` は行動に使わない. 読む側は使わないキーがあってよい (`load_policy` には同じ組み立てのモデルを渡すので, 全てのキーを読む)
- **外れたときの動き** (`fallback`): `stop` (行動を返さず止める. `status` が `stop`), `handover` (行動を返さず元の制御に戻す. `status` が `handover`), `action` (決めた行動. 連続の行動は既定で 0 の行動. `status` が `fallback`)
- `apply_guards(spec, observation, output)` は, モデルの出力に守りを当てるだけの純粋な関数. 推論の段階 (ML-Inference) は, この関数と `policy.json` で同じ守りを当てられる
- `load_policy` には, 書き出したときと同じ組み立てのモデルを渡す (重みは一式から読み, cpu で動かす). 一式に入るのはクラスの名前 (`model_class`) だけなので, 組み立て方は `metadata` に書く (下の「組み立て直し」)
- 返す方策は, 観測から行動と守りの結果を返すだけで, 環境や機械には触れない. 実機での実行は道具の外にあり, 方策を使ってよいかと守りの範囲は人が決める

**組み立て直し**: `model_class` は読み込んだモジュールの名前 (スクリプトの隣の `model.py` なら `model.PolicyNet`) で, それだけでは組み立てられない. `metadata` に `env_id`, `algorithm` と, `model` (`class` とコンストラクタの引数 `args`, またはモデルの定義 (ML-ModelControl) のファイルを一式に写した場所 `spec`) を書くのを推奨する (形は docs/FORMAT.md の「metadata の推奨の形」). 読む側は次のように作り直す.

```python
from ml_found.rl import load_policy, read_policy_spec

spec = read_policy_spec(policy_dir)
args = spec["metadata"]["model"]["args"]
policy = load_policy(policy_dir, PolicyNet(**args))
```

**守り付きの評価**: `evaluate_policy(policy, env_or_id, episodes, seed)` は, 読み直した方策を守りを当てたまま環境で動かし, エピソードごとの結果を返す. `stop` と `handover` では行動を返さないので, そこでエピソードを終える (止めたか元の制御に戻した扱い). `fallback` は決めた行動で続ける. 環境の名前を渡すと `make_env` で作り (追加の依存 `rl` が要る), 終えたら閉じる. 環境を渡した場合は閉じない. 返す行は `save_evaluation` にそのまま渡せる.

```python
from ml_found import save_evaluation
from ml_found.rl import evaluate_policy

rows = evaluate_policy(policy, "CartPole-v1", episodes=10, seed=20000)
# [{"episode": 0, "seed": 20000, "reward": 500.0, "length": 500,
#   "end": "truncated", "ok": 500, "clipped": 0, "fallback": 0, "stop": 0,
#   "handover": 0}, ...]
save_evaluation(experiment, "guarded_episodes", rows, "simulator",
                description="守りを当てて動かしたエピソード", guard=True)
```

- 列は `episode` (0 から), `seed` (エピソードの乱数の種. `seed`, `seed + 1`, ...), `reward` (報酬の和), `length` (環境を進めた歩数), `end` (終えた理由. `terminated`, `truncated`, `max_steps`, `stop`, `handover`), 守りの結果ごとの数 (`ok`, `clipped`, `fallback`, `stop`, `handover`. `act` を呼んだ回数で, 止めた判断も 1 回と数える)
- `max_steps` で 1 エピソードの歩数を切れる. `seed=None` は環境の乱数の続き
- 推論の段階 (ML-Inference) と画面でも, 同じ数え方 (`stop` と `handover` で終える) にする

### 評価の軌跡の表

評価で走らせた軌跡を, 1 歩が 1 行の表 (`trajectories/<名前>.csv`) と説明 (`trajectories/<名前>.json`) に書き出せる. 列はエピソードの番号, 歩の番号, 観測, 行動, 報酬, 終了と打ち切りの別, 乱数の種と, 守りを当てた評価では守りの結果と理由, 記録した方策が確率を返した場合は行動を選んだ確率 (形は [docs/FORMAT.md](FORMAT.md) の「trajectories」). ML-EDA がこの表を読み, ML-Fathom が分析する. 既定では記録も書き出しもしない.

```python
from ml_found.rl import (
    EvaluationLoader, Trajectory, evaluate_policy, save_trajectory
)

# 学習中の検証の軌跡. ベストモデルを更新した評価の軌跡を書き出す
valid_loader = EvaluationLoader(
    valid_env, model, episodes=10, seed=10000, record_trajectory=True
)


def keep_best(info: dict) -> None:
    if info["improved"]:
        save_trajectory(
            experiment, "valid_best", valid_loader.trajectory,
            epoch=info["epoch"], description="ベストモデルの検証の軌跡"
        )


trainer.add_callback(keep_best)
history = trainer.fit(train_loader, valid_loader)

# 書き出した方策を守り付きで動かした軌跡 (守りの結果の列が付く)
trajectory = Trajectory()
rows = evaluate_policy(
    policy, "CartPole-v1", episodes=10, seed=20000, trajectory=trajectory
)
save_trajectory(experiment, "guarded", trajectory, epoch=best_epoch)
```

- **記録**: `EvaluationLoader(record_trajectory=True)` は評価のたびに新しい軌跡を作り, 終えた評価の軌跡を `trajectory` に置く (前の評価のものは置き換える. 記録する前は `None`). `trainer.valid_every_n_epochs` が 2 以上では, 検証しないエポックの `trajectory` は前の検証のまま. `evaluate_policy` には何も記録していない `Trajectory` を渡す (1 回の評価に 1 つ. エピソードの番号が重ならないため). 学習の `RolloutLoader` は記録しない
- **行の中身**: 観測は行動を選ぶ前の観測, 行動は環境に渡した行動 (連続の行動は空間の範囲で clip した値. `evaluate_policy` では守りを当てた行動). `evaluate_policy` で守りが `stop` か `handover` にした判断は, 行動と報酬の空の行にする (その観測で止めたことが分かるように). `max_episode_steps` と `max_steps` で切った歩は打ち切り (`truncated`) にする (`RolloutLoader` の扱いと同じ)
- **行動を選んだ確率**: `evaluate_policy` は, 方策の `act` が `action_prob` (とった行動を選んだ確率. 0 より大きく 1 以下) を返した歩だけ, 表の `action_prob` の列に書く (確率で選ぶ方策や ε-greedy の方策を自分で書いた場合). 守りで変えた行動 (`clipped`, `fallback`) の確率も `act` が返した値のまま書くので, 環境に渡した行動の確率を返す. `load_policy` の方策と `EvaluationLoader` は最もスコアの高い行動を選ぶ (確率は常に 1 で, 他の方策の見積もりに使えない) ので書かない. 自分で記録する場合は `Trajectory.add(..., action_prob=確率)` で渡す. 範囲の外の値と, 行動の無い歩 (止めた判断) の確率はエラー
- **上限**: `Trajectory(max_rows=100000)` (`EvaluationLoader` では `max_trajectory_rows`) が貯める歩の数の上限. 記録の間も書く文字数 (見出しを含む) を足し上げ, 書き出しと同じ 64 MiB (`MAX_TRAJECTORY_BYTES`) を超えたら貯めるのを止める. どちらかの上限に達したエピソードと後のエピソードは途中までを残さずに捨てる (メモリを使い切らないため). `save_trajectory` も表を 64 MiB までのエピソードで書く. どちらで切ったかに関わらず, 入らなかったエピソードの数を説明の `dropped_episodes` に, 全てが入ったかを `complete` に書く. 観測と行動の要素の数はそれぞれ 256 まで (画像のような観測は表にしない. 超える空間と, Discrete と Box の他の空間 (Dict, Tuple など) は `EvaluationLoader` を作るときと `evaluate_policy` の始めに, 空間で分からない値は記録の最初の歩でエラー)
- **書き出し**: `save_trajectory(実験, 名前, 軌跡, epoch=None, description=None)`. 名前は評価の名前と同じ決まりで, 同じ名前で書き直すと置き換える. 説明, 表の順に置き換え, 失敗すると元の表と説明を残す. 書くのは評価の後だけで, 学習中の軌跡の記録は `last.pt` に入らない (再開で戻さない)
- CSV にしたのは, 追加の依存 (pyarrow など) なしに書け, ML-Dashboard と人がそのまま読めるため. 利用者の Parquet の表も同じ列の決まりで読む (FORMAT.md の「利用者の表」)

### 記録した軌跡からの方策の見積もり

記録した軌跡 (確率の列 `action_prob` の付いた表) から, 別の方策 (見積もる方策) のエピソードの割り引いた報酬の和の期待値を, 環境で動かさずに重点サンプリングで見積もる.

```python
from ml_found.rl import estimate_policy_value


def target_prob(observation, action) -> float:
    # 見積もる方策がその観測でその行動を選ぶ確率
    return float(new_policy_probs(observation)[action])


result = estimate_policy_value(
    experiment.run_dir / "trajectories" / "explore.csv", target_prob, gamma=1.0
)
print(result["value"], result["effective_sample_size"], result["notes"])
```

- **入力**: `save_trajectory` が書いた表 (CSV) の場所か, メモリの `Trajectory` (記録を終えたエピソードだけ). 表は隣の説明 (`<名前>.json`) があれば型と観測の形に使い, 無ければ列の名前と値から決める. 表は ML-Found の中で小さく読み (ML-EDA には頼らない. ML-Found は ML-EDA に依存しないため), 普通のファイルだけを, リンクを辿らずに 64 MiB まで読む. `target_prob(観測, 行動)` には, 1 つの値の列 (`obs`, `action`) は数 (番号の空間は整数), 番号の列は配列 (観測は説明の空間の形) を渡す
- **見積もり**: エピソードの重み `w` (歩ごとの `target_prob` / `action_prob` の積) と割り引いた報酬の和 `G` から, 主の値 `value` は重み付きの重点サンプリング (`w * G` の和 / `w` の和). 普通の重点サンプリングより偏りはあるが, 値が報酬の和の範囲に収まり, ばらつきが小さく, 記録の方策と同じ方策では記録した報酬の平均にそのまま一致するため. 比べるために, 普通の重点サンプリング (`ordinary`. `w * G` の平均. 偏りが無い) と歩ごとの重点サンプリング (`per_decision`. 各歩の報酬にその歩までの比の積を掛ける) も返す. `behavior_value` は記録した方策の報酬の和の平均
- **重みの計算**: 比の積は対数の和で求め, 最も大きい重みの対数を引いてから戻す (長いエピソードで溢れないため). 見積もる方策の確率が 0 の歩の後の重みは 0 のまま (無限大と 0 を掛けた nan にならない). 普通と歩ごとの重点サンプリングは, 元に戻した値が有限にならなければ `None` にして `notes` で知らせる (重み付きの見積もりは最大の値を引いたままで決まる). 重みの和が有限にならない場合は止まる
- **重みのばらつきの印**: 有効なサンプルの数 `effective_sample_size` (`w` の和の 2 乗 / `w` の 2 乗の和), 最も大きい重みの割合 `max_weight_share`, 見積もる方策の確率が 0 の行動で重みが 0 のエピソードの数 `zero_weight_episodes`, 最も大きい重みより非常に小さく 0 として計算したエピソードの数 `underflow_episodes`. 有効なサンプルの数がエピソードの数の 1 割より少なければ, `low_ess` を `true` にして `notes` で知らせる (画面が文を読まずに見分けるため. 全ての重みが 0 で `value` が決まらない場合は `false`)
- **止まる場合**: 確率の列の無い表 (最もスコアの高い行動で選んだ評価など), 値の数が見出しと違う行, 行動があるのに確率か報酬の無い行, nan か無限大の報酬, 範囲の外の確率 (表は 0 より大きく 1 以下, `target_prob` は 0 以上 1 以下), 計算できないほど大きな重み
- **知らせる場合** (`notes`): 見積もる方策の確率が 0 の行動を記録の方策がとった歩 (そのエピソードの重みは 0. 全ての重みが 0 なら `value` は `None`), 0 として計算した小さな重み, 守りで止めた判断の行 (行動と報酬が無いので飛ばし, そこでエピソードを終えたとする), 終わりの印の無いエピソード, 説明の `complete` が `false` の表
- 記録の方策が選ばない (確率 0 の) 行動を見積もる方策が選ぶ場合は, 表から見分けられず, 見積もりは偏る. 記録の方策は見積もる方策の選ぶ行動を全て正の確率で選ぶもの (ε-greedy など) にする
- 番号で選ぶ行動 (Discrete) の確率を使う. 連続の行動の確率密度は扱わない

## 23. config の項目

ML-Found が使う項目と既定値.

| 項目 | 既定値 | 説明 |
| --- | --- | --- |
| `model.spec` | (無し) | モデルの定義 (ML-ModelControl の定義の YAML) の場所, 相対パスの場合はプロジェクトのディレクトリから辿る. あれば `start_experiment` が実験に `model_spec.yaml` として写す (「モデルの定義の写し」). `model` の他の項目はプロジェクトのもの |
| `requires` | (無し) | 学習に要る機能の名前のリスト (「追加の依存」). `start_experiment` が出力先を作る前に確かめる. experiment で変える場合は default にも書く |
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
| `scheduler.warmup_steps` | `null` | 学習の始めに学習率を線形に上げる回数 (`interval` の単位) |
| `scheduler.warmup_ratio` | `null` | warmup の回数を学習全体のステップ数に対する割合で書く (学習を始めるときに `warmup_steps` に直す) |
| `scheduler.warmup_start_factor` | `0.01` | warmup の最初の学習率の倍率 |
| `optimizers` | `{}` | 名前付きの複数の Optimizer (「複数の Optimizer」). 名前ごとに `name` (既定 `Adam`), `params` (既定 `{lr: 0.001}`), `modules` (必須), `scheduler` (一番上の `scheduler` と同じ項目) |
| `trainer.epochs` | `100` | エポック数 (同じ `Trainer` で `fit` を呼び直した場合は, 1 回の `fit` で追加するエポック数) |
| `trainer.monitor` | `null` | ベストモデルの判定に使う指標, `null` の場合は `valid/loss`, 検証しない場合は `train/loss` |
| `trainer.mode` | `min` | 指標は小さい方と大きい方のどちらが良いか |
| `trainer.monitor_tiebreak` | `null` | 監視する指標が最良値と同じ値 (相対誤差 1e-6 以内) か, 最良値より悪くないが `min_delta` 未満の改善の場合に比べる指標 (例: `valid/loss`), `null` の場合は同じ値を改善とみなさない |
| `trainer.monitor_tiebreak_mode` | `min` | 同じ値の場合に比べる指標は小さい方と大きい方のどちらが良いか |
| `trainer.grad_clip_norm` | `null` | 勾配のノルムの上限, `null` の場合はクリッピングしない. Optimizer が更新するパラメータだけが対象 (`optimizers` の場合は Optimizer ごと) |
| `trainer.log_grad_norm` | `false` | クリップ前の勾配のノルムを `train/grad_norm` (`optimizers` の場合は `train/grad_norm_<名前>`) として記録するかどうか |
| `trainer.restore_best` | `true` | 学習後にベストモデルの重みに戻すかどうか, `false` の場合は最後のエポックの重み |
| `trainer.select_best` | `true` | 監視する指標でベストモデルを選ぶかどうか. GAN のように損失で良し悪しを決められない学習では `false` にし, `best.pt` を作らず `final_model.pt` に最後のエポックの重みを残す (Early Stopping と `monitor_tiebreak` とは一緒に使えない. 監視する指標は発散の確認と ReduceLROnPlateau に使う). 選ぶ設定の `last.pt` から `false` で再開すると, 再開元のベストモデルは使わない |
| `trainer.valid_grad` | `false` | 検証時にも勾配を計算するかどうか |
| `trainer.valid_seed` | `null` | 検証の前に乱数を固定するシード, 検証の後は学習の乱数の状態に戻す |
| `trainer.eval_train` | `false` | 学習の後に学習用データを推論モードで評価し直し, `train_eval/` を付けて記録するかどうか |
| `trainer.valid_every_n_epochs` | `1` | 検証するエポックの間隔 (1 以上の整数). エポックの番号が割り切れるエポックと予定の最後のエポックだけ検証 (`eval_train` の評価し直しも) し, ベストモデル, Early Stopping, ReduceLROnPlateau を判定する (「検証の間隔」) |
| `trainer.manual_optimization` | `false` | ML-Found が逆伝播と更新を行わず, `step` の中で更新するかどうか |
| `trainer.accumulate_grad_batches` | `1` | 何バッチ分の勾配を貯めてから 1 回更新するか (1 以上の整数. 「大きな学習済みのモデルの微調整」). 勾配を使う 1 つの Optimizer を ML-Found が更新する場合だけ |
| `trainer.precision` | `fp32` | 計算の精度, `fp32` か `bf16`. `bf16` では `step` を bfloat16 の自動の混合精度で計算する (重みの型は変えない) |
| `trainer.resume_scheduler` | `checkpoint` | 再開時に Scheduler の引数が config と違う場合に, 再開元の状態のまま (`checkpoint`) か, config の引数で作り直して同じステップ数まで進めるか (`config`) |
| `trainer.resume_from` | `null` | 再開するチェックポイントのパス, 相対パスの場合はプロジェクトのディレクトリから辿る |
| `trainer.progress_bar` | `true` | 進捗バーを表示するかどうか |
| `trainer.progress_bar_batches` | `false` | エポックの中のバッチ単位の進捗バーも表示するかどうか (1 エポックが長い場合に使う) |
| `trainer.early_stopping.use` | `false` | Early Stopping を使うかどうか |
| `trainer.early_stopping.patience` | `10` | 改善しないまま待つ回数. 判定するエポック (`valid_every_n_epochs` が 1 なら毎エポック) の回数で数える |
| `trainer.early_stopping.min_delta` | `0.0` | これより小さい変化は改善とみなさない |
| `trainer.checkpoint.save_best` | `true` | `best.pt` を保存するかどうか |
| `trainer.checkpoint.save_last` | `true` | `last.pt` を保存するかどうか |
| `trainer.checkpoint.every_n_epochs` | `1` | `last.pt` を保存するエポックの間隔, 最後のエポックと, エポックの区切りで止めたエポック (Early Stopping, `STOP`, `add_callback`) は必ず保存. `STOP` の `now` でエポックの途中で止めた場合は保存しない |
| `trainer.checkpoint.best_weights_only` | `false` | `best.pt` にモデルの重みだけを保存する (Optimizer などを含めないので軽いが, 再開には使えない) |
| `trainer.checkpoint.last_with_best` | `true` | `last.pt` にベストモデルの重みも入れる. `false` にすると軽くなり, 再開時は再開元の `best.pt` から戻す |
| `trainer.ema.decay` | `null` | 重みの指数移動平均 (EMA) の減衰率 (0 より大きく 1 より小さい数). 書くと EMA の重みで検証し, ベストモデルと `final_model.pt` も EMA の重みにする (「重みの指数移動平均 (EMA)」). `null` の場合は使わない |
| `trainer.ema.use_buffers` | `false` | EMA でバッファ (BatchNorm の統計量など) も平均するかどうか. `false` の場合は学習しているモデルの値を写す |
| `trainer.checkpoint.trainable_only` | `false` | `last.pt` と `best.pt` のモデルの重みを, 学習するパラメータとバッファだけにする (凍結した重みを省いて軽くする. 再開では今のモデルの値で補う). `final_model.pt` は全ての重み |
| `trainer.init_weights.path` | `null` | 学習を始める前にモデルに読む重みのファイル (`state_dict` か ML-Found のチェックポイント), 相対パスの場合はプロジェクトのディレクトリから辿る. `null` の場合は読まない |
| `trainer.init_weights.source_prefix` | `""` | ファイルのキーのうち, この前置きで始まるものだけを前置きを外して読む |
| `trainer.init_weights.target_prefix` | `""` | 外した後のキーの頭に足す前置き (モデルの一部に読む場合) |
| `trainer.init_weights.strict` | `false` | モデルの全ての重みがそろい, 使わない重みも無いことを求めるかどうか |

ReduceLROnPlateau は `trainer.monitor` の値で学習率を下げる.
`scheduler.interval` を `step` にした場合や, `scheduler.params.mode` を `trainer.mode` と違う値にした場合はエラーになる.
`mode` を省略した場合は `trainer.mode` と同じになる.

監視する指標が nan や inf になった場合はエラーで止まる.
監視する指標が最良値とちょうど同じ値の場合は改善とみなさないので, 正解率のように同じ値を取りやすい指標を監視すると, ベストモデルは最初にその値に達したエポックのままになり, Early Stopping も止まりやすい. `trainer.monitor_tiebreak` に `valid/loss` などを指定すると, 同じ値の場合はその指標で比べ, ベストモデルの判定と Early Stopping の両方に使う. バッチごとの平均の丸め誤差で同じ正解率がわずかに違う値になることがあるので, 相対誤差 1e-6 以内は同じ値とみなす. Early Stopping では, 最良値より悪くないが `min_delta` 未満の改善の場合も比べる指標で判定する (`min_delta` は Early Stopping だけに使い, ベストモデルの判定には使わない). 比べる指標が nan や inf になった場合もエラーで止まる. 再開するときは, 比べる指標と向きを再開元と同じにする (違う場合はエラー). この設定を入れる前のチェックポイントから再開した場合は, 次に監視する指標が改善するまで比べない.
学習の途中で起きたエラーは `train.log` にも記録する. Optimizer と Scheduler のクラス名と引数 (名前, 必須の引数, 範囲を持つ値), `scheduler.interval`, ReduceLROnPlateau の組み合わせ, デバイス, `trainer.epochs` (1 以上の整数), `trainer.mode`, `trainer.checkpoint.every_n_epochs`, `trainer.valid_every_n_epochs` (1 以上の整数), `trainer.monitor_tiebreak_mode`, `manual_optimization` と `grad_clip_norm`・`log_grad_norm` の組み合わせ, `select_best: false` と Early Stopping・`monitor_tiebreak` の組み合わせの誤り, `trainer.resume_scheduler` の値, `eval_train: false` で `train_eval/` の指標を監視する設定, `auto` と書けない引数の `auto` と, 存在しないか再開に使えない `resume_from` は, 出力先を作る前に見つかる. 範囲を持つ引数が配列 (Adam の `betas` など) の場合は要素ごとに確かめる. 範囲を持たない引数や自作のクラスの値の誤りも, 仮のパラメータで作ってみて見つける. それ以外の `Trainer` の作成のエラー (モデルをデバイスに移せない, 実際のパラメータでだけ起きる Optimizer の誤りなど) は, 出力先が作られた後なので, エラーを `train.log` に書き, `run.json` を `failed` にしてから投げ直す (継承したクラスが `super().__init__()` の後に行う処理のエラーは対象外). 作成の中身は `_build_parts` にあるので, 継承したクラスではこの名前を使わない. 自分で作った Optimizer や Scheduler を `Trainer` に渡す場合も, config の `optimizer` と `scheduler` の名前と組み合わせは確かめるので, 使わない設定も正しい値にしておく.

`last.pt` と `best.pt` には, モデル, Optimizer, Scheduler, 記録の全て (列ごとの数の配列で持つので, エポックが多くても保存は軽い), 乱数の状態が入り, `last.pt` にはベストモデルの重みも入る (`best.pt` では今のモデルがベストモデルなので, 同じ重みを 2 つは持たない. `trainer.ema.decay` を使う場合と `eval_model` を上書きした場合は別の重みなので持つ). `trainer.ema.decay` を使う場合は EMA の状態 (`ema`) も入り, モデルの重みの分だけ重くなる. 凍結した重みの多い微調整では, `trainer.checkpoint.trainable_only: true` で学習しないパラメータの重みを省ける (「大きな学習済みのモデルの微調整」). 1 エポックが短い学習 (フルバッチなど) では保存が学習時間の大部分を占めることがあるので, `trainer.checkpoint.every_n_epochs` で `last.pt` の間隔を空けるか, 再開しない場合は `save_best` と `save_last` を `false` にする. `best.pt` は改善するたびに保存するので `every_n_epochs` では減らない (ベストモデルの重みは `last.pt` にも入るので, 再開だけなら `save_best: false` でもよい. ただし `last_with_best: false` と合わせると再開後にベストモデルに戻せないので, `start_experiment` で警告が出る). 大きなモデルでは 1 ファイルが数 GB になるので, 再開に使わない `best.pt` は `best_weights_only: true` で重みだけにし, `last_with_best: false` で `last.pt` からベストモデルの重みを外すと軽くなる (100 万個の引数のモデルと Adam で, 既定の `best.pt` 12 MB と `last.pt` 16 MB が, それぞれ 4 MB と 12 MB). 重みだけの `best.pt` は `weights_only` (true), `epoch`, `monitor`, `best_value`, `model` (重み) の辞書なので, `torch.load(...)["model"]` で読む (`trainer.ema.decay` を使う場合は EMA の重み, `eval_model` を上書きした場合はそのモデルの重み). 継承して加えた状態 (`extra_state_dict`) は入らない.

`best.pt` の保存の間隔の設定は持たない. ML_tester の 06 と同じ形のフルバッチの学習 (1 次元の関数を近似するパラメータ数 8,513 の MLP, Adam, 5000 エポックのうち 2438 回改善, 1 エポックは約 1 ms, Apple M3 の cpu, PyTorch 2.15.0) を設定ごとに 3 回ずつ回し, `train.log` の学習時間を比べた. 保存が占める割合は, 保存しない設定 (`save_best` と `save_last` が `false`) との平均の差を, その設定の平均の学習時間で割った値.

| 設定 (`trainer.checkpoint`) | 学習時間の平均 (最小から最大) | その設定の学習時間のうち保存が占める割合 |
| --- | --- | --- |
| `save_best: false`, `save_last: false` | 4.8 秒 (4.5 から 5.2 秒) | - |
| `every_n_epochs: 5000`, `best_weights_only: true` | 6.0 秒 (5.3 から 6.6 秒) | 約 2 割 (`best.pt` の 1 回の保存は約 0.5 ms) |
| `every_n_epochs: 5000` | 7.8 秒 (7.0 から 8.5 秒) | 約 4 割 |
| 既定 | 14.2 秒 (13.7 から 14.8 秒) | 約 7 割 |

重みだけの `best.pt` の保存は学習時間の大部分を占めないので, 間隔を空けるより, `best.pt` が常にベストモデルの重みを持つことを優先する. ただし 1 エポックが ms 級で毎エポック改善する学習では, 重みだけの `best.pt` でも約 2 割の上乗せがあり, 省く場合は `save_best: false` にする (ベストモデルの重みは `last.pt` に入る).

`train.log` のモデルの構造は `print(model)` の内容なので, 中身を `nn.Module` の子として持たない層 (外部ライブラリの層など) は構造が残らない. そのため, 学習を始めるときにパラメータとバッファの一覧を `model_summary.csv` (名前, `parameter` か `buffer`, 持っている層のクラス名, 形, 型, 要素数, 学習するか, 重みを共有している場合は最初の名前) に保存し, 学習するパラメータ, 学習しないパラメータ, バッファの要素数を `train.log` に書く. `nn.Module` のパラメータやバッファとして持たない値 (外部ライブラリの回路の設定など) は残らないので, 再現に必要な情報は, モデルに `extra_repr` を書いて出すと良い.

`experiment.device: auto` は小さなモデルでも GPU を選ぶ (Apple Silicon の Mac では CPU 版の PyTorch でも `mps`). パラメータが少ないモデルや, 中で cpu を使うモデル (外部ライブラリの計算など) は `cpu` を明示した方が速いことがある.

## 24. 出力

実験ごとに `<output_dir>/<name>/<日付>/<時刻>/` が作られる.
他のシステム (ML-Dashboard など) と受け渡すプロジェクトと実験のディレクトリの形と, その版の決まりは [docs/FORMAT.md](FORMAT.md) にまとめた.

| ファイル | 内容 |
| --- | --- |
| `config.yaml` | 学習に使った config 全体, default として読み込めば同じ条件で学習できる. 再開した実験の場合は `trainer.resume_from` も残っているので, 最初から学習し直す場合は消す |
| `diff.yaml` | default から変えた項目 |
| `source.yaml` | 実験の起動のしかた. 形式の版 (`format_version`), 実行したスクリプト (`script`), default と experiment の config (`default`, `experiment`), 「項目=値」の上書き (`overrides`), モデルの定義の写し (`model_spec`. 場所 `path`, 写したか `status`, 中身の `sha256`). パスはプロジェクトの中ならプロジェクトからの相対パス. `load_config`, `load_project_config`, `load_config_from_args` で作った config を `start_experiment` に渡すと記録され, 別の方法で作った config (写したものを含む) では `default`, `experiment`, `overrides` が `null`. ノートブックの中と `python -c` では `script` が `null`. コマンドラインの他の引数は, 秘密の値を含み得るので記録しない |
| `train.log` | 差分, 環境の情報, モデルの構造, エポックごとの指標, 結果 |
| `model_summary.csv` | モデルのパラメータとバッファの一覧 (名前, 種類, 持っている層のクラス名, 形, 型, 要素数, 学習するか, 共有している場合は最初の名前 `same_as`. 要素数の合計では共有したものを 1 回だけ数える) |
| `model_spec.yaml` | config の `model.spec` の場所にプロジェクトの中の普通のファイルがあり, 写せた場合のみ. 学習を始めたときのモデルの定義の写し (「モデルの定義の写し」). 同じモデルで起動し直すときは, これをプロジェクトの定義の場所に写す |
| `git_diff.patch` | 未コミットの変更がある場合のみ, commit id からの差分. 未追跡のファイルも含むが, `.gitignore` で除外したファイルと submodule の中の変更は含まない. バイナリファイルは変更されたことだけが記録される |
| `history.csv` | エポックごとの指標と, エポックを始めた時点の学習率 (`lr`, `optimizers` の場合は `lr_<名前>`. エポックの中の変化は `lr_steps.csv`), `step` の `extra` で返した値 |
| `lr_steps.csv` | Scheduler を使い `scheduler.interval: step` (`optimizers` の場合は名前ごとの `scheduler.interval: step`) の場合だけ (エポックごとに変える設定で再開した場合は保存しない). 更新ごとの通しの番号 (`step`), エポック (`epoch`), その更新に使った学習率 (`lr`, `history.csv` と同じく最初のパラメータのグループの値). 学習を終えたときに保存し, 再開しても続きから記録する |
| `<指標>_curve.png` | 指標ごとの学習曲線, 学習率の推移 (`lr_curve.png`, `optimizers` の場合は `lr_<名前>_curve.png`. エポックを始めた時点の値) も含む |
| `lr_steps.png` | `lr_steps.csv` の学習率の推移 (横軸は更新の番号) |
| `best.pt` | ベストモデルのチェックポイント (`best_weights_only: true` の場合は重みだけ. `select_best: false` の場合は作らない) |
| `last.pt` | 最後のエポックのチェックポイント, ベストモデルの重みも含む (`last_with_best: false` の場合は含まない). `trainer.ema.decay` を使う場合は EMA の状態も含む. 再開に使う |
| `final_model.pt` | ベストモデルの重みだけ (`restore_best: false` か `select_best: false` の場合は最後のエポックの重み. `trainer.ema.decay` を使った場合は EMA の重み, `eval_model` を上書きした場合はそのモデルの重み), cpu のテンソル |
| `run.json` | 実験の状態 (`running`, `completed`, `stopped`, `failed`), 開始, 更新, 終了の時刻 (タイムゾーン付き) と所要時間 (`duration_sec`), 予定と終えたエポック数 (`fit` を呼び直した場合は前の `fit` までを含めた数), 再開した最初のエポック (`fit` を呼び直した場合は続きの最初のエポック), 監視する指標とその向き (`mode`), ベストモデルを選ぶか (`select_best`. `false` の場合はベストのエポックと値が `null`), ベストのエポックと値, Early Stopping で止めたエポック, `final_model.pt` の重み (`best`, `last`, エポックの途中で止めて戻せなかった場合 (`eval_model` を上書きし, そのモデルの状態を `extra_state_dict` で保存していない場合を含む) は `partial`), 乱数シード, デバイス, Optimizer ごとの名前 (名前付きの場合), クラスの名前, 受け持つ部分 (名前付きの場合), Scheduler のクラスの名前 (warmup を付けた場合は `SequentialLR(LinearLR, CosineAnnealingLR)` のように中のクラスも. `fit` を呼ぶまで作らない Scheduler は学習を始めるときに書く) と主な引数 (`scheduler_params`) (`optimizers`. `Trainer` の作成で失敗した場合は `null`), プロセスの ID とホスト名, エラーの内容, 形式の版. 学習を始めたときに作り, エポックごと (1 秒以上の間隔. 間隔の中の更新は次の進み具合の書き込みと一緒に書く), 再開したときと終了時に更新. `Trainer` の作成で失敗した場合と, `start_experiment` の後に `fit` の外 (データの読み込みやモデルの作成など) で処理されなかったエラーでスクリプトが止まった場合も `failed` (キーボードの割り込みは `stopped`) で書く |
| `task.yaml` | 任意. 何の実験か (受け持ち, 出力の意味, 正解の種類と出どころ, 作り方, 元にしたもの, データセット). config の `task` か `write_task` で書く (「何の実験か, データ全体の評価, 見本」) |
| `evaluations/` | 任意. `save_evaluation` で書いたデータ全体の評価 (`<名前>.json`) |
| `samples/` | 任意. `save_sample` で書いた見本と一覧 (`index.json`) |
| `trajectories/` | 任意. `save_trajectory` で書いた強化学習の評価の軌跡の表 (`<名前>.csv`) と説明 (`<名前>.json`) (「評価の軌跡の表」) |
| `progress.json` | 学習中だけ置かれる, 今のフェーズ (`train`, `train_eval`, `valid`), エポックと予定のエポック数 (`fit` を呼び直した場合は前の `fit` までを含めた数), エポックの中で終えたバッチの数と総数, 段階 (`stage`. `on_epoch_end` を継承した場合, その処理の間だけ `epoch_end`. 高価な評価の間を画面に知らせる), 経過時間, 更新時刻 (`updated_at`). 1 秒以上の間隔で上書きする (1 エポックが短い学習の邪魔にならないように, フェーズの始めと終わりも間隔を守る). 学習を終えると消す |

### エポックごとの結果を受け取る

Python のコードでエポックごとの結果を受け取る場合は, `trainer.add_callback(fn)` で関数を登録する. 関数は, 各エポックの記録を保存する前に, 次の辞書を受け取る.

- `epoch`: 終えたエポック
- `record`: そのエポックの記録の写し (`history.csv` の 1 行と同じ. 書き換えても記録は変わらない)
- `improved`: ベストモデルを更新したか
- `best_epoch`, `best_value`: これまでのベストモデルのエポックと値 (ベストモデルを選ばない設定では `None`)
- `early_stopping`: このエポックで Early Stopping が止めるか
- `judged`: このエポックで検証し, ベストモデルなどを判定したか (`trainer.valid_every_n_epochs` が 2 以上の場合, 判定しないエポックの `record` には `valid/` の指標が無い)

関数が真の値 (`True`, numpy の真偽値など) を返すと, `STOP` のファイルと同じく, そのエポックで学習を止める (`last.pt` を保存し, 状態は `stopped`. 予定の最後のエポックなら `completed`). 同じエポックで Early Stopping も止める場合は, Early Stopping として記録する. 関数の中のエラーは学習のエラーとして扱う (状態は `failed`). 呼ばれるのはそのエポックのチェックポイントを保存する前なので, 関数の中から読む `last.pt` と `best.pt` は前のエポックまでのもの. 関数の中の処理の間, `progress.json` の段階は `epoch_end` にならない (高価な評価は `on_epoch_end` を継承して書くと, 画面に評価中と出る).

```python
def notify(info: dict) -> bool:
    print(info["epoch"], info["record"]["valid/loss"])
    # 検証用の損失が十分に下がったら止める
    return info["record"]["valid/loss"] < 0.01


trainer.add_callback(notify)
history = trainer.fit(train_loader, valid_loader)
```

### 学習の外からの確認と中断

`run.json` と `progress.json` は, 書きかけを読まないように一時ファイルに書いてから置き換える. ディスクまで書き出す (fsync) のは `run.json` の開始と終了のときだけ. ダッシュボードや別の端末のスクリプトから, 学習中の状態と進み具合を読める. 書き込めない場合 (ディスクの不足など) は `train.log` に警告を 1 回残し, 学習は続ける. 状態は `Trainer.fit` が書くので, `fit` を使わずに `train_epochs` などを直接呼んだ場合は `running` のまま終わりが残らない (下の処理されなかったエラーは記録する).

学習中の実験のディレクトリに `STOP` という名前のファイルを置くと (中身が `now` 以外の場合), そのエポックを終えたところで学習を止める. 結果 (`final_model.pt` など) を保存し, 状態を `stopped` にする. `checkpoint.save_last` が true の場合は, 保存の間隔に関係なく `last.pt` も保存するので, そこから再開できる. 1 エポックが長い学習ですぐに止めたい場合は, 下のように `STOP` に `now` と書く. 予定の最後のエポックで置いた場合と, Early Stopping と同じエポックで止まった場合は `completed` になる. `STOP` は学習を止めた後も残るので, 同じ出力先で `fit` をやり直す場合は消しておく (同じ `Trainer` で `fit` を呼び直す場合は, ML-Found が消す). エポックの区切りの確かめは, 毎エポック行うのでファイルを開かずにリンクを辿らずに調べるだけにし, リンク, ディレクトリ, FIFO の `STOP` も止める要求とする (開かないので, FIFO でも止まらず, リンクの先も読まない).

`STOP` に `now` と書くと, エポックの途中でも止める (バッチを終えるたびに, 前に確かめてから 1 秒以上空いていれば確かめる). 中身は `load_evaluations` と同じ読み方で, リンクでない普通のファイルの 1 KiB までだけを読む. リンク, 普通でないファイル, 1 KiB を超えるファイルは中身を読まず, 中身の無い `STOP` と同じくエポックの区切りで止める. 途中のエポックは捨てて記録に加えず, 最後に終えたエポックの状態に戻す. そのエポックの最後のフェーズ (検証, 検証しない場合は `eval_train` の評価し直しか学習. 判定しないエポックでは学習) の最後のバッチで気付いた場合は, 中身の無い `STOP` と同じくエポックの区切りで止める.

- 前のエポックの `last.pt` (学習を始めたエポックなら始めた時点) から, モデル, Optimizer, Scheduler, 更新回数 (`global_step`), 乱数, 継承したクラスが加えた状態 (`extra_state_dict`) をメモリの上でも戻す. `last.pt` は, この `fit` で前のエポックに保存したもの (エポックと更新回数が合うもの) だけを使う. ただし学習の後に, 設定に応じてモデルの重みはベストモデルや `eval_model` の重みに置き換わる. 続きを学習する場合は, 同じ `Trainer` で `fit` を呼び直すか (「学習の再開」), `last.pt` から再開する
- `final_model.pt` は, `restore_best` と `select_best` でベストモデルがあればその重み. そうでなければ戻した重み. `every_n_epochs` などで前のエポックの `last.pt` が無い場合と, `eval_model` を上書きしてそのモデルの状態を `extra_state_dict` で保存していない場合 (戻した後もそのモデルの重みが変わらない場合) は, 途中までの重みが残り, 警告が出て `run.json` の `final_weights` が `partial` になる
- `last.pt` は書かないので, 再開は最後に保存した `last.pt` から. `every_n_epochs` が 2 以上だと, 再開できる `last.pt` が無いこともある
- 状態は予定の最後のエポックでも `stopped` になる. 最初の `fit` の 1 エポック目の途中で止めると記録は 0 件になる
- `train_epochs` を上書きしたクラスでは, エポックの途中で止めるための例外が `fit` まで抜けて `failed` になるので, `now` は使わない

```bash
touch outputs/wine/2026-11-12/10-00-00/STOP
echo now > outputs/wine/2026-11-12/10-00-00/STOP   # エポックの途中でも止める
```

キーボードの割り込み (Ctrl-C) で止めた場合も `stopped`, エラーで止まった場合は `failed` になる.

`start_experiment` の後に `fit` の外で起きて処理されなかったエラー (データの読み込みの失敗など) も, `train.log` に traceback を残し, `run.json` を `failed` (キーボードの割り込みは `stopped`) で書く. `start_experiment` が `sys.excepthook` を元のものに繋いで置き換えて記録するので, スクリプトを変える必要は無く, エラーの表示と終了コードは変わらない.

- 記録するのは, `run.json` が無い場合 (学習を始める前に止まった場合. `load_evaluations` と同じ読み方で読み, リンク, 普通でないファイル, 1 MiB を超えるファイルも無いものとする) と `running` のままの場合 (`fit` を使わずに `train_epochs` などを直接呼んだ場合) だけ. `fit` と `Trainer` の作成が記録したエラーと, `fit` で終えた実験 (学習の後のテストデータの評価で起きたエラーなど) は書き換えない. `train_epochs` を直接呼んだ実験は終わりが記録されないので, 学習を終えた後のエラーでも `failed` になる
- 同じプロセスで `start_experiment` を何度も呼んだ場合は, 最後に始めた実験に記録する
- ノートブックと IPython のセルで起きたエラーは, カーネルが動き続けて `sys.excepthook` を通らないので記録しない (`run.json` が無い実験はダッシュボードで未完了に見える). `sys.exit()` での終了と, メインでないスレッドで起きたエラーも記録しない. 後から別のライブラリ (traceback の表示を変えるものなど) が `sys.excepthook` を元のものに繋がずに置き換えた場合も記録しない

プロセスが強制終了された場合は `running` のまま残るので, 同じ端末であれば `run.json` の `hostname` と `pid` のプロセスが動いているかで見分ける. 別の端末からは, `progress.json` の `updated_at` が長く更新されていないかで見分ける.

## 25. 安全

- チェックポイント, 重み, 方策の一式の重み, 始める前に読む重み (`trainer.init_weights.path`) は `torch.load(weights_only=True)` で読む (任意のコードを動かせる `pickle` を読まない)
- config は OmegaConf で, ML-EDA の `dataset.yaml` と `task.yaml` は `yaml.safe_load` で読む
- Optimizer と Scheduler の名前に `.` を含めると, そのモジュールを import して自作のクラスを読む. config は学習のスクリプトと同じく, 信頼できるものだけを使う
- 評価, 見本, 軌跡の名前は英数字と `_ . @ -` だけ (先頭は `.` 以外). 名前から実験のディレクトリの外にファイルを書かない. `trajectories/` がリンクなら軌跡を書かない
- 評価の軌跡の表を読む側 (ML-EDA, 画面, ML-Found の `estimate_policy_value`) は, 普通のファイルだけを, リンクを辿らずに, 表は 64 MiB (`MAX_TRAJECTORY_BYTES`), 説明は 1 MiB (`MAX_TRAJECTORY_INFO_BYTES`) までだけ読む (FORMAT.md の「trajectories」)
- `source.yaml` には, コマンドラインの他の引数を残さない (秘密の値を含み得るため)
- config の参照の誤りの文には, 解決した値と OmegaConf の文を入れない (環境変数の値などの秘密を出さないため). ただし `config.yaml` は解決した値で保存するので, 環境変数の参照 (`${oc.env:...}`) で config に秘密の値 (トークンなど) を入れない. 秘密の値は学習のスクリプトで環境変数から読む
- モデルの定義の写し (`model_spec.yaml`) は, プロジェクトの中の普通のファイル (16 MiB まで) だけを, リンク (途中のディレクトリ, リンクの後の `..`, Windows のジャンクションを含む) を辿らずに読む. プロジェクトの外とリンクは写さない. `train.log` には場所を `repr` で書く (改行で偽の行を作らないため)
- 実験のディレクトリのファイルは, 普通のファイルだけを, リンクを辿らずに大きさの上限までだけ読む. FIFO などの普通でないファイルは開かない. 対象と上限は, `load_evaluations` の `evaluations/*.json` (20 MiB), `load_samples` と `save_sample` の `samples/index.json` (20 MiB), `load_task` の `task.yaml` (1 MiB), 処理されなかったエラーの記録の `run.json` (1 MiB), 学習中の `STOP` の中身 (1 KiB. `STOP` があるかはリンクを辿らずに開かずに調べる). `evaluations/` と `samples/` 自体がリンク (Windows のジャンクションを含む) なら中を読まず, `samples/` には見本を書かない
- ML-Found はネットワークを使わない. 追加の依存の入れ方は, コマンドと公式の手順の URL を示すだけで, 自分では入れない
