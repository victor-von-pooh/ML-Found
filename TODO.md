# TODO

このリポジトリでまだやっていないことを, 優先度の順に並べる. 終えたら消す.
優先度: 高 = 「表の予測を画面だけで通す」に要るものか, リポジトリの間の食い違い. 中 = 「点群と領域を画面で通す」に要るもの. 低 = その先. 条件待ち = 実例か利用者の判断が要る.

## 優先度 高

### 追加の依存の登録簿に段階のライブラリを足す
- 何を: 追加の依存の登録簿に, ML-Quest が書き出す骨組みの学習のスクリプトが使う ML-EDA (`ml_eda`), ML-ModelControl (`ml_modelcontrol`), ML-Inference (`ml_inference`) を足し, 入れ方を書く
- なぜ: 骨組みの学習のスクリプトはこの 3 つを使うが, 登録簿に無いので `describe_requirements` で確かめられず, ML-Dashboard が入れ方を案内できない. 足りない環境では学習の途中の import の誤りで止まる
- 終わりの条件: 3 つが `describe_requirements` に出て, 入っていない環境で `require` が入れ方を示して止まる. 名前は ML-Quest の要る機能の名前と同じ
- つながり: ML-Quest の「骨組みの config に要る機能を書く」, ML-Dashboard の「学習の環境を画面から作る」

## 優先度 中

無し

## 優先度 低

### 画面が使う確かめの関数を公開する
- 何を: ML-Dashboard が今は内部のモジュールから読んでいる関数を公開の関数にし, `__all__` と文書に入れる. 実験を始める前の確かめ (`start_experiment` が出力先を作る前にする, 名前, 再開のパス, trainer, Optimizer と Scheduler の確かめ. 今は `check_name`, `check_resume_path`, `check_trainer_config`, `check_optim_config`, `missing_data_size_params`, `try_build_optim`) は 1 つの公開の関数にまとめる案. 他は, 名前付きの Optimizer の扱い (`named_views`, `uses_named_optimizers`), Optimizer と Scheduler のクラスの解決 (`resolve_optimizer_class`, `resolve_scheduler_class`), 引数の確かめ (`check_optim_params`, `check_param_values`), config の上書きの読み方 (`parse_overrides`, `params_sections`), 見本の索引の読み込み (`read_samples_index`), 方策の重みのハッシュ (`file_sha256`)
- なぜ: 公開の口が無いので ML-Dashboard が内部のモジュールを読み, ML-Found の中を変えると画面が壊れうる
- 終わりの条件: ML-Dashboard が要る確かめと読み込みが全て公開の関数にあり, 公開の名前をテストで確かめる
- つながり: ML-Dashboard の「ML-Found の内部のモジュールを使わない」

### 大きな学習済みのモデルの微調整の補助
- 何を: 勾配の蓄積, 混合精度 (bf16 から), 学習するパラメータだけの保存, 重みの一部の読み込みの補助を config の項目にする
- なぜ: 今は学習のスクリプトか継承したクラスで書く. 学習済みのモデルの微調整 (問題の設定の作り方「微調整」) で毎回同じ手間がかかり, チェックポイントも凍結した重みの分だけ重い
- 終わりの条件: config の項目だけで勾配の蓄積と bf16 の学習ができ, 再開と通しの学習が一致する. 学習するパラメータだけのチェックポイントから再開できる. mps での効果を確かめて文書に書く
- つながり: ML-ModelControl (凍結と外部の重み), ML-Dashboard の「学習の起動に混合精度と勾配の蓄積の欄」

### EMA を config で
- 何を: `trainer.ema_decay` のような設定で, EMA の重みの更新, 保存, 再開, 検証に使うモデルの切り替えまで ML-Found が行う
- なぜ: 今は `update_parameters`, `extra_state_dict`, `load_extra_state_dict`, `eval_model` を継承して利用者が書く. 拡散のモデルなどで毎回同じ書き方になる
- 終わりの条件: 設定だけで EMA の重みで検証し, `final_model.pt` が EMA の重みになり, 再開しても通しの学習と一致する
- つながり: 無し

### 毎エポック改善する学習の保存の時間
- 何を: 1 エポックが短く毎エポック改善するフルバッチの学習で, `every_n_epochs` と `best_weights_only: true` を合わせたときの保存の時間を測り, 足りなければ `best.pt` の保存の間隔を設定にする
- なぜ: `best.pt` は改善のたびに保存するので, 保存が学習の時間の大部分を占めることがある. 記録を `history.csv` に追記する形は再開の扱いが変わるので取らない
- 終わりの条件: 測った結果で, 設定を足すか今の設定で足りるかを決め, 文書に書く
- つながり: 無し

### 強化学習の連続の行動と価値の関数を使う方法
- 何を: 連続の行動 (Box) の方策の学習と, 価値の関数を使う方法 (Actor-Critic, PPO など) の loader と step を足す
- なぜ: 今の `RolloutLoader` と `ReinforceStep` は番号で選ぶ行動と REINFORCE だけで, 他は利用者が loader と step を書く
- 終わりの条件: 連続の行動の環境 (Pendulum など) で方策を学習し, 守りと一緒に書き出して `evaluate_policy` で確かめる例が動く
- つながり: ML-Quest の「型ごとの骨組み」

### 自分で渡した Scheduler の引数を出力先に残す
- 何を: `Trainer` に自分で作って渡した Scheduler の主な引数を, 出力先 (`run.json` か `config.yaml` の横) に残す
- なぜ: 今は `config.yaml` の `scheduler` が `name: null` のままで, クラスの名前と状態は `run.json` と `train.log` にしか残らない. 同じ条件で学習し直すときに引数を探す手間がかかる
- 終わりの条件: 渡した Scheduler の引数が, 画面 (ML-Dashboard) で読める形で出力先に残る
- つながり: ML-Dashboard (実験の設定の表示)

### 追加の依存の版の確かめ
- 何を: 要る版がある機能に, `importlib.metadata` で import せずに版を確かめる確かめ方を足す
- なぜ: 今の確かめ方はモジュールがあるかだけを見て版を見ないので, 古い版は使うときのエラーになる
- 終わりの条件: 版の下限を書いた機能で, 古い版が入っていると `require` が入れ方を案内して止まる
- つながり: ML-Dashboard (入れ方のカード), ML-Quest (要る機能の受け渡し)

### 強化学習の環境の状態の再開
- 何を: `resume_from` での再開で, 環境の状態と環境の乱数も戻す
- なぜ: 今はモデルと Optimizer は戻るが, 環境の状態と乱数は戻らないので, 再開した学習は通しの学習と一致しない
- 終わりの条件: 状態を保存できる環境で, 再開した学習が通しの学習と一致する
- つながり: 無し

## 条件待ち

### 出力のファイルの権限
- 何を: `run.json` と `progress.json` の権限 (一時ファイルから置き換えるので 0600) を, `config.yaml` と同じ (umask に従う) にそろえるかを決める
- 待っているもの: 画面 (ML-Dashboard) を学習と別のユーザーで動かす実例

### モデルの定義を実験に写す
- 何を: 学習に使ったモデルの定義 (ML-ModelControl) を, ML-Found が実験の出力先に写すか. 今は学習のスクリプトが `model_spec.yaml` を写し, 同じモデルで起動し直すときは利用者がそれを `models/` に戻す
- 待っているもの: 定義の置き場とプロジェクトの結び方の判断 (利用者の判断. ML-Dashboard の「この設定でもう一度起動するときのモデルの定義」, ML-CreateArchitecture の「学習のプロジェクトとの結び方」, ML-ModelControl の「ML-Found との連携の仕組み」と合わせて決める)

### 記録した経験からの方策の見積もり
- 何を: 記録した経験 (軌跡) から, 環境で動かさずに方策の良さを見積もる (重点サンプリングなど)
- 待っているもの: 記録した軌跡のデータセット (ML-EDA の「型ごとのデータの約束」) と, それを使う実例

### 型ごとの指標の共有
- 何を: NDCG, Qini, Chamfer, FID のような型ごとの指標の計算を, 共有の場所に置く
- 待っているもの: 2 つ以上の段階で同じ計算が要る実例. それまではプロジェクトに書く

### 追加の依存の登録簿に機能を足す
- 何を: 問題の設定が要る機能 (学習済みのモデルの重みの取得など) を登録簿に足す
- 待っているもの: その機能が要る実例 (ML-Quest の要る機能の答え)
