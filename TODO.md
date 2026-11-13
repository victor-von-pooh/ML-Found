# TODO

今後対応する予定の項目をまとめる.

## 前提

ML-Dashboard (別のリポジトリ) は, ML-Found で行った実験の結果を確かめ, 学習を起動する Web アプリである.
ML-Found はその下で動く学習ライブラリとして, ダッシュボードから使える API (実験の状態の書き出し, 登録簿, config の確認, 中断など) を提供するところまでを担当する.
画面, 入力フォーム, データの取り込み, タスクの雛形などはダッシュボード側で作り, ML-Found には入れない.

## 項目

### 1. Optimizer と Scheduler の登録簿 (対応済み)

`describe_optimizers()` と `describe_schedulers()` で, `torch.optim` の Optimizer と Scheduler ごとの引数の種類, 既定値, 必須か, 説明, 範囲を返すようにした. 引数は signature から取り出し, よく使うものには説明と範囲を付けた. 範囲の端を含まない場合は `exclusive_minimum` と `exclusive_maximum` で表す. Muon と SparseAdam は, そのままでは使えないことを理由付きで返す. 他の Scheduler を組み合わせるものと関数を受け取るもの (`SequentialLR`, `ChainedScheduler`, `LambdaLR`, `MultiplicativeLR`) は返さない.
`start_experiment` は, config の引数の名前の typo, 必須の引数の不足, 範囲の外の値を確かめ, 仮のパラメータで Optimizer と Scheduler を作ってみて (範囲を持たない引数や自作のクラスの値の誤りも), 出力先を作る前にエラーにする.
学習全体の長さの引数 (`CosineAnnealingLR` の `T_max` など) は `auto` と書け, 登録簿の説明にもそのことを書く. Optimizer の仕様には, closure が必要か (`requires_closure`) も返す.

プロジェクト側で作った Optimizer は `register_optimizer` で, 名前と説明, 引数の範囲を登録できる.

### 2. closure を渡す更新手順への対応 (対応済み)

`update_parameters` が, `step` に closure が必要な Optimizer (LBFGS など) を自動で見分け, 同じバッチで損失と勾配を計算し直す closure を渡すようにした. closure が必要かは, クラスの属性 `requires_closure` (自作の Optimizer が宣言), 登録簿の仕様 (`describe_optimizers()` の `requires_closure`), `step` のシグネチャの順に決める.
あわせて `trainer.manual_optimization` で, ML-Found が逆伝播と更新を行わない設定も加えた.

勾配を使わない最適化法 (Nelder-Mead 法など) は, Optimizer がクラスの属性 `uses_gradients = False` で宣言すると, 逆伝播をせず損失だけを求める closure を渡す. 登録簿の仕様にも `uses_gradients` を返す. ML-Found 自身は勾配を使わない Optimizer を持たない (プロジェクトで作る).

### 3. 進捗を渡す仕組みと学習の中断 (対応済み)

実験のディレクトリに `run.json` (実験の状態) と `progress.json` (進み具合) を書き出し, `STOP` のファイルを置くとエポックの区切りで, `STOP` に `now` と書くとエポックの途中でも (途中のエポックを捨てて) 学習を止めるようにした. ファイルでやり取りするので, ダッシュボード以外からも使える.
Python のコードからは, `Trainer.add_callback` で登録した関数がエポックごとの結果を受け取り, `True` を返すと学習を止められる.

### 4. 利用者のテストで見つかった, まだ対応していない項目

いろいろなタスクで ML-Found を使ったテストで見つかった項目のうち, まだ対応していないもの.

- 複数の Optimizer を名前付きで config に書き, 損失ごとに対応させて更新, 保存, 再開する仕組み (GAN など). 今は `manual_optimization` と `extra_state_dict` で利用者が書く
- チェックポイントに記録の全て (`history` と, step ごとの Scheduler では更新ごとの学習率 `lr_steps`) を毎回入れず, `history.csv` と `lr_steps.csv` に追記する形にする. 1 エポックが短く毎エポック改善する学習 (フルバッチなど) では, `best.pt` を改善のたびに記録ごと保存する時間が残る (`best_weights_only: true` なら重みだけ)
- 同じ `Trainer` で `fit` を呼び直す使い方を決める (記録を続けるか, ベストモデルと Early Stopping の状態, `resume_from` の扱い). 今は記録が重なる
- warmup 以外で複数の Scheduler を組み合わせる設定 (`SequentialLR` を config に書く形). warmup は `scheduler.warmup_steps` と `warmup_ratio` で対応済み

