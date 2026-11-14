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

- 1 エポックが短く毎エポック改善する学習 (フルバッチなど) の, チェックポイントの保存の時間. 記録を列ごとの数の配列で持つようにして, 5000 エポックのフルバッチ学習で 41 秒が 13 秒になった (保存しない場合は 4 秒). 残りは `best.pt` と `last.pt` を毎エポック書く分で, `every_n_epochs` と `best_weights_only: true` で減らせる. 記録をチェックポイントに入れず `history.csv` に追記する形は, 再開の扱いが変わるので見送り

### 5. 複数の Optimizer を config に名前付きで書く (対応済み)

`optimizers` に名前付きで Optimizer を書けるようにした (README の「複数の Optimizer」). 名前ごとに `name`, `params`, 受け持つ部分 `modules`, `scheduler` を書き, `step` は名前ごとの損失を返す. ML-Found が損失ごとに受け持つパラメータの勾配を求め, クリップして config の順に更新し, 状態の保存と再開, 登録簿による確認 (範囲, typo, 作れるか), 名前ごとの損失, 学習率, 勾配のノルムの記録を行う. 一番上の `scheduler` との併用, Scheduler ごとに違う `interval`, closure が必要な Optimizer, `resume_scheduler: config` には対応しない.

### 6. 同じ Trainer で fit を呼び直したら続きとして扱う (対応済み)

呼び直した `fit` は, 最後に終えたエポックの次から `trainer.epochs` の分だけ学習し, 記録を同じ実験に続ける. 重みは前の `fit` の最後のエポックのもの (ベストモデルなどに置き換える前) から続け, Early Stopping は止めた印と我慢の回数を戻す. `resume_from` は最初の `fit` だけで使う. `auto` で補った Scheduler の長さは延ばさず警告し, `OneCycleLR` で総ステップ数を超える場合はエラーにする. Optimizer を差し替えたい場合は新しい Trainer を作る.

なお, warmup 以外で複数の Scheduler を組み合わせる設定 (`SequentialLR` を config に書く形) は, warmup と 1 つの Scheduler で実用の大半に足りるため, 対応しない.
