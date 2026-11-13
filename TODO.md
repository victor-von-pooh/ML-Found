# TODO

今後対応する予定の項目をまとめる.

## 前提

データを入れるだけで学習が回るダッシュボードアプリを, このリポジトリとは別に作る予定である.
ML-Found はその下で動く学習ライブラリとして, ダッシュボードから使える API を提供するところまでを担当する.
画面, 入力フォーム, データの取り込み, タスクの雛形などはダッシュボード側で作り, ML-Found には入れない.

着手する時期はダッシュボード側の進み具合に合わせて決める.

## 項目

### 1. Optimizer と Scheduler の登録簿 (一部対応)

`describe_optimizers()` と `describe_schedulers()` で, `torch.optim` の Optimizer と Scheduler ごとの引数の種類, 既定値, 必須か, 説明, 範囲を返すようにした. 引数は signature から取り出し, よく使うものには説明と範囲を付けた. 範囲の端を含まない場合は `exclusive_minimum` と `exclusive_maximum` で表す. Muon と SparseAdam は, そのままでは使えないことを理由付きで返す. 他の Scheduler を組み合わせるものと関数を受け取るもの (`SequentialLR`, `ChainedScheduler`, `LambdaLR`, `MultiplicativeLR`) は返さない.
`start_experiment` は, config の引数の名前の typo, 必須の引数の不足, 範囲の外の値を, 出力先を作る前に確かめる.

残っていること.

- プロジェクト側で作った Optimizer を, 同じ仕組みで登録する方法

### 2. closure を渡す更新手順への対応 (一部対応)

`update_parameters` が, `step` に closure が必要な Optimizer (LBFGS など) を自動で見分け, 同じバッチで損失と勾配を計算し直す closure を渡すようにした.
あわせて `trainer.manual_optimization` で, ML-Found が逆伝播と更新を行わない設定も加えた.

残っていること.

- Nelder-Mead 法のような勾配を使わない最適化法. 勾配を使うかどうかを 1 の登録簿の仕様に持たせる
- 1 の登録簿の仕様に closure が必要かどうかも持たせ, 判定をシグネチャ (`requires_closure`) ではなく仕様から行う

### 3. 進捗を渡す仕組みと学習の中断 (一部対応)

実験のディレクトリに `run.json` (実験の状態) と `progress.json` (進み具合) を書き出し, `STOP` のファイルを置くとエポックの区切りで, `STOP` に `now` と書くとエポックの途中でも (途中のエポックを捨てて) 学習を止めるようにした. ファイルでやり取りするので, ダッシュボード以外からも使える.

残っていること.

- Python のコードからエポックごとの指標を受け取るコールバック

### 4. 利用者のテストで見つかった, まだ対応していない項目

いろいろなタスクで ML-Found を使ったテストで見つかった項目のうち, まだ対応していないもの.

- 複数の Optimizer を名前付きで config に書き, 損失ごとに対応させて更新, 保存, 再開する仕組み (GAN など). 今は `manual_optimization` と `extra_state_dict` で利用者が書く
- 検証に使うモデルを差し替えるフック (EMA の重みで検証する, など)
- モデルの構造のログを `print(model)` 以外でも残す仕組み (外部ライブラリの層は中身が残らない). 今は README で `extra_repr` を勧めている
- 学習用データを推論モードで評価し直す設定 (Dropout が効いた状態の `train/` の指標を補う)
- 外部ライブラリの乱数 (qiskit など) を, 固定と再開時の保存の対象に登録する仕組み
- 範囲の確認から外れる引数の値 (範囲を持たない引数, 自作のクラス) の誤りも, 出力先を作る前に確かめる. 今は `Trainer` の作成で PyTorch のエラーになり, 出力先が残る (`train.log` と `run.json` には記録する)
- チェックポイントに記録の全て (`history` と, step ごとの Scheduler では更新ごとの学習率 `lr_steps`) を毎回入れず, `history.csv` と `lr_steps.csv` に追記する形にする. 1 エポックが短く毎エポック改善する学習 (フルバッチなど) では, `best.pt` を改善のたびに記録ごと保存する時間が残る (`best_weights_only: true` なら重みだけ)
- エポックの途中で止めた場合に, 捨てたエポックの更新回数, Optimizer, Scheduler, 乱数の状態をメモリの上でも戻す (同じ `Trainer` で `fit` を呼び直す場合のため)
- `OneCycleLR` 以外のデータの量で決まる引数 (warmup 付きの Scheduler, step ごとに使う `CosineAnnealingLR` の `T_max` など) も補う仕組み
- 再開時に Scheduler の引数が config と違う場合に, config の値で上書きするかを選べる設定 (今は警告だけ)

