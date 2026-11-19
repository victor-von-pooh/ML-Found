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

`optimizers` に名前付きで Optimizer を書けるようにした (README の「複数の Optimizer」). 名前ごとに `name`, `params`, 受け持つ部分 `modules`, `scheduler` を書き, `step` は名前ごとの損失を返す. ML-Found が損失ごとに受け持つパラメータの勾配を求め, クリップして config の順に更新し, 状態の保存と再開, 登録簿による確認 (範囲, typo, 作れるか), 名前ごとの損失, 学習率, 勾配のノルムの記録を行う. 一番上の `scheduler` との併用, Scheduler ごとに違う `interval`, closure が必要な Optimizer と勾配を使わない Optimizer, `Trainer` に渡す自分で作った Optimizer と Scheduler, `resume_scheduler: config` には対応しない.

ダッシュボードで名前ごとに扱うための情報は, `run.json` の `optimizers` (Optimizer ごとの名前, クラス, 受け持つ部分, Scheduler) に書き出す (1 つの Optimizer の場合も同じ形で, 名前と受け持つ部分は `null`)

### 6. 同じ Trainer で fit を呼び直したら続きとして扱う (対応済み)

呼び直した `fit` は, 最後に終えたエポックの次から `trainer.epochs` の分だけ学習し, 記録を同じ実験に続ける. 重みは前の `fit` の最後のエポックのもの (ベストモデルなどに置き換える前) から続け, Early Stopping は止めた印と我慢の回数を戻す. `resume_from` は最初の `fit` だけで使う. `auto` で補った Scheduler の長さは延ばさず警告し, `OneCycleLR` で総ステップ数を超える場合はエラーにする. Optimizer を差し替えたい場合は新しい Trainer を作る.

なお, warmup 以外で複数の Scheduler を組み合わせる設定 (`SequentialLR` を config に書く形) は, warmup と 1 つの Scheduler で実用の大半に足りるため, 対応しない.

### 7. 実験の起動のしかたと, fit の外のエラーの記録 (対応済み)

`start_experiment` は, 実行したスクリプト, default と experiment の config のパス, 「項目=値」の上書きを `source.yaml` に書く. 読み込み関数 (`load_config` など) が返した config に結び付けて覚えるので, 学習スクリプトは変えなくてよい. コマンドラインの引数の全体は, 秘密の値を含み得るので記録しない.
`start_experiment` の後に `fit` の外で処理されなかったエラー (データの読み込みやモデルの作成の失敗) も, `sys.excepthook` で `train.log` と `run.json` (`failed`, キーボードの割り込みは `stopped`) に記録し, ダッシュボードで未完了のまま残らないようにした. ノートブックのセルのエラーは記録しない.

### 8. 課題の型への広げ方 (2026-11-18 の方針)

学習のループ (step の約束, 名前付きの複数の Optimizer, 手で行う更新, エポックの終わりの処理) は型に依らず, オートエンコーダ, VAE, GAN, 拡散, 量子, BERT の事前学習と微調整まで通っている. 足りないのはループの外側. 機械学習の課題は, 1 件の分からない値を当てる「予測する」(分類, 回帰, 時系列の予測. 検出, 領域分け, 文字起こし, 3D の形のような構造のある出力と, 正解のほぼ無い異常の検知も) のほかに, 正解が 1 つに決まらない中身を「作る」, 多くの候補から「並べる」(推薦, 検索), 正解の無いデータを「まとめる」(群分け), 行動の「効果を見積もる」(uplift, バンディットや強化学習の方策) に分けられる. 正解の出どころ (列, 対のファイル, 自分自身, 今の処理の出力, 行動の結果, 介入の記録, 指示 (プロンプト)), 1 件の単位 (行, まとまり, 対, 要求と候補, 窓), 作り方 (規則, 既製のモデルをそのまま, 頭だけ学習, 微調整, 一から) で, データの形と確かめ方が変わる (2026-11-18 に, モデルの系統と課題の事例を調べて決めた). 広げる順は, 約束 (ファイルの形の版) を変えずに出せるもの → 約束の版を上げるもの → 型ごとの通り道.

- 高価な評価 (FID, Chamfer, 検索の再現率) を毎エポックでなく行う: 検証の間隔の設定と, 検証したエポックだけで最良を選ぶ (済み: `trainer.valid_every_n_epochs`. 間隔で割り切れるエポックと予定の最後のエポックだけ検証し, ベストモデル, Early Stopping, ReduceLROnPlateau を判定する. 待つ回数は判定の回数で数える. 2026-11-19 の相談の答え). 評価の間も進み具合に「評価中」を出す (済み: `progress.json` の `stage` が `epoch_end`. `on_epoch_end` を継承した場合だけ書く). 書き方 (勾配を切る, 乱数を学習と分ける) を README に
- 実験のディレクトリに任意のファイルを約束として足す (docs/FORMAT.md. 版は上げない): `evaluations/` (データ全体の評価. グループごとの表, 曲線, 比べる相手との差, 守る指標), `samples/` (画像, 文, 音, 点群の見本), `task.yaml` (受け持ち, 出力の意味, 作り方と元にした実験や重み. 後の推論の段階が読む)
- 大きな学習済みのモデルの微調整: 勾配の蓄積, 混合精度 (bf16 から. mps での効果は未確認), 学習するパラメータだけの保存, 重みの一部の読み込みの補助
- 強化学習は, 2026-11-19 の相談の答えで「方策を作り, 確かめ, 守りと一緒に書き出すところまで」に広げた (下の 10)
- 型ごとの指標の計算 (NDCG, Qini, Chamfer, FID) は, 当面はプロジェクトに書き, 2 つ以上の段階で同じ計算が要るものが増えたら共有の場所へ

### 9. 追加の依存の一覧と案内 (最初の分は対応済み. 2026-11-19 の相談の答え)

外のライブラリは要るまで入れず, 入れるときは追加の依存にする. 機能ごとに「要るもの, 入っているかの確かめ方, 入れ方」の登録簿を持ち (`describe_requirements`), 足りなければ `require` が次の一手 (pip のコマンドか, pip で入らないものは公式の手順) を言って止まる. 最初は `rl` (Gymnasium), `mujoco`, `ros2`, `cuda` を入れた.

- 残り: 画面 (ML-Dashboard) のカードで一覧を見せる. 問題の設定の受け渡し (ML-Quest) が要る機能の名前を書き, 学習を始める前に確かめる. 機能を足すときは登録簿にも足す (学習済みのモデルの重みの取得など)

### 10. 強化学習の方策 (最初の分は対応済み. 2026-11-19 の相談の答え)

自律して行動する方策を作り, 確かめ, 守りと一緒に書き出すところまでを扱う. 実機での実行は道具の外. 環境とのやり取りを loader にして `Trainer` の step とエポックの決まりに乗せ (`RolloutLoader`, `EvaluationLoader`), 経験の記録 (`ExperienceBuffer`), 探索の設定 (`Exploration`), 基準付きの REINFORCE (`ReinforceStep`), エピソードの報酬, 歩数, 安全の違反, 成功の割合の記録, 守り (行動の上限, 止める条件, 外れたときの動き) と一緒の書き出し (`export_policy`, `policy/` の形は docs/FORMAT.md) と読み直し (`load_policy`, `apply_guards`) を入れた. 例は `examples/cartpole/`.

- 残り: 連続の行動 (Box) の学習と価値の関数を使う方法 (Actor-Critic, PPO など) の補助, 記録した経験からの方策の見積もり (重点サンプリングなど), 環境の状態の再開, 評価の成功率を画面で見せること
