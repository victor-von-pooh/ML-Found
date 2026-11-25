# TODO

このリポジトリでまだやっていないことを, 優先度の順に並べる. 終えたら消す.
優先度: 高・中・低は相対で, 項目があるかぎり高と中を空にしない. 高 = 今の節目「まだ扱っていない型の最初の 1 つを画面で通す」に要るものと, リポジトリの間の食い違い. 中 = 高の次に進めるもの. 低 = 残り. 条件待ち = 実例か利用者の判断が要る.

## 優先度 高

### 学習するパラメータの無い作りを同じ実験として残す
- 何を: 距離に基づく異常の度合い (学習用の正常の行をバッファとして持ち, 新しい行の近さから度合いを出す) のように, 学習するパラメータを持たないモデルを `Trainer` に渡すと, Optimizer を作らずに (更新は 0 回) 検証と `on_epoch_end` の指標だけを数え, `run.json`, `history.csv`, `final_model.pt` (バッファを含む) を他の実験と同じ形で残す. 学習するパラメータが無いのに Optimizer の設定を書いた場合は, 誤りにせず Optimizer と Scheduler を作らない (使わないことを `train.log` に 1 行書く). `trainer.epochs` が 1 でない場合も誤りにせず, 毎エポック同じ検証をする (`train.log` に警告を 1 行書く). この扱いを docs/DESIGN.md に書く
- なぜ: 今の `Trainer` は学習するパラメータの無いモデルで Optimizer を作れずに止まる (torch の「空のパラメータの並び」の誤り). そのため距離に基づく度合いの骨組みは `fit` を使えず, `run.json` が無い実験になり, 画面が他の実験と同じに状態と評価を並べられないため
- 終わりの条件: 学習するパラメータが無くバッファだけを持つモデルで `fit` すると, `run.json` の状態が終わりになり, `history.csv` に検証の指標が残り, `final_model.pt` を読み直すとバッファが同じ値になることをテストで確かめる. Optimizer と Scheduler の設定を書いた場合は止まらずに作らないことと `train.log` の 1 行を, `trainer.epochs` が 1 でない場合は止まらずに毎エポック同じ検証の値が `history.csv` に残ることと `train.log` の警告を, テストで確かめる. 学習するパラメータのあるモデルの学習は変わらない
- つながり: ML-Quest の docs/DESIGN.md の 11 節「異常の検知の骨組み」(距離に基づく度合いは今 `Trainer` を使わずに評価と成果物を書く), ML-Quest の「距離に基づく度合いの実験を他の実験と同じ形で残す」(この項目の後に `fit` を使う), ML-Inference の docs/DESIGN.md の「異常の度合いの一式」と「人の確認の記録」 (距離に基づく度合いの成果物を読む)

## 優先度 中

### 推薦の指標を共有の場所に置く
- 何を: 利用者ごとの上位 K 件の Recall@K と NDCG を数える関数を, 異常の検知の指標 (`precision_at_k`, `average_precision`) と同じ場所と形で置く. 決まり (正解の数が K より多い利用者, 正解の無い利用者を数えない, 数える利用者がいない場合は 0) は, 今の骨組みと ML-Inference の計算と同じにして docs/DESIGN.md に書く
- なぜ: ML-Quest の推薦の骨組み (`src/ml_quest/rank_skeleton.py` の `rank_metrics`) と ML-Inference の `rank_metrics` が, 同じ Recall@K と NDCG をそれぞれ書いて持っている. 2 つ以上の段階で同じ計算が要る実例で, 「型ごとの指標の共有」の条件に当たるため
- 終わりの条件: 骨組みと ML-Inference の今の計算と同じ見本 (正解が K より多い利用者, 正解の無い利用者, 上位が空の利用者を含む) で同じ値が返ることをテストで確かめる. 公開の名前が `__all__` にあり, docs/DESIGN.md に決まりがある
- つながり: 異常の検知の指標 (置き場所と, ML-Inference が ML-Found に依存せずに docs/DESIGN.md の見本で確かめる決まりも同じ), ML-Quest の `src/ml_quest/rank_skeleton.py` の `rank_metrics`, ML-Inference の docs/DESIGN.md の「並べるの一式」の節の `rank_metrics`

## 優先度 低

### 異常の検知の学習の例
- 何を: `examples/` に, 正常の行が多い表で正常の行だけを学ぶ表の行の自己符号化器の例を足す. モデルは ML-ModelControl の同梱の例 `tabular_autoencoder` (ML-ModelControl の docs/DESIGN.md の 9 節) と同じ形 (縮めてから戻す全結合の層. 出力は入力と同じ幅の再構成 `recon`) にする. step は正常の行の再構成の誤差を損失にし (検証用に正解の付いた異常の行が混ざる場合は, 正常の行だけで平均して `batch_size` に正常の行の数を返す), `on_epoch_end` で行ごとの再構成の誤差を度合いとして上位 K 件の適合率と PR-AUC (`precision_at_k`, `average_precision`) を数えて監視する指標にする. 学習の後に, 指標の表 (`better` 付き), 度合いの分布の曲線 (`kind="curve"`), 上位の行の要約の表 (行の id と度合い) を `save_evaluation` で書く
- なぜ: 正常だけで学ぶ流れと, 再構成の誤差を評価に残す形は今の作り (`batch_size`, `on_epoch_end`, `save_evaluation`) で書けるが, 形を示す例が無く, 骨組みと画面の見本が同じ形にそろう手がかりが無いため
- 終わりの条件: 例を少ないエポックで動かすテストで, 監視する指標が上位 K 件の適合率か PR-AUC になり, `load_evaluations` で指標の表と度合いの分布の曲線と上位の行の要約の表が読めることを確かめる. README.md と docs/DESIGN.md の例の一覧に載る
- つながり: ML-Dashboard の「異常の検知の評価を結果に出す」(同梱の `demo` の評価の形), ML-ModelControl の同梱の例 `tabular_autoencoder` (docs/DESIGN.md の 9 節)

## 条件待ち

### 出力のファイルの権限
- 何を: `run.json` と `progress.json` の権限 (一時ファイルから置き換えるので 0600) を, `config.yaml` と同じ (umask に従う) にそろえるかを決める
- 待っているもの: 画面 (ML-Dashboard) を学習と別のユーザーで動かす実例

### 型ごとの指標の共有
- 何を: Qini, Chamfer のような型ごとの指標の計算を, 共有の場所に置く (FID は ML-Found の `FrechetInceptionDistance`, 異常の検知は `precision_at_k` と `average_precision`. 推薦の分は「推薦の指標を共有の場所に置く」)
- 待っているもの: 2 つ以上の段階で同じ計算が要る実例. それまではプロジェクトに書く

### 追加の依存の登録簿に機能を足す
- 何を: 登録簿にある機能 (`rl`, `mujoco`, `ros2`, `cuda`, `fid` と段階のライブラリの `eda`, `modelcontrol`, `inference`) の他に, 問題の設定が要る機能 (学習済みのモデルの重みの取得など) を足す
- 待っているもの: その機能が要る実例 (ML-Quest の要る機能の答え)
