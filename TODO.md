# TODO

このリポジトリでまだやっていないことを, 優先度の順に並べる. 終えたら消す.
優先度: 高・中・低は相対で, 項目があるかぎり高と中を空にしない. 高 = 今の節目「5 つの段階のアプリを 1 つのアプリに統合する」に要るものと, リポジトリの間の食い違い. 中 = 高の次に進めるもの. 低 = 残り (まだ扱っていない型を足す作業もここに置く). 条件待ち = 実例か利用者の判断が要る.

## 優先度 高

### task.yaml の dataset.format_version に版 1 を書けるようにする
- 何を: docs/FORMAT.md の `dataset.format_version` の説明 (「2 か 3」) を, 版 1 (列の正解のデータセット) も書ける決まりに直す. 版 1 の `dataset.yaml` は受け持ちと正解の出どころを持たないので, `task.dataset` から埋めないこと (`role` と `target` は直接書く) は今のまま. 今の `check_dataset_version` は `dataset.yaml` を読む場合だけ版 1 を止める. 書くだけの `task.dataset.format_version` は整数なら通すので, 文書と動きの食い違いを揃える
- なぜ: 画像の生成のデータセットは版 1 で, 文書どおりだと `task.yaml` に版を書けず null にするしかない. どのデータセットで学んだかの版が実験のディレクトリから読めなくなる
- 終わりの条件: FORMAT.md と DESIGN.md の記述と `tests/test_format.py`, `task.py` の検査が同じ決まり (版 1, 2, 3 を書け, 整数でない値は誤り) を言う. 版 1 を書いた `task.yaml` の読み書きのテストがある. 項目の意味は変わらず版を上げないことを書く
- つながり: ML-Quest の生成の骨組み (版 1 の画像の生成のデータセットを書くとき, 版を null にしている).

## 優先度 中

無し

## 優先度 低

無し

## 条件待ち

### 出力のファイルの権限
- 何を: `run.json` と `progress.json` の権限 (一時ファイルから置き換えるので 0600) を, `config.yaml` と同じ (umask に従う) にそろえるかを決める
- 待っているもの: 画面 (ML-Dashboard) を学習と別のユーザーで動かす実例

### 型ごとの指標の共有
- 何を: 共有の場所に置いた型ごとの指標 (FID の `FrechetInceptionDistance`, 異常の検知の `precision_at_k` と `average_precision`, 効果を見積もるの `qini_curve`, `qini_coefficient`, `uplift_at_k`, 値段を変える判断の `price_elasticity` と `profit_lift`, 推薦の `recall_at_k` と `ndcg_at_k`, 時間をまたぐ量の判断の `pinball_loss` と `order_cost`) の他に, Chamfer 距離のような型ごとの指標の計算も共有の場所に置く
- 待っているもの: 2 つ以上の段階で同じ計算が要る実例. それまではプロジェクトに書く

### 追加の依存の登録簿に機能を足す
- 何を: 登録簿にある機能 (`rl`, `mujoco`, `ros2`, `cuda`, `fid` と段階のライブラリの `eda`, `modelcontrol`, `inference`) の他に, 問題の設定が要る機能 (学習済みのモデルの重みの取得など) を足す
- 待っているもの: その機能が要る実例 (ML-Quest の要る機能の答え)
