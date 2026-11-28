# TODO

このリポジトリでまだやっていないことを, 優先度の順に並べる. 終えたら消す.
優先度: 高・中・低は相対で, 項目があるかぎり高と中を空にしない. 高 = 今の節目「5 つの段階のアプリを 1 つのアプリに統合する」に要るものと, リポジトリの間の食い違い. 中 = 高の次に進めるもの. 低 = 残り (まだ扱っていない型を足す作業もここに置く). 条件待ち = 実例か利用者の判断が要る.

## 優先度 高

### task.yaml の正解の出どころに, 学んだデータそのものが正解の値を足す
- 何を: `target.origin.type` に, 学んだデータそのものが正解である場合の値を足す. 今の値 (`process`, `annotation`, `measurement`, `synthetic`, `environment`, `unknown`) は, どれも条件の無い生成 (画像や表の生成) に当たらない. 値の名前と `describe_origin` の説明文を決め, docs/FORMAT.md の「task.yaml」の表と `describe_origin` の説明と `tests/test_format.py` を直す. 値は ML-EDA の `dataset.yaml` には無いので, `environment` と同じく「ML-Found だけの値」と書く
- なぜ: ML-Quest の生成の骨組みは, 正解の出どころに合う値が無いので `unknown` を書いている. `unknown` は「指標が何との一致かを確かめる」と読まれ, 生成では正解が学んだデータ自身であることが指標の横に出ない
- 版: 選択肢の値を足すだけなので版は上げない (docs/FORMAT.md の「版を変えない変更」). 知らない値は `unknown` として読む決まりなので, 古い ML-Found や ML-Dashboard は誤った説明を出さず, 「分からない」の説明のまま. 新しい説明が出るのは ML-Found を更新してからで, ML-Dashboard の出どころの表示は `describe_origin` の文を出すので, 更新だけで足りる (表示の作り替えは要らない). 値を使って書くのは更新後の ML-Quest の生成の骨組みだけにする
- 終わりの条件: 値と説明が FORMAT.md, DESIGN.md, `describe_origin`, テストでそろい, 古い読み手が知らない値を `unknown` として読むことをテストで確かめる
- つながり: ML-Quest の生成の骨組み (`unknown` を書いているのを新しい値に改める), ML-Dashboard (`describe_origin` の文を指標の横に出す)

## 優先度 中

### task.yaml の dataset.format_version に版 1 を書けるようにする
- 何を: docs/FORMAT.md の `dataset.format_version` の説明 (「2 か 3」) を, 版 1 (列の正解のデータセット) も書ける決まりに直す. 版 1 の `dataset.yaml` は受け持ちと正解の出どころを持たないので, `task.dataset` から埋めないこと (`role` と `target` は直接書く) は今のまま. 今の `check_dataset_version` は `dataset.yaml` を読む場合だけ版 1 を止める. 書くだけの `task.dataset.format_version` は整数なら通すので, 文書と動きの食い違いを揃える
- なぜ: 画像の生成のデータセットは版 1 で, 文書どおりだと `task.yaml` に版を書けず null にするしかない. どのデータセットで学んだかの版が実験のディレクトリから読めなくなる
- 終わりの条件: FORMAT.md と DESIGN.md の記述と `tests/test_format.py`, `task.py` の検査が同じ決まり (版 1, 2, 3 を書け, 整数でない値は誤り) を言う. 版 1 を書いた `task.yaml` の読み書きのテストがある. 項目の意味は変わらず版を上げないことを書く
- つながり: ML-Quest の生成の骨組み (版 1 の画像の生成のデータセットを書くとき, 版を null にしている). 上の出どころの項目と同じ生成の骨組みの改めで足りる

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
