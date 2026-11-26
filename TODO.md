# TODO

このリポジトリでまだやっていないことを, 優先度の順に並べる. 終えたら消す.
優先度: 高・中・低は相対で, 項目があるかぎり高と中を空にしない. 高 = 今の節目「まだ扱っていない型の最初の 1 つを画面で通す」に要るものと, リポジトリの間の食い違い. 中 = 高の次に進めるもの. 低 = 残り. 条件待ち = 実例か利用者の判断が要る.

## 優先度 高

### 時間をまたぐ量の判断の学習の例
- 何を: `examples/` に, 商品ごとの日ごとの売れた数を合成した表から先の期間の売れた数の合計を分位点で予測する例を足す. モデルは ML-ModelControl の同梱の例 `tabular_demand` (ML-ModelControl の docs/DESIGN.md の 9 節) と同じ形 (全結合の層の出力に softplus をかけて水準の次元で前から足す. 出力は (バッチ, 水準の数) の `quantiles`) にする. step は `PinballLoss` を損失にし, `on_epoch_end` で検証用の全ての行の `pinball_loss` の `mean` を監視する指標にする. 学習の後に, 品切れの損と売れ残りの損の比から決まる水準の分位点を発注の数にし (学んだ水準に無ければ隣の 2 つの水準を直線で補う), `order_cost` で前の期間と同じ数の発注と比べ, 指標の表 (`better` 付き), 水準ごとの表, 行ごとの表を `save_evaluation` で書く
- なぜ: 分位点の損失で学び, 発注の数の見込みの損を評価に残す流れは今の作り (`PinballLoss`, `pinball_loss`, `order_cost`, `save_evaluation`) で書けるが, 形を示す例が無く, 骨組み (ML-Quest) と画面 (ML-Dashboard) の見本が同じ形にそろう手がかりが無いため
- 終わりの条件: 例を少ないエポックで動かすテストで, 監視する指標が分位点の損失になり, `load_evaluations` で指標の表と水準ごとの表と行ごとの表が読め, 見込みの損の比 `order_cost_ratio` が数で出ることを確かめる. README.md と docs/DESIGN.md の例の一覧に載る
- つながり: docs/DESIGN.md の 14 節の「時間をまたぐ量の判断の分位点の損失と見込みの損」, ML-ModelControl の同梱の例 `tabular_demand` (docs/DESIGN.md の 9 節), ML-Quest の docs/DESIGN.md の 11 節の「時間をまたぐ量の判断の骨組み (予測するで, 先の期間の需要)」, ML-Dashboard の docs/DESIGN.md の「一覧と比較の評価」(時間をまたぐ量の判断の評価の読み方)

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
