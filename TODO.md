# TODO

このリポジトリでまだやっていないことを, 優先度の順に並べる. 終えたら消す.
優先度: 高・中・低は相対で, 項目があるかぎり高と中を空にしない. 高 = 今の節目「まだ扱っていない型の最初の 1 つを画面で通す」に要るものと, リポジトリの間の食い違い. 中 = 高の次に進めるもの. 低 = 残り. 条件待ち = 実例か利用者の判断が要る.

## 優先度 高

### 分位点の損失を学習の損失に使える形で置く
- 何を: step の中で勾配を流せる分位点の損失を, `pinball_loss` と同じ場所に置く. 実際の値 (N,), 分位点ごとの予測 (N, 水準の数) と水準の並びを受け, 水準 q の損 max(q (y − ŷ), (q − 1) (y − ŷ)) の全ての行と水準の平均をスカラーのテンソルで返す (`torch.nn.Module` の形で, 水準は作るときに受ける). 入力の決まり (水準の範囲と重なり, 形の誤り) は `pinball_loss` と同じにし, 値が `pinball_loss` の `mean` と同じになることを docs/DESIGN.md の 14 節の見本で示す
- なぜ: 時間をまたぐ量の判断の骨組み (ML-Quest) は損失にも ML-Found の分位点の損失を使うが, `pinball_loss` は評価のために勾配を外して Python の数を返すので, step の損失には使えないため. 骨組みが損失を自分で書くと, 評価の値と学習の損失の数え方がずれる
- 終わりの条件: 勾配が予測に流れ, 値が docs/DESIGN.md の見本の `pinball_loss` の `mean` と同じになることをテストで確かめる. float32 と GPU の予測もそのままの装置と型で数える. 公開の名前が `__all__` にあり, docs/DESIGN.md の 14 節に決まりがある
- つながり: docs/DESIGN.md の 14 節の「時間をまたぐ量の判断の分位点の損失と見込みの損」, ML-Quest の「時間をまたぐ量の判断の骨組み」, ML-ModelControl の「先の需要の分位点を出す雛形」(出力 `quantiles`)

## 優先度 中

### 異常の検知の学習の例
- 何を: `examples/` に, 正常の行が多い表で正常の行だけを学ぶ表の行の自己符号化器の例を足す. モデルは ML-ModelControl の同梱の例 `tabular_autoencoder` (ML-ModelControl の docs/DESIGN.md の 9 節) と同じ形 (縮めてから戻す全結合の層. 出力は入力と同じ幅の再構成 `recon`) にする. step は正常の行の再構成の誤差を損失にし (検証用に正解の付いた異常の行が混ざる場合は, 正常の行だけで平均して `batch_size` に正常の行の数を返す), `on_epoch_end` で行ごとの再構成の誤差を度合いとして上位 K 件の適合率と PR-AUC (`precision_at_k`, `average_precision`) を数えて監視する指標にする. 学習の後に, 指標の表 (`better` 付き), 度合いの分布の曲線 (`kind="curve"`), 上位の行の要約の表 (行の id と度合い) を `save_evaluation` で書く
- なぜ: 正常だけで学ぶ流れと, 再構成の誤差を評価に残す形は今の作り (`batch_size`, `on_epoch_end`, `save_evaluation`) で書けるが, 形を示す例が無く, 骨組みと画面の見本が同じ形にそろう手がかりが無いため
- 終わりの条件: 例を少ないエポックで動かすテストで, 監視する指標が上位 K 件の適合率か PR-AUC になり, `load_evaluations` で指標の表と度合いの分布の曲線と上位の行の要約の表が読めることを確かめる. README.md と docs/DESIGN.md の例の一覧に載る
- つながり: ML-Dashboard の docs/DESIGN.md の「一覧と比較の評価」(異常の検知の評価の読み方), ML-ModelControl の同梱の例 `tabular_autoencoder` (docs/DESIGN.md の 9 節)

## 優先度 低

無し

## 条件待ち

### 出力のファイルの権限
- 何を: `run.json` と `progress.json` の権限 (一時ファイルから置き換えるので 0600) を, `config.yaml` と同じ (umask に従う) にそろえるかを決める
- 待っているもの: 画面 (ML-Dashboard) を学習と別のユーザーで動かす実例

### 型ごとの指標の共有
- 何を: Chamfer のような型ごとの指標の計算を, 共有の場所に置く (FID は ML-Found の `FrechetInceptionDistance`, 異常の検知は `precision_at_k` と `average_precision`, 効果を見積もるは `qini_curve`, `qini_coefficient`, `uplift_at_k`. 値段を変える判断は `price_elasticity` と `profit_lift`, 推薦は `recall_at_k` と `ndcg_at_k`, 時間をまたぐ量の判断は `pinball_loss` と `order_cost`)
- 待っているもの: 2 つ以上の段階で同じ計算が要る実例. それまではプロジェクトに書く

### 追加の依存の登録簿に機能を足す
- 何を: 登録簿にある機能 (`rl`, `mujoco`, `ros2`, `cuda`, `fid` と段階のライブラリの `eda`, `modelcontrol`, `inference`) の他に, 問題の設定が要る機能 (学習済みのモデルの重みの取得など) を足す
- 待っているもの: その機能が要る実例 (ML-Quest の要る機能の答え)
