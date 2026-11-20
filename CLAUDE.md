# CLAUDE.md

ML-Found で作業するときの方針をまとめる.

## このリポジトリの目的

PyTorch のモデルを config から学習させる基盤のライブラリ.

- プロジェクト (config とスクリプトを持つディレクトリ) から submodule として使う. 学習の画面 (ML-Dashboard) が必要とする API (実験の状態の書き出し, 登録簿, config の確認, 中断など) はこのライブラリが返す. 画面に固有のものは入れない
- モデルの組み立て (ML-ModelControl) と推論は対象外

## コードの書き方

- [CODING_STYLE.md](CODING_STYLE.md) に従う (日本語の NumPy 形式の docstring, コメントは体言止め, 79 文字以内, `torch.nn.functional` を使わないなど)
- ドキュメントとコードのコメントでは, 「機械学習を全て GUI 上で行う」という目的のために開発しているリポジトリ (今は ML-Found, ML-Dashboard, ML-ModelControl, ML-CreateArchitecture, ML-EDA, ML-Fathom, ML-Inference, ML-Provide, ML-Quest, ML-Pathfinder, ML_tester) 以外に触れない. ローカルのディレクトリ構成やパスも書かない

## 文書

- [README.md](README.md): 何をするライブラリか, 導入, 短い使い方
- [docs/DESIGN.md](docs/DESIGN.md): 今の設計と決まり. 1. 役割と位置づけ, 2. 他のリポジトリとの約束 から始める. まだやっていないことは書かない
- [docs/FORMAT.md](docs/FORMAT.md): 他のリポジトリと受け渡すプロジェクトと実験のディレクトリの形と版
- [TODO.md](TODO.md): まだやっていないことを優先度の順に. 終えたら消す
- [CODING_STYLE.md](CODING_STYLE.md): コードの書き方

## 進める流れ

1. 方針を変える場合や新しい機能を加える場合は, 先に docs/DESIGN.md と TODO.md を直す (使い方が変わる場合は README.md も). プロジェクトや実験のディレクトリの形 (他のシステムとの約束) を変える場合は, [docs/FORMAT.md](docs/FORMAT.md) の版の決まりに従い, 文書と `tests/test_format.py` も直す
2. テストを書き, `pytest` を通す
3. **セルフレビュー**: 複数のエージェントを並行して動かし, バグ, 既存のプロジェクトとの互換, CODING_STYLE.md への準拠を確認する
4. 指摘を直し, 直した内容と残した指摘をユーザーに報告する

## git

- 意味のある変更ごとにブランチを切って PR を作る. `main` に直接プッシュしない
- CI が通ったことを確かめてからマージする. 自分の PR は GitHub の仕様で Approve できないので, 確かめた内容をレビューのコメントに残してからマージする
- コミットとプッシュ, PR のマージは基本的にユーザーが行う. ユーザーが `[Commit] <日時>までなら...` の prefix を付けて指示した場合は, その日時まで Claude の判断で行ってよい
- コミットメッセージと PR のタイトルは日本語の 1 行
