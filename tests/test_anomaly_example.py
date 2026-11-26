from pathlib import Path
import subprocess
import sys

from ml_found import load_evaluations

# 異常の検知の例のスクリプト
SCRIPT = Path(__file__).resolve().parents[1] / "examples/anomaly/train.py"


def test_anomaly_example_runs(tmp_path: Path) -> None:
    """
    異常の検知の例が短い学習で動き, 検証用の上位 K 件の適合率を監視して,
    指標の表, 度合いの分布の曲線, 上位の行の要約の表を書くことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 出力先を一時ディレクトリにして, 3 エポックだけ学習
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "trainer.epochs=3",
         f"experiment.output_dir={tmp_path}"],
        capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert "テスト用の上位 20 件の適合率" in result.stdout

    # 監視する指標と, 学習の記録の検証用の指標
    run_dirs = [path.parent for path in tmp_path.rglob("run.json")]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    history = (run_dir / "history.csv").read_text(encoding="utf-8")
    assert "valid/precision_at_k" in history
    assert "valid/average_precision" in history
    run = (run_dir / "run.json").read_text(encoding="utf-8")
    assert "valid/precision_at_k" in run

    # 指標の表 (向き付き), 度合いの分布の曲線, 上位の行の要約の表
    found = {item["name"]: item for item in load_evaluations(run_dir)}
    table = found["anomaly_test"]
    assert table["better"]["precision_at_k"] == "higher"
    row = table["rows"][0]
    assert row["anomalies"] == 20 and row["k"] == 20
    assert 0.0 <= row["average_precision"] <= 1.0
    curve = found["anomaly_scores_test"]
    assert curve["kind"] == "curve" and curve["x"] == "score"
    assert sum(item["anomaly_rows"] for item in curve["rows"]) == 20
    top = found["anomaly_top_test"]["rows"]
    assert len(top) == 10
    assert [item["score"] for item in top] == sorted(
        (item["score"] for item in top), reverse=True
    )
