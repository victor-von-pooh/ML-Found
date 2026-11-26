import json
from pathlib import Path
import subprocess
import sys

import pytest

from ml_found import load_evaluations

# 時間をまたぐ量の判断の例のスクリプト
SCRIPT = Path(__file__).resolve().parents[1] / "examples/demand/train.py"


def test_demand_example_runs(tmp_path: Path) -> None:
    """
    時間をまたぐ量の判断の例が短い学習で動き, 分位点の損失を監視して,
    指標の表, 水準ごとの表, 行ごとの表を書き, 見込みの損の比が数で出る
    ことを確かめる関数

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
    assert "見込みの損の比" in result.stdout

    # 監視する指標は検証用の分位点の損失
    run_dirs = [path.parent for path in tmp_path.rglob("run.json")]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert "valid/pinball_loss" in json.dumps(run)
    history = (run_dir / "history.csv").read_text(encoding="utf-8")
    assert "valid/pinball_loss" in history

    # 指標の表 (向き付き. 比は数), 水準ごとの表, 行ごとの表
    found = {item["name"]: item for item in load_evaluations(run_dir)}
    row = found["demand_test"]["rows"][0]
    assert found["demand_test"]["better"]["pinball_loss"] == "lower"
    assert row["level"] == pytest.approx(0.75)
    assert row["order_cost_ratio"] > 0
    assert row["order_cost_saving"] == pytest.approx(
        row["order_cost_baseline"] - row["order_cost"]
    )
    levels = [item["level"] for item in found["pinball_levels_test"]["rows"]]
    assert levels == [0.1, 0.25, 0.5, 0.75, 0.9]
    items = found["order_items_test"]["rows"]
    assert len(items) == row["rows"]
    assert all(item["order"] >= 0 for item in items)
