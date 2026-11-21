from pathlib import Path
import subprocess
import sys

import pytest

from ml_found.rl import read_policy_spec

# Pendulum の例のスクリプト
SCRIPT = Path(__file__).resolve().parents[1] / "examples/pendulum/train.py"


def test_pendulum_example_runs(tmp_path: Path) -> None:
    """
    Pendulum の例が短い学習で動き, 守りと一緒に書き出した方策を守り付きで
    評価することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    pytest.importorskip("gymnasium")

    # 出力先を一時ディレクトリにして, 2 エポックだけ学習
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "trainer.epochs=2",
            "rollout.episodes=2", "evaluation.episodes=1",
            f"experiment.output_dir={tmp_path}"
        ],
        capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert "守り付きの評価の報酬の和の平均" in result.stdout

    # 連続の行動の一式で, 守りはトルクの範囲と角速度の止める条件
    run_dirs = [path.parent for path in tmp_path.rglob("policy.json")]
    assert len(run_dirs) == 1
    spec = read_policy_spec(run_dirs[0])
    assert spec["head"] == "deterministic"
    assert spec["action_space"]["type"] == "box"
    assert spec["guards"]["action_high"] == [1.5]
    assert spec["guards"]["observation_high"] == [None, None, 7.0]
    assert spec["guards"]["fallback"] == {"kind": "action", "action": [0.0]}
    assert spec["metadata"]["algorithm"] == "ActorCritic"

    # 守り付きの評価の表と, 価値の関数の損失の記録
    run_dir = run_dirs[0].parent
    assert list(run_dir.rglob("guarded_episodes*"))
    history = (run_dir / "history.csv").read_text(encoding="utf-8")
    assert "train/value_loss" in history
