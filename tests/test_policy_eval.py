import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import ml_found
from ml_found import (
    MissingRequirementError, load_evaluations, save_evaluation
)
from ml_found.rl import (
    evaluate_policy, export_policy, load_policy, make_env, make_guards,
    read_policy_spec
)
from test_requirements import hide_modules
from test_rl import Space, ToyEnv
from test_trainer import make_experiment


class RightModel(nn.Module):
    def __init__(self, hidden_dim: int = 1):
        # 親クラスのコンストラクタを呼び出す
        super(RightModel, self).__init__()

        # 隠れ層の次元数を保存
        self.hidden_dim = hidden_dim

        # 行動 1 のスコアが常に高い線形層
        self.layer = nn.Linear(1, 2)
        with torch.no_grad():
            self.layer.weight.zero_()
            self.layer.bias.copy_(torch.tensor([0.0, 1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            観測

        Returns
        ----------
        scores: torch.Tensor
            行動ごとのスコア
        """
        scores = self.layer(x)

        return scores


class ClosingEnv(ToyEnv):
    def __init__(self):
        # 親クラスのコンストラクタを呼び出す
        super(ClosingEnv, self).__init__()

        # 閉じたかの印を保存
        self.closed = False

    def close(self) -> None:
        """
        環境を閉じる関数

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        self.closed = True


class BrokenPolicy:
    def act(self, observation: Any) -> dict:
        """
        status の無い結果を返す関数

        Parameters
        ----------
        observation: Any
            観測

        Returns
        ----------
        result: dict
            行動だけの辞書
        """
        result = {"action": 1}

        return result


def guarded_policy(tmp_path: Path, high: float, fallback: str) -> Any:
    """
    観測の上限で止める守りを付けた方策を書き出して読み直す関数

    Parameters
    ----------
    tmp_path: Path
        書き出すディレクトリの親
    high: float
        観測の上限 (止める条件)
    fallback: str
        外れたときの動き

    Returns
    ----------
    policy: Any
        load_policy の方策
    """
    env = ToyEnv()
    guards = make_guards(
        env.observation_space, env.action_space, observation_low=[None],
        observation_high=[high], fallback=fallback,
        fallback_action=0 if fallback == "action" else None
    )
    directory = export_policy(
        tmp_path / f"policy_{fallback}", RightModel(), env.observation_space,
        env.action_space, guards
    )
    policy = load_policy(directory, RightModel())

    return policy


def test_evaluate_policy_handover_ends_episode(tmp_path: Path) -> None:
    """
    handover でエピソードを終え, 守りの結果を数えることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 観測は 0, 1, 2, ... なので, 3 を見たところで元の制御に戻す
    policy = guarded_policy(tmp_path, 2.5, "handover")
    env = ClosingEnv()
    rows = evaluate_policy(policy, env, episodes=2, seed=7)
    assert rows[0] == {
        "episode": 0, "seed": 7, "reward": 3.0, "length": 3,
        "end": "handover", "ok": 3, "clipped": 0, "fallback": 0, "stop": 0,
        "handover": 1
    }
    assert rows[1]["seed"] == 8 and rows[1]["episode"] == 1
    assert env.seeds == [7, 8]

    # 渡された環境は閉じない
    assert env.closed is False


def test_evaluate_policy_stop_fallback_and_terminated(
    tmp_path: Path
) -> None:
    """
    stop で終え, fallback は続け, 守りに当たらなければ環境の終了まで
    動かすことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    stop = evaluate_policy(
        guarded_policy(tmp_path, 0.5, "stop"), ToyEnv(), episodes=1
    )
    assert stop[0]["end"] == "stop"
    assert (stop[0]["ok"], stop[0]["stop"], stop[0]["length"]) == (1, 1, 1)
    assert stop[0]["seed"] is None

    # 決めた行動 0 (報酬 0) で最後まで続ける
    fallback = evaluate_policy(
        guarded_policy(tmp_path, 1.5, "action"), ToyEnv(), episodes=1
    )
    assert fallback[0]["end"] == "terminated"
    assert fallback[0]["length"] == 5
    assert (fallback[0]["ok"], fallback[0]["fallback"]) == (2, 3)
    assert fallback[0]["reward"] == 2.0

    # 止める条件の外では環境の終了まで. 歩数の上限で切ることも
    free = guarded_policy(tmp_path, 9.0, "handover")
    full = evaluate_policy(free, ToyEnv(), episodes=1, seed=0)
    assert (full[0]["end"], full[0]["reward"], full[0]["ok"]) == (
        "terminated", 5.0, 5
    )
    cut = evaluate_policy(free, ToyEnv(), episodes=1, seed=0, max_steps=2)
    assert (cut[0]["end"], cut[0]["length"]) == ("max_steps", 2)


def test_evaluate_policy_rows_go_to_save_evaluation(tmp_path: Path) -> None:
    """
    返した行を save_evaluation にそのまま渡せることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    experiment = make_experiment(tmp_path)
    rows = evaluate_policy(
        guarded_policy(tmp_path, 2.5, "handover"), ToyEnv(), episodes=3,
        seed=0
    )
    save_evaluation(
        experiment, "guarded_episodes", rows, "simulator", guard=True
    )
    evaluation = load_evaluations(experiment.run_dir)[0]
    assert evaluation["columns"] == [
        "episode", "seed", "reward", "length", "end", "ok", "clipped",
        "fallback", "stop", "handover"
    ]
    assert len(evaluation["rows"]) == 3
    assert evaluation["guard"] is True


def test_evaluate_policy_argument_checks(tmp_path: Path) -> None:
    """
    エピソード数, 種, 歩数の上限, 方策の返す値の誤りを拒むことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    policy = guarded_policy(tmp_path, 2.5, "handover")
    for kwargs in (
        {"episodes": 0}, {"episodes": True}, {"episodes": 1, "seed": 1.5},
        {"episodes": 1, "seed": True}, {"episodes": 1, "max_steps": 0}
    ):
        with pytest.raises(ValueError):
            evaluate_policy(policy, ToyEnv(), **kwargs)

    # status の無い結果を返す方策は誤り
    with pytest.raises(ValueError, match="status"):
        evaluate_policy(BrokenPolicy(), ToyEnv(), episodes=1)


def test_evaluate_policy_env_id(tmp_path: Path) -> None:
    """
    環境の名前を渡すと make_env で作って動かすことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    pytest.importorskip("gymnasium")
    # 台車の位置が狭い範囲を外れたら元の制御に戻す守り
    env = make_env("CartPole-v1")
    guards = make_guards(
        env.observation_space, env.action_space,
        observation_low=[-0.05, None, None, None],
        observation_high=[0.05, None, None, None], fallback="handover"
    )
    torch.manual_seed(0)
    directory = export_policy(
        tmp_path / "policy", nn.Linear(4, 2), env.observation_space,
        env.action_space, guards
    )
    policy = load_policy(directory, nn.Linear(4, 2))
    rows = evaluate_policy(policy, "CartPole-v1", episodes=2, seed=0)
    assert len(rows) == 2
    for row in rows:
        assert row["end"] in ("terminated", "truncated", "handover")
        assert row["length"] == row["ok"] + row["clipped"] + row["fallback"]


def test_evaluate_policy_env_id_without_gymnasium(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    gymnasium が無い場合に環境の名前では入れ方を言って止まることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        gymnasium を隠すのに使う

    Returns
    ----------
    None
    """
    policy = guarded_policy(tmp_path, 2.5, "handover")
    hide_modules(monkeypatch, {"gymnasium"})
    with pytest.raises(MissingRequirementError, match="gymnasium"):
        evaluate_policy(policy, "CartPole-v1", episodes=1)


def test_missing_requirement_error_at_top() -> None:
    """
    MissingRequirementError を ml_found の直下から読めることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    from ml_found.utils import MissingRequirementError as from_utils

    assert ml_found.MissingRequirementError is from_utils
    assert "MissingRequirementError" in ml_found.__all__
    assert issubclass(MissingRequirementError, ImportError)


def test_policy_metadata_rebuilds_model(tmp_path: Path) -> None:
    """
    推奨の metadata (組み立ての引数) から同じモデルを作り直して読めることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # README の例と同じ形の metadata で書き出し
    env = ToyEnv()
    args = {"hidden_dim": 3}
    directory = export_policy(
        tmp_path / "policy", RightModel(**args), env.observation_space,
        env.action_space,
        metadata={
            "env_id": "Toy-v0", "algorithm": "REINFORCE",
            "model": {"class": "test_policy_eval.RightModel", "args": args}
        }
    )

    # metadata の引数だけで組み立て直して読み, 同じ行動
    spec = read_policy_spec(directory)
    rebuilt = spec["metadata"]["model"]
    assert rebuilt["class"].rsplit(".", 1)[1] == spec["model_class"].rsplit(
        ".", 1
    )[1]
    policy = load_policy(directory, RightModel(**rebuilt["args"]))
    assert policy.model.hidden_dim == 3
    assert policy.act([0.0])["action"] == 1
    text = json.loads((directory / "policy.json").read_text(encoding="utf-8"))
    assert text["metadata"]["env_id"] == "Toy-v0"
