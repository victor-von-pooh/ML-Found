from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import nn

# 方策の一式 (policy.json) の形の版
POLICY_FORMAT_VERSION = 1

# 一式のファイルの名前
SPEC_FILE = "policy.json"
WEIGHTS_FILE = "policy.pt"

# モデルの出力の読み方
HEADS = {"categorical", "deterministic"}

# 外れたときの動き
FALLBACK_KINDS = {"stop", "handover", "action"}


def bounds_to_json(values: Any) -> list[Optional[float]]:
    """
    範囲の配列を, 有限でない値を null にした JSON のリストにする関数

    Parameters
    ----------
    values: Any
        範囲の値の配列

    Returns
    ----------
    items: list[Optional[float]]
        平らにした値, 有限でない値は None (その向きに制限なし)
    """
    # inf と nan は JSON に書けないので制限なしの印に変換
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    items = [float(value) if math.isfinite(value) else None for value in flat]

    return items


def bounds_from_json(
    items: Optional[list], size: int, fill: float, name: str
) -> np.ndarray:
    """
    JSON の範囲のリストを, 制限なしを inf にした配列に戻す関数

    Parameters
    ----------
    items: Optional[list]
        範囲の値のリスト, None の場合は全て制限なし
    size: int
        要素の数
    fill: float
        制限なしの値 (下限は -inf, 上限は inf)
    name: str
        エラーのメッセージに出す名前

    Returns
    ----------
    bounds: np.ndarray
        平らな float64 の配列
    """
    # 無い場合は全て制限なし, ある場合は要素の数を確認
    if items is None:
        bounds = np.full(size, fill, dtype=np.float64)
        return bounds
    if len(items) != size:
        raise ValueError(
            f"{name} の要素の数 {len(items)} が空間の大きさ {size} と違います"
        )
    bounds = np.array(
        [fill if value is None else float(value) for value in items],
        dtype=np.float64
    )

    return bounds


def space_to_dict(space: Any) -> dict[str, Any]:
    """
    Gymnasium の空間 (Discrete か Box) を JSON にできる辞書にする関数

    Parameters
    ----------
    space: Any
        n を持つ Discrete の形か, shape, low, high を持つ Box の形の空間.
        辞書の場合はそのまま確かめて返す

    Returns
    ----------
    spec: dict[str, Any]
        discrete なら type, n, start. box なら type, shape, low, high
    """
    # 辞書はそのまま確認, Gymnasium は読み込まずに属性で見分け
    if isinstance(space, dict):
        spec = dict(space)
    elif isinstance(getattr(space, "n", None), (int, np.integer)):
        spec = {
            "type": "discrete", "n": int(space.n),
            "start": int(getattr(space, "start", 0))
        }
    elif hasattr(space, "low") and hasattr(space, "high"):
        spec = {
            "type": "box", "shape": [int(size) for size in space.shape],
            "low": bounds_to_json(space.low),
            "high": bounds_to_json(space.high)
        }
    else:
        raise ValueError(f"Discrete か Box の形の空間を渡してください: {space!r}")
    check_space(spec)

    return spec


def check_space(spec: dict) -> None:
    """
    空間の辞書の形を確かめる関数

    Parameters
    ----------
    spec: dict
        space_to_dict の形の辞書

    Returns
    ----------
    None
    """
    # 種類ごとに要る項目の確認
    if spec.get("type") == "discrete":
        if not isinstance(spec.get("n"), int) or spec["n"] < 1:
            raise ValueError(f"discrete の n は 1 以上の整数にしてください: {spec}")
        spec.setdefault("start", 0)
        return
    if spec.get("type") != "box":
        raise ValueError(f"空間の type は discrete か box にしてください: {spec}")
    shape = spec.get("shape")
    if not isinstance(shape, list) or not all(
        isinstance(size, int) and size >= 1 for size in shape
    ):
        raise ValueError(f"box の shape は正の整数のリストにしてください: {spec}")
    size = space_size(spec)
    low = bounds_from_json(spec.get("low"), size, -math.inf, "low")
    high = bounds_from_json(spec.get("high"), size, math.inf, "high")
    if np.any(low > high):
        raise ValueError(f"box の low が high より大きい要素があります: {spec}")


def space_size(spec: dict) -> int:
    """
    box の空間の要素の数を返す関数

    Parameters
    ----------
    spec: dict
        box の空間の辞書

    Returns
    ----------
    size: int
        shape の積
    """
    # 各次元の大きさの積
    size = int(np.prod(spec["shape"])) if spec["shape"] else 1

    return size


def make_guards(
    observation_space: Any, action_space: Any,
    action_low: Optional[Any] = None, action_high: Optional[Any] = None,
    allowed_actions: Optional[list[int]] = None,
    observation_low: Optional[Any] = None,
    observation_high: Optional[Any] = None,
    fallback: str = "stop", fallback_action: Optional[Any] = None
) -> dict[str, Any]:
    """
    方策の守り (行動の上限, 止める条件, 外れたときの動き) を作る関数

    Parameters
    ----------
    observation_space: Any
        観測の空間 (Box の形)
    action_space: Any
        行動の空間 (Discrete か Box の形)
    action_low: Optional[Any] = None
        box の行動の下限 (clip する), None の場合は空間の下限
    action_high: Optional[Any] = None
        box の行動の上限 (clip する), None の場合は空間の上限
    allowed_actions: Optional[list[int]] = None
        discrete で選んでよい行動, None の場合は全て
    observation_low: Optional[Any] = None
        観測の下限. 下回ったら外れたときの動きにする (要素ごとに None で
        制限なし)
    observation_high: Optional[Any] = None
        観測の上限. 上回ったら外れたときの動きにする
    fallback: str = "stop"
        外れたときの動き. stop (止める), handover (元の制御に戻す),
        action (決めた行動にする)
    fallback_action: Optional[Any] = None
        action の場合の行動, None の場合は box なら 0 の行動

    Returns
    ----------
    guards: dict[str, Any]
        JSON にできる守りの辞書
    """
    # 空間を辞書にして, 範囲を JSON の形に変換
    obs_spec = space_to_dict(observation_space)
    act_spec = space_to_dict(action_space)
    guards = {
        "action_low": None if action_low is None
        else bounds_to_json(action_low),
        "action_high": None if action_high is None
        else bounds_to_json(action_high),
        "allowed_actions": None if allowed_actions is None
        else [int(action) for action in allowed_actions],
        "observation_low": None if observation_low is None
        else bounds_to_json(observation_low),
        "observation_high": None if observation_high is None
        else bounds_to_json(observation_high),
        "fallback": {"kind": fallback, "action": None}
    }

    # 0 の行動を補ってから, 決めた行動を JSON の形に変換
    if fallback == "action":
        if fallback_action is None and act_spec["type"] == "box":
            fallback_action = np.zeros(space_size(act_spec))
        if fallback_action is None:
            raise ValueError("discrete の fallback_action を指定してください")
        guards["fallback"]["action"] = (
            int(fallback_action) if act_spec["type"] == "discrete"
            else bounds_to_json(fallback_action)
        )
    check_guards(guards, obs_spec, act_spec)

    return guards


def check_guards(guards: dict, obs_spec: dict, act_spec: dict) -> None:
    """
    守りの辞書が空間と食い違わないことを確かめる関数

    Parameters
    ----------
    guards: dict
        make_guards の形の守り
    obs_spec: dict
        観測の空間の辞書
    act_spec: dict
        行動の空間の辞書

    Returns
    ----------
    None
    """
    # 観測は box に限り, 止める条件の要素の数と向きを確認
    if obs_spec["type"] != "box":
        raise ValueError("観測の空間は box にしてください")
    obs_size = space_size(obs_spec)
    obs_low = bounds_from_json(
        guards.get("observation_low"), obs_size, -math.inf,
        "observation_low"
    )
    obs_high = bounds_from_json(
        guards.get("observation_high"), obs_size, math.inf,
        "observation_high"
    )
    if np.any(obs_low > obs_high):
        raise ValueError("observation_low が observation_high より大きい要素があります")

    # 行動の上限は空間の中で, 空間の種類に合うものだけ
    if act_spec["type"] == "discrete":
        check_discrete_guards(guards, act_spec)
    else:
        check_box_guards(guards, act_spec)

    # 外れたときの動きの種類
    fallback = guards.get("fallback") or {}
    if fallback.get("kind") not in FALLBACK_KINDS:
        raise ValueError(
            f"fallback は {sorted(FALLBACK_KINDS)} のどれかにしてください: "
            f"{fallback.get('kind')!r}"
        )


def check_discrete_guards(guards: dict, act_spec: dict) -> None:
    """
    discrete の行動の守りを確かめる関数

    Parameters
    ----------
    guards: dict
        守りの辞書
    act_spec: dict
        行動の空間の辞書

    Returns
    ----------
    None
    """
    # box の上限は書けず, 選んでよい行動と決めた行動は空間の中
    if guards.get("action_low") is not None or (
        guards.get("action_high") is not None
    ):
        raise ValueError("discrete の行動には action_low と action_high を書けません")
    valid = set(range(act_spec["start"], act_spec["start"] + act_spec["n"]))
    allowed = guards.get("allowed_actions")
    if allowed is not None and (not allowed or set(allowed) - valid):
        raise ValueError(
            f"allowed_actions は {sorted(valid)} の中から 1 つ以上にしてください: "
            f"{allowed}"
        )
    fallback = guards.get("fallback") or {}
    if fallback.get("kind") == "action":
        action = fallback.get("action")
        usable = valid if allowed is None else set(allowed)
        if action not in usable:
            raise ValueError(
                f"fallback の行動 {action!r} は選んでよい行動にしてください"
            )


def check_box_guards(guards: dict, act_spec: dict) -> None:
    """
    box の行動の守りを確かめる関数

    Parameters
    ----------
    guards: dict
        守りの辞書
    act_spec: dict
        行動の空間の辞書

    Returns
    ----------
    None
    """
    # 選んでよい行動は書けず, 上限は空間の範囲の中
    if guards.get("allowed_actions") is not None:
        raise ValueError("box の行動には allowed_actions を書けません")
    low, high = action_limits(guards, act_spec)
    if np.any(low > high):
        raise ValueError("action_low が action_high より大きい要素があります")

    # 決めた行動は上限の中
    fallback = guards.get("fallback") or {}
    if fallback.get("kind") == "action":
        action = fallback.get("action")
        if not isinstance(action, list) or len(action) != low.size:
            raise ValueError(
                f"fallback の行動は要素が {low.size} 個のリストにしてください"
            )
        values = np.array(
            [math.nan if value is None else value for value in action],
            dtype=np.float64
        )
        if not np.all(np.isfinite(values)) or np.any(
            (values < low) | (values > high)
        ):
            raise ValueError(
                f"fallback の行動 {action} は行動の上限の中にしてください"
            )


def action_limits(
    guards: dict, act_spec: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    box の行動の下限と上限を, 空間と守りの狭い方で求める関数

    Parameters
    ----------
    guards: dict
        守りの辞書
    act_spec: dict
        box の行動の空間の辞書

    Returns
    ----------
    low: np.ndarray
        行動の下限
    high: np.ndarray
        行動の上限
    """
    # 空間の範囲と守りの範囲の狭い方
    size = space_size(act_spec)
    low = np.maximum(
        bounds_from_json(act_spec.get("low"), size, -math.inf, "low"),
        bounds_from_json(
            guards.get("action_low"), size, -math.inf, "action_low"
        )
    )
    high = np.minimum(
        bounds_from_json(act_spec.get("high"), size, math.inf, "high"),
        bounds_from_json(
            guards.get("action_high"), size, math.inf, "action_high"
        )
    )

    return low, high


def check_spec(spec: dict) -> None:
    """
    policy.json の中身の版と形を確かめる関数

    Parameters
    ----------
    spec: dict
        policy.json を読んだ辞書

    Returns
    ----------
    None
    """
    # 版が無いものと新しい版は, 守りを読み違えないように止める
    version = spec.get("format_version")
    if not isinstance(version, int):
        raise ValueError("policy.json に format_version (整数) がありません")
    if version > POLICY_FORMAT_VERSION:
        raise ValueError(
            f"policy.json の版 {version} はこの ML-Found が読める版 "
            f"{POLICY_FORMAT_VERSION} より新しいです. ML-Found を新しくして"
            "ください (守りを読み違えないように止めます)"
        )

    # 出力の読み方と行動の空間の組み合わせ, 空間と守り
    obs_spec = spec["observation_space"]
    act_spec = spec["action_space"]
    check_space(obs_spec)
    check_space(act_spec)
    head = spec.get("head")
    expected = "categorical" if act_spec["type"] == "discrete" else (
        "deterministic"
    )
    if head != expected:
        raise ValueError(
            f"{act_spec['type']} の行動の head は {expected} にしてください: "
            f"{head!r}"
        )
    check_guards(spec["guards"], obs_spec, act_spec)


def export_policy(
    path: Union[str, Path], model: nn.Module, observation_space: Any,
    action_space: Any, guards: Optional[dict] = None,
    metadata: Optional[dict] = None
) -> Path:
    """
    方策の重み, 観測と行動の空間, 守りを一式のディレクトリに書き出す関数

    Parameters
    ----------
    path: Union[str, Path]
        一式のディレクトリ (実験のディレクトリの policy/ を推奨)
    model: nn.Module
        方策のモデル (discrete なら行動ごとのスコア, box なら行動を出す)
    observation_space: Any
        観測の空間 (Box の形)
    action_space: Any
        行動の空間 (Discrete か Box の形)
    guards: Optional[dict] = None
        make_guards で作った守り, None の場合は止める条件なしで,
        外れたとき (観測が有限でないなど) は止める
    metadata: Optional[dict] = None
        JSON にできる補足 (元の実験, モデルの組み立て方など)

    Returns
    ----------
    directory: Path
        書き出したディレクトリ
    """
    # 空間, 出力の読み方, 守りをまとめて確認
    obs_spec = space_to_dict(observation_space)
    act_spec = space_to_dict(action_space)
    if guards is None:
        guards = make_guards(obs_spec, act_spec)
    model_class = type(model)
    spec = {
        "format_version": POLICY_FORMAT_VERSION,
        "kind": "policy",
        "head": "categorical" if act_spec["type"] == "discrete"
        else "deterministic",
        "observation_space": obs_spec,
        "action_space": act_spec,
        "guards": guards,
        "weights": WEIGHTS_FILE,
        "model_class": f"{model_class.__module__}.{model_class.__qualname__}",
        "created_at": datetime.now().astimezone().isoformat(),
        "metadata": metadata or {}
    }
    check_spec(spec)

    # 重みは cpu のテンソルで保存し, 中身の文書を書く
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    weights = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    torch.save(weights, directory / WEIGHTS_FILE)
    text = json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False)
    (directory / SPEC_FILE).write_text(text + "\n", encoding="utf-8")

    return directory


def read_policy_spec(path: Union[str, Path]) -> dict[str, Any]:
    """
    一式のディレクトリから policy.json を読んで確かめる関数

    Parameters
    ----------
    path: Union[str, Path]
        一式のディレクトリ

    Returns
    ----------
    spec: dict[str, Any]
        policy.json の中身
    """
    # 読んで版と形を確認
    spec = json.loads(
        (Path(path) / SPEC_FILE).read_text(encoding="utf-8")
    )
    check_spec(spec)

    return spec


def check_observation(spec: dict, observation: Any) -> list[str]:
    """
    観測が止める条件に当たるかを確かめる関数

    Parameters
    ----------
    spec: dict
        policy.json の中身
    observation: Any
        環境の観測

    Returns
    ----------
    reasons: list[str]
        当たった条件, 当たらなければ空
    """
    # 形と有限であることの確認
    obs_spec = spec["observation_space"]
    size = space_size(obs_spec)
    obs = np.asarray(observation, dtype=np.float64)
    if obs.size != size:
        reasons = [f"観測の大きさ {obs.size} が {size} と違う"]
        return reasons
    obs = obs.reshape(-1)
    if not np.all(np.isfinite(obs)):
        reasons = ["観測に有限でない値がある"]
        return reasons

    # 止める条件の範囲の外の要素
    guards = spec["guards"]
    low = bounds_from_json(
        guards.get("observation_low"), size, -math.inf, "observation_low"
    )
    high = bounds_from_json(
        guards.get("observation_high"), size, math.inf, "observation_high"
    )
    outside = np.nonzero((obs < low) | (obs > high))[0]
    reasons = [
        f"観測の要素 {int(index)} が範囲の外 ({obs[index]:.6g})"
        for index in outside
    ]

    return reasons


def fallback_result(spec: dict, reasons: list[str]) -> dict[str, Any]:
    """
    外れたときの動きの結果を作る関数

    Parameters
    ----------
    spec: dict
        policy.json の中身
    reasons: list[str]
        外れた理由

    Returns
    ----------
    result: dict[str, Any]
        action (stop と handover は None), status (stop, handover,
        fallback), reasons
    """
    # 止める, 元の制御に戻す, 決めた行動にする, のどれか
    fallback = spec["guards"]["fallback"]
    action = None
    status = fallback["kind"]
    if fallback["kind"] == "action":
        status = "fallback"
        action = to_action(spec, fallback["action"])
    result = {"action": action, "status": status, "reasons": reasons}

    return result


def to_action(spec: dict, value: Any) -> Union[int, np.ndarray]:
    """
    行動の値を空間の型にする関数

    Parameters
    ----------
    spec: dict
        policy.json の中身
    value: Any
        行動の値

    Returns
    ----------
    action: Union[int, np.ndarray]
        discrete なら整数, box なら shape の形の float32 の配列
    """
    # 空間の種類ごとの型
    act_spec = spec["action_space"]
    if act_spec["type"] == "discrete":
        action = int(value)
        return action
    action = np.asarray(value, dtype=np.float32).reshape(act_spec["shape"])

    return action


def apply_guards(
    spec: dict, observation: Any, output: Any
) -> dict[str, Any]:
    """
    観測とモデルの出力に守りを当てて, 行動を決める関数 (純粋な関数)

    Parameters
    ----------
    spec: dict
        policy.json の中身
    observation: Any
        環境の観測
    output: Any
        モデルの出力 (discrete なら行動ごとのスコア, box なら行動)

    Returns
    ----------
    result: dict[str, Any]
        action (行動. 止めるか元の制御に戻す場合は None), status (ok,
        clipped, fallback, stop, handover), reasons (守りが働いた理由)
    """
    # 止める条件に当たった場合は外れたときの動き
    reasons = check_observation(spec, observation)
    if reasons:
        result = fallback_result(spec, reasons)
        return result

    # 出力が有限でない場合も外れたときの動き
    values = np.asarray(output, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        result = fallback_result(spec, ["モデルの出力に有限でない値がある"])
        return result

    # 空間の種類ごとに, 行動の上限を当てて決める
    if spec["action_space"]["type"] == "discrete":
        result = guard_discrete(spec, values)
    else:
        result = guard_box(spec, values)

    return result


def guard_discrete(spec: dict, scores: np.ndarray) -> dict[str, Any]:
    """
    discrete の行動を, 選んでよい行動の中の最もスコアの高いものにする関数

    Parameters
    ----------
    spec: dict
        policy.json の中身
    scores: np.ndarray
        行動ごとのスコア

    Returns
    ----------
    result: dict[str, Any]
        action, status, reasons
    """
    # スコアの数の確認
    act_spec = spec["action_space"]
    if scores.size != act_spec["n"]:
        result = fallback_result(
            spec, [f"モデルの出力の大きさ {scores.size} が {act_spec['n']} と違う"]
        )
        return result

    # 選んでよい行動以外を除いて最もスコアの高い行動
    start = act_spec["start"]
    best = int(np.argmax(scores)) + start
    allowed = spec["guards"].get("allowed_actions")
    if allowed is None or best in allowed:
        result = {"action": best, "status": "ok", "reasons": []}
        return result
    masked = np.full(scores.shape, -math.inf)
    indices = [action - start for action in allowed]
    masked[indices] = scores[indices]
    action = int(np.argmax(masked)) + start
    result = {
        "action": action, "status": "clipped",
        "reasons": [f"行動 {best} は選んでよい行動でない"]
    }

    return result


def guard_box(spec: dict, values: np.ndarray) -> dict[str, Any]:
    """
    box の行動を上限の中に clip する関数

    Parameters
    ----------
    spec: dict
        policy.json の中身
    values: np.ndarray
        モデルが出した行動

    Returns
    ----------
    result: dict[str, Any]
        action, status, reasons
    """
    # 大きさの確認
    act_spec = spec["action_space"]
    low, high = action_limits(spec["guards"], act_spec)
    if values.size != low.size:
        result = fallback_result(
            spec, [f"モデルの出力の大きさ {values.size} が {low.size} と違う"]
        )
        return result

    # 上限の中に clip して, 変えた要素を理由に記録
    clipped = np.clip(values, low, high)
    changed = np.nonzero(clipped != values)[0]
    reasons = [
        f"行動の要素 {int(index)} を {values[index]:.6g} から "
        f"{clipped[index]:.6g} に clip"
        for index in changed
    ]
    result = {
        "action": to_action(spec, clipped),
        "status": "clipped" if reasons else "ok",
        "reasons": reasons
    }

    return result


class GuardedPolicy:
    def __init__(self, model: nn.Module, spec: dict):
        # 推論モードのモデルと, policy.json の中身を保存
        self.model = model.eval()
        self.spec = spec

    def act(self, observation: Any) -> dict[str, Any]:
        """
        観測から, 守りを当てた行動を決める関数

        Parameters
        ----------
        observation: Any
            環境の観測

        Returns
        ----------
        result: dict[str, Any]
            action, status (ok, clipped, fallback, stop, handover), reasons
        """
        # 止める条件に当たる場合はモデルを動かさずに外れたときの動き
        reasons = check_observation(self.spec, observation)
        if reasons:
            result = fallback_result(self.spec, reasons)
            return result

        # 勾配を記録せずにモデルを動かして守りを適用
        shape = self.spec["observation_space"]["shape"]
        obs = torch.as_tensor(
            np.asarray(observation, dtype=np.float32).reshape(shape)
        )
        with torch.no_grad():
            output = self.model(obs.unsqueeze(0))[0]
        result = apply_guards(
            self.spec, observation, output.detach().cpu().numpy()
        )

        return result


def load_policy(path: Union[str, Path], model: nn.Module) -> GuardedPolicy:
    """
    一式を読み, 守りを当てて行動を返す方策を作る関数

    実機での実行は道具の外. 返す方策は, 観測から行動と守りの結果を返す
    だけで, 環境や機械には触れない

    Parameters
    ----------
    path: Union[str, Path]
        一式のディレクトリ
    model: nn.Module
        書き出したときと同じ組み立てのモデル (重みは一式から読む. cpu で動かす)

    Returns
    ----------
    policy: GuardedPolicy
        act(observation) で行動を返す方策
    """
    # 版と形を確かめてから重みを読み込み
    directory = Path(path)
    spec = read_policy_spec(directory)
    state = torch.load(
        directory / spec["weights"], map_location="cpu", weights_only=True
    )
    model = model.cpu()
    model.load_state_dict(state)
    policy = GuardedPolicy(model, spec)

    return policy
