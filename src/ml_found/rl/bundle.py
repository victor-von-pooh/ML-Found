from datetime import datetime
import hashlib
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

# 空間, 守り, 外れたときの動きに書ける項目 (書き損じを見つけるため,
# 他の項目はエラー)
SPACE_KEYS = {
    "discrete": {"type", "n", "start"},
    "box": {"type", "shape", "low", "high"}
}
GUARD_KEYS = {
    "action_low", "action_high", "allowed_actions", "observation_low",
    "observation_high", "fallback"
}
FALLBACK_KEYS = {"kind", "action"}

# 数として読める配列の種類 (真偽値, 整数, 小数)
NUMERIC_KINDS = "biuf"


def bounds_to_json(values: Any) -> list[Optional[float]]:
    """
    範囲の配列を, 無限大と None を null にした JSON のリストにする関数

    Parameters
    ----------
    values: Any
        範囲の値の配列

    Returns
    ----------
    items: list[Optional[float]]
        平らにした値, 無限大は None (その向きに制限なし)
    """
    # None は制限なしのまま無限大に置き換え, nan は守りを黙って外すことに
    # なるのでエラー, 無限大は制限なしの印
    raw = np.asarray(values, dtype=object).reshape(-1)
    flat = np.array(
        [math.inf if value is None else value for value in raw],
        dtype=np.float64
    )
    if np.any(np.isnan(flat)):
        raise ValueError(f"範囲に nan は使えません (制限なしは None): {values}")
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
    if not isinstance(items, list) or len(items) != size:
        raise ValueError(
            f"{name} は要素が {size} 個のリストにしてください: {items!r}"
        )

    # 要素は有限の数か None だけ (nan, 無限大, 真偽値, 文字はエラー)
    for value in items:
        is_number = isinstance(value, (int, float)) and not isinstance(
            value, bool
        )
        if value is not None and not (is_number and math.isfinite(value)):
            raise ValueError(
                f"{name} の要素は有限の数か null にしてください: {value!r}"
            )
    bounds = np.array(
        [fill if value is None else float(value) for value in items],
        dtype=np.float64
    )

    return bounds


def is_int(value: Any) -> bool:
    """
    真偽値を除く整数かを確かめる関数

    Parameters
    ----------
    value: Any
        確かめる値

    Returns
    ----------
    result: bool
        int (と NumPy の整数) で, 真偽値でない場合は True
    """
    # bool は int の子なので除外
    result = isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )

    return result


def reject_constant(name: str) -> Any:
    """
    JSON の外の値 (NaN, Infinity) を読んだらエラーにする関数

    Parameters
    ----------
    name: str
        読んだ値の名前

    Returns
    ----------
    value: Any
        返らない
    """
    # 守りの範囲を黙って外さないための停止
    raise ValueError(f"policy.json に {name} は書けません (制限なしは null)")


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
    # 辞書はそのまま確認, Gymnasium は読み込まずに属性での見分け
    # (MultiBinary なども n を持つので, 形が () の場合だけ discrete)
    shape = tuple(getattr(space, "shape", ()) or ())
    if isinstance(space, dict):
        spec = dict(space)
    elif isinstance(getattr(space, "n", None), (int, np.integer)) and (
        shape == ()
    ):
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
    # 辞書であることと, 種類ごとに書ける項目だけであることの確認
    if not isinstance(spec, dict):
        raise ValueError(f"空間は辞書にしてください: {spec!r}")
    if spec.get("type") not in SPACE_KEYS:
        raise ValueError(f"空間の type は discrete か box にしてください: {spec}")
    unknown = sorted(set(spec) - SPACE_KEYS[spec["type"]])
    if unknown:
        raise ValueError(f"空間に知らない項目 {unknown} があります: {spec}")

    # 種類ごとに要る項目の確認
    if spec.get("type") == "discrete":
        if not is_int(spec.get("n")) or spec["n"] < 1:
            raise ValueError(f"discrete の n は 1 以上の整数にしてください: {spec}")
        spec.setdefault("start", 0)
        if not is_int(spec["start"]):
            raise ValueError(f"discrete の start は整数にしてください: {spec}")
        return
    shape = spec.get("shape")
    if not isinstance(shape, list) or not all(
        is_int(size) and size >= 1 for size in shape
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
    方策の守り (行動の範囲, 止める条件, 外れたときの動き) を作る関数

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
        if act_spec["type"] == "discrete":
            if int(fallback_action) != fallback_action:
                raise ValueError(
                    f"fallback_action は整数にしてください: {fallback_action}"
                )
            guards["fallback"]["action"] = int(fallback_action)
        else:
            guards["fallback"]["action"] = bounds_to_json(fallback_action)
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
    # 守りと外れたときの動きは辞書で, 書ける項目だけ
    if not isinstance(guards, dict):
        raise ValueError(f"guards は辞書にしてください: {guards!r}")
    unknown = sorted(set(guards) - GUARD_KEYS)
    if unknown:
        raise ValueError(f"guards に知らない項目 {unknown} があります")
    fallback = guards.get("fallback")
    if not isinstance(fallback, dict):
        raise ValueError(f"guards の fallback は辞書にしてください: {fallback!r}")
    unknown = sorted(set(fallback) - FALLBACK_KEYS)
    if unknown:
        raise ValueError(f"fallback に知らない項目 {unknown} があります")

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

    # 行動の範囲は空間の中で, 空間の種類に合うものだけ
    if act_spec["type"] == "discrete":
        check_discrete_guards(guards, act_spec)
    else:
        check_box_guards(guards, act_spec)

    # 外れたときの動きの種類
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
    # box の範囲は書けず, 選んでよい行動と決めた行動は空間の中
    if guards.get("action_low") is not None or (
        guards.get("action_high") is not None
    ):
        raise ValueError("discrete の行動には action_low と action_high を書けません")
    valid = set(range(act_spec["start"], act_spec["start"] + act_spec["n"]))
    allowed = guards.get("allowed_actions")
    if allowed is not None and (
        not isinstance(allowed, list) or not allowed
        or not all(is_int(action) for action in allowed)
        or set(allowed) - valid
    ):
        raise ValueError(
            f"allowed_actions は {sorted(valid)} の中から 1 つ以上にしてください: "
            f"{allowed}"
        )
    fallback = guards.get("fallback") or {}
    if fallback.get("kind") == "action":
        action = fallback.get("action")
        usable = valid if allowed is None else set(allowed)
        if not is_int(action) or action not in usable:
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
    # 選んでよい行動は書けず, 範囲は空間の範囲の中
    if guards.get("allowed_actions") is not None:
        raise ValueError("box の行動には allowed_actions を書けません")
    low, high = action_limits(guards, act_spec)
    if np.any(low > high):
        raise ValueError("action_low が action_high より大きい要素があります")

    # 決めた行動は範囲の中
    fallback = guards.get("fallback") or {}
    if fallback.get("kind") == "action":
        action = fallback.get("action")
        if not isinstance(action, list) or len(action) != low.size:
            raise ValueError(
                f"fallback の行動は要素が {low.size} 個のリストにしてください"
            )
        values = bounds_from_json(action, low.size, math.nan, "fallback")
        if not np.all(np.isfinite(values)) or np.any(
            (values < low) | (values > high)
        ):
            raise ValueError(
                f"fallback の行動 {action} は行動の範囲の中にしてください"
            )
        if float32_inside(values, low, high) is None:
            raise ValueError(
                f"fallback の行動 {action} は float32 で行動の範囲の中に"
                "表せません"
            )


def float32_inside(
    values: np.ndarray, low: np.ndarray, high: np.ndarray
) -> Optional[np.ndarray]:
    """
    行動を float32 にし, 丸めで範囲の外に出た要素を内側に寄せる関数

    0.1 のような範囲は float32 で少し大きな値 (0.10000000149) になるので,
    丸めた後に範囲を確かめ直し, 外に出た要素は内側の隣の float32 に寄せる

    Parameters
    ----------
    values: np.ndarray
        範囲の中の行動 (float64)
    low: np.ndarray
        行動の下限
    high: np.ndarray
        行動の上限

    Returns
    ----------
    action: Optional[np.ndarray]
        float32 の行動, 有限でないか範囲の中に表せない場合は None
    """
    # 大きすぎる値は無限大になるので, 警告を出さずに丸めて確認
    with np.errstate(over="ignore"):
        action = np.asarray(values, dtype=np.float64).astype(np.float32)
    if not np.all(np.isfinite(action)):
        action = None
        return action

    # 丸めで外に出た要素を, 内側の隣の float32 に移動
    above = action.astype(np.float64) > high
    action[above] = np.nextafter(action[above], np.float32(-np.inf))
    below = action.astype(np.float64) < low
    action[below] = np.nextafter(action[below], np.float32(np.inf))

    # それでも外の要素 (範囲に float32 が無い場合) は表せない
    widened = action.astype(np.float64)
    if np.any((widened > high) | (widened < low)):
        action = None
        return action

    return action


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
    # 版が無いものと新しい版は, 守りを読み違えないように停止
    if not isinstance(spec, dict):
        raise ValueError("policy.json は辞書にしてください")
    version = spec.get("format_version")
    if not is_int(version) or version < 1:
        raise ValueError(
            "policy.json に format_version (1 以上の整数) がありません"
        )
    if version > POLICY_FORMAT_VERSION:
        raise ValueError(
            f"policy.json の版 {version} はこの ML-Found が読める版 "
            f"{POLICY_FORMAT_VERSION} より新しいです. ML-Found を新しくして"
            "ください (守りを読み違えないように止めます)"
        )

    # 出力の読み方と行動の空間の組み合わせ, 空間と守り
    obs_spec = spec.get("observation_space")
    act_spec = spec.get("action_space")
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
    check_guards(spec.get("guards"), obs_spec, act_spec)

    # 重みは一式の中の決まったファイルだけで, 中身の sha256 を持つこと
    if spec.get("weights") != WEIGHTS_FILE:
        raise ValueError(
            f"policy.json の weights は {WEIGHTS_FILE} にしてください: "
            f"{spec.get('weights')!r}"
        )
    digest = spec.get("weights_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or not all(
        char in "0123456789abcdef" for char in digest
    ):
        raise ValueError(
            "policy.json の weights_sha256 は 16 進の小文字 64 文字に"
            f"してください: {digest!r}"
        )


def file_sha256(path: Path) -> str:
    """
    ファイルの中身の sha256 を求める関数

    Parameters
    ----------
    path: Path
        ファイルのパス

    Returns
    ----------
    digest: str
        16 進の小文字 64 文字
    """
    # 大きな重みでも少しずつ読んで計算
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(ReadChunk(file), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()

    return digest


class ReadChunk:
    def __init__(self, file: Any, size: int = 1 << 20):
        # 読むファイルと 1 回に読む大きさを保存
        self.file = file
        self.size = size

    def __call__(self) -> bytes:
        """
        ファイルから次の塊を読む関数

        Parameters
        ----------
        None

        Returns
        ----------
        chunk: bytes
            読んだ塊, 終わりでは空
        """
        # 決めた大きさだけ読み込み
        chunk = self.file.read(self.size)

        return chunk


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
        "weights_sha256": "0" * 64,
        "model_class": f"{model_class.__module__}.{model_class.__qualname__}",
        "created_at": datetime.now().astimezone().isoformat(),
        "metadata": metadata or {}
    }
    check_spec(spec)

    # 重みを cpu のテンソルで保存し, 中身の文書を書き出し
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    weights = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    torch.save(weights, directory / WEIGHTS_FILE)
    spec["weights_sha256"] = file_sha256(directory / WEIGHTS_FILE)
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
    # NaN と Infinity は JSON の外なので拒んで読み, 版と形を確認
    spec = json.loads(
        (Path(path) / SPEC_FILE).read_text(encoding="utf-8"),
        parse_constant=reject_constant
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
    # 数の配列にできること, 形と有限であることの確認
    # 形は空間と同じか, 同じ大きさの平らな配列だけ
    obs_spec = spec["observation_space"]
    size = space_size(obs_spec)
    try:
        raw = np.asarray(observation)
        obs = raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        reasons = [f"観測を数の配列にできない ({exc})"]
        return reasons
    if raw.dtype.kind not in NUMERIC_KINDS:
        reasons = [f"観測が数の配列でない ({raw.dtype})"]
        return reasons
    if obs.shape not in (tuple(obs_spec["shape"]), (size,)):
        reasons = [
            f"観測の形 {obs.shape} が {tuple(obs_spec['shape'])} と違う"
        ]
        return reasons
    # モデルには float32 で渡すので, 丸めた後も有限であることを確認
    # (1e39 は float64 では有限でも float32 では無限大)
    obs = obs.reshape(-1)
    with np.errstate(over="ignore"):
        obs32 = obs.astype(np.float32)
    if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(obs32)):
        reasons = ["観測に有限でない値がある (float32 での丸めも含む)"]
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
    # 空間の種類ごとの型, box は float32 にしても範囲の中に収める
    act_spec = spec["action_space"]
    if act_spec["type"] == "discrete":
        action = int(value)
        return action
    low, high = action_limits(spec["guards"], act_spec)
    inside = float32_inside(np.asarray(value, dtype=np.float64), low, high)
    if inside is None:
        raise ValueError(f"行動 {value} は float32 で範囲の中に表せません")
    action = inside.reshape(act_spec["shape"])

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

    # 出力が数の配列でないか, 有限でない場合も外れたときの動き
    try:
        values = np.asarray(output, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        result = fallback_result(spec, [f"モデルの出力を数にできない ({exc})"])
        return result
    if not np.all(np.isfinite(values)):
        result = fallback_result(spec, ["モデルの出力に有限でない値がある"])
        return result

    # 空間の種類ごとに, 行動の範囲を当てて決める
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
    box の行動を範囲の中に clip する関数

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

    # 範囲の中に clip し, float32 にしても有限で範囲の中であることを確認
    clipped = np.clip(values, low, high)
    inside = float32_inside(clipped, low, high)
    if inside is None:
        result = fallback_result(
            spec, ["clip した行動が float32 で有限な範囲の中に表せない"]
        )
        return result
    changed = np.nonzero(clipped != values)[0]
    reasons = [
        f"行動の要素 {int(index)} を {values[index]:.6g} から "
        f"{clipped[index]:.6g} に clip"
        for index in changed
    ]
    result = {
        "action": inside.reshape(act_spec["shape"]),
        "status": "clipped" if reasons else "ok",
        "reasons": reasons
    }

    return result


class GuardedPolicy:
    def __init__(self, model: nn.Module, spec: dict):
        # cpu の推論モードのモデルと, policy.json の中身を保存
        self.model = model.cpu().eval()
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
        # モデルのエラーも外れたときの動き
        shape = self.spec["observation_space"]["shape"]
        obs = torch.as_tensor(
            np.asarray(observation, dtype=np.float32).reshape(shape)
        )
        try:
            with torch.no_grad():
                output = self.model(obs.unsqueeze(0))[0]
        except Exception as exc:
            result = fallback_result(
                self.spec, [f"モデルのエラー ({type(exc).__name__}: {exc})"]
            )
            return result
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
    # 版と形と, 重みのファイルの sha256 を確かめてから読み込み
    directory = Path(path)
    spec = read_policy_spec(directory)
    digest = file_sha256(directory / spec["weights"])
    if digest != spec["weights_sha256"]:
        raise ValueError(
            f"{spec['weights']} の sha256 が policy.json と違います "
            "(書き出した後に重みが変わっています)"
        )
    state = torch.load(
        directory / spec["weights"], map_location="cpu", weights_only=True
    )
    model = model.cpu()
    model.load_state_dict(state)
    policy = GuardedPolicy(model, spec)

    return policy
