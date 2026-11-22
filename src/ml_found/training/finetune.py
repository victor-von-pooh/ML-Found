import contextlib
import logging
from pathlib import Path
from typing import Any, ContextManager, Optional

from omegaconf import DictConfig
import torch
from torch import nn

# 選べる計算の精度 (bf16 は順伝播と損失を自動の混合精度で計算)
PRECISIONS = ("fp32", "bf16")

# 誤りの文と train.log に並べるキーの数の上限
MAX_LISTED_KEYS = 10


def autocast_context(
    precision: str, device: torch.device
) -> ContextManager[Any]:
    """
    計算の精度に合わせて自動の混合精度の範囲を作る関数

    Parameters
    ----------
    precision: str
        計算の精度, fp32 か bf16
    device: torch.device
        計算するデバイス

    Returns
    ----------
    context: ContextManager[Any]
        bf16 の場合は torch.autocast, fp32 の場合は何もしない範囲
    """
    # fp32 は普通の計算のまま
    if precision != "bf16":
        context = contextlib.nullcontext()
        return context

    # bf16 は行列の積などを bfloat16 で計算 (重みは float32 のまま)
    context = torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    return context


def check_autocast(precision: str, device: torch.device) -> None:
    """
    計算の精度を学習のデバイスで使えるかを確かめる関数

    Parameters
    ----------
    precision: str
        計算の精度, fp32 か bf16
    device: torch.device
        学習に使うデバイス

    Returns
    ----------
    None
    """
    # 知らない値と fp32 は確認不要 (知らない値は check_trainer_config で停止)
    if precision != "bf16":
        return

    # 使えない場合に torch.autocast は警告だけで混合精度を外すことがあるので,
    # 小さな行列の積の結果の型で確認
    reason = None
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        reason = "この GPU は bfloat16 に対応していません"
    else:
        try:
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16
            ):
                a = torch.ones(2, 2, device=device)
                result = a @ a
            if result.dtype != torch.bfloat16:
                reason = "自動の混合精度が bfloat16 で計算しませんでした"
        except (RuntimeError, ValueError) as error:
            reason = f"{type(error).__name__}: {error}"
    if reason is not None:
        raise ValueError(
            f"trainer.precision: bf16 は {device} で使えません ({reason}). "
            "fp32 にしてください"
        )


def resolve_weights_path(path: str, project_dir: Path) -> Path:
    """
    読む重みのファイルの場所を, プロジェクトのディレクトリから辿って求める関数

    Parameters
    ----------
    path: str
        config に書いた場所
    project_dir: Path
        ML-Found を使っているプロジェクトのディレクトリ

    Returns
    ----------
    resolved: Path
        絶対パスの場所
    """
    # 相対パスは resume_from と同じくプロジェクトのディレクトリから
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path(project_dir) / resolved

    return resolved


def read_init_weights(path: str, project_dir: Path) -> dict:
    """
    学習を始める前に読む重みのファイルから, 重みの辞書を取り出す関数

    state_dict を保存したファイル (final_model.pt など) と, model を持つ
    ML-Found のチェックポイント (last.pt, best.pt) を読める

    Parameters
    ----------
    path: str
        config の trainer.init_weights.path
    project_dir: Path
        ML-Found を使っているプロジェクトのディレクトリ

    Returns
    ----------
    weights: dict
        キーごとのテンソル (cpu の上. 大きなファイルでも読み込まずに参照)
    """
    # ファイルがあることの確認
    resolved = resolve_weights_path(path, project_dir)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"trainer.init_weights.path のファイルがありません: {path!r} "
            "(相対パスはプロジェクトのディレクトリから辿ります)"
        )

    # 任意のコードが動かないようにテンソルと基本的な型だけを読み込み
    # 大きなファイルでもメモリに読み込まないように参照 (古い形は参照なし)
    try:
        state = torch.load(
            resolved, map_location="cpu", weights_only=True, mmap=True
        )
    except RuntimeError:
        state = torch.load(resolved, map_location="cpu", weights_only=True)

    # ML-Found のチェックポイントはモデルの重みを取り出し
    if (
        isinstance(state, dict) and isinstance(state.get("model"), dict)
        and ("monitor" in state or "weights_only" in state)
    ):
        state = state["model"]

    # キーが文字で値がテンソルの辞書だけを重みとして受け入れ
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError(
            f"trainer.init_weights.path は重みの辞書 (state_dict) ではありません:"
            f" {path!r}"
        )
    weights = state

    return weights


def listed(keys: list[str]) -> str:
    """
    キーを上限の数まで並べた文を作る関数

    Parameters
    ----------
    keys: list[str]
        並べるキー

    Returns
    ----------
    text: str
        上限を超えた分は数だけを添えた文
    """
    # 上限までを並べ, 残りは数だけ
    shown = ", ".join(keys[:MAX_LISTED_KEYS])
    rest = len(keys) - MAX_LISTED_KEYS
    text = shown if rest <= 0 else f"{shown} ほか {rest} 個"

    return text


def load_init_weights(
    model: nn.Module, init_cfg: DictConfig, project_dir: Path,
    logger: Optional[logging.Logger] = None
) -> dict:
    """
    trainer.init_weights の設定で, ファイルの重みの一部をモデルに読む関数

    Parameters
    ----------
    model: nn.Module
        重みを読むモデル
    init_cfg: DictConfig
        config の trainer.init_weights
    project_dir: Path
        ML-Found を使っているプロジェクトのディレクトリ
    logger: Optional[logging.Logger] = None
        読んだ結果を書くロガー

    Returns
    ----------
    report: dict
        読んだキー (loaded), 読まなかったモデルのキー (missing), 使わなかった
        ファイルのキー (unused)
    """
    # ファイルの重みのうち前置きで始まるものを, 前置きを付け替えて選択
    weights = read_init_weights(init_cfg.path, project_dir)
    source_prefix = init_cfg.source_prefix
    target_prefix = init_cfg.target_prefix
    selected = {
        target_prefix + key[len(source_prefix):]: value
        for key, value in weights.items() if key.startswith(source_prefix)
    }
    if not selected:
        raise ValueError(
            "trainer.init_weights.path に "
            f"{source_prefix!r} で始まる重みがありません"
        )

    # モデルにあるキーと無いキーに分け, 形の違いは誤り
    model_state = model.state_dict()
    loaded = [key for key in selected if key in model_state]
    unused = [key for key in selected if key not in model_state]
    missing = [key for key in model_state if key not in selected]
    mismatched = [
        f"{key} (ファイル {tuple(selected[key].shape)}, モデル "
        f"{tuple(model_state[key].shape)})"
        for key in loaded if selected[key].shape != model_state[key].shape
    ]
    if mismatched:
        raise ValueError(
            "trainer.init_weights.path の重みの形がモデルと違います: "
            f"{listed(mismatched)}"
        )
    if not loaded:
        raise ValueError(
            "trainer.init_weights.path の重みにモデルと同じ名前のものがありません"
            f" (付け替えた後のキー: {listed(unused)}). source_prefix と "
            "target_prefix を確かめてください"
        )
    if init_cfg.strict and (missing or unused):
        raise ValueError(
            "trainer.init_weights.strict: true ですが, キーがそろいません "
            f"(ファイルに無いモデルの重み: {listed(missing) or '無し'}, "
            f"モデルに無いファイルの重み: {listed(unused) or '無し'})"
        )

    # 選んだ重みだけを読み込み, 結果を記録
    model.load_state_dict(
        {key: selected[key] for key in loaded}, strict=False
    )
    if logger is not None:
        logger.info(
            f"trainer.init_weights.path ({init_cfg.path!r}) から {len(loaded)}"
            f" 個の重みを読みました. ファイルに無いモデルの重み "
            f"{len(missing)} 個, 使わなかったファイルの重み {len(unused)} 個"
        )
        if missing:
            logger.info(f"ファイルに無いモデルの重み: {listed(missing)}")
        if unused:
            logger.info(f"使わなかったファイルの重み: {listed(unused)}")
    report = {"loaded": loaded, "missing": missing, "unused": unused}

    return report


def frozen_parameters(model: nn.Module) -> dict[str, list[int]]:
    """
    学習しない (requires_grad が偽の) パラメータの名前と形を集める関数

    Parameters
    ----------
    model: nn.Module
        学習するモデル

    Returns
    ----------
    frozen: dict[str, list[int]]
        state_dict のキーごとの形 (共有したパラメータは全ての名前)
    """
    # 共有したパラメータは state_dict に全ての名前で入るので, 重複も含めて収集
    frozen = {
        name: list(parameter.shape)
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if not parameter.requires_grad
    }

    return frozen


def omit_keys(
    weights: Optional[dict], frozen: dict, prefix: str = ""
) -> Optional[dict]:
    """
    重みの辞書から学習しないパラメータを除く関数

    Parameters
    ----------
    weights: Optional[dict]
        重みの辞書, None の場合はそのまま
    frozen: dict
        frozen_parameters で集めた名前と形
    prefix: str = ""
        重みの辞書のキーの前置き (EMA の module. など)

    Returns
    ----------
    light: Optional[dict]
        学習しないパラメータを除いた写し (値は共有)
    """
    if weights is None:
        return weights
    omitted = {prefix + name for name in frozen}
    light = {
        key: value for key, value in weights.items() if key not in omitted
    }

    return light


def complete_weights(
    partial: dict, reference: dict, frozen: dict, prefix: str = "",
    label: str = "model", copy: bool = False
) -> dict:
    """
    学習するパラメータだけのチェックポイントの重みを, 今のモデルの学習しない
    パラメータで補って全ての重みにする関数

    Parameters
    ----------
    partial: dict
        チェックポイントの重み (学習しないパラメータを除いたもの)
    reference: dict
        今のモデルの state_dict (学習しないパラメータの値に使う)
    frozen: dict
        チェックポイントに残した, 除いたパラメータの名前と形
    prefix: str = ""
        重みの辞書のキーの前置き (EMA の module. など)
    label: str = "model"
        誤りの文に出す重みの名前
    copy: bool = False
        補う値を cpu に複製するか (学習で書き換わらないように保持する場合)

    Returns
    ----------
    weights: dict
        全ての重み
    """
    # 除いたパラメータは今のモデルにも同じ形で必要
    problems = []
    for name, shape in frozen.items():
        key = prefix + name
        if key not in reference:
            problems.append(f"{key} (今のモデルに無い)")
        elif list(reference[key].shape) != list(shape):
            problems.append(
                f"{key} (再開元 {tuple(shape)}, 今のモデル "
                f"{tuple(reference[key].shape)})"
            )
    if problems:
        raise ValueError(
            f"学習するパラメータだけのチェックポイントの {label} を補えません: "
            f"{listed(problems)}"
        )

    # 除いたもの以外で足りない重みは, 再開元に無いので誤り
    lacking = [
        key for key in reference
        if key not in partial and key[len(prefix):] not in frozen
    ]
    if prefix:
        lacking = [key for key in lacking if key.startswith(prefix)]
    if lacking:
        raise ValueError(
            f"チェックポイントの {label} に重みがありません: {listed(lacking)}"
        )

    # 除いたパラメータを今のモデルの値で補完
    weights = dict(partial)
    for name in frozen:
        key = prefix + name
        if key in weights:
            continue
        value = reference[key]
        if copy:
            value = value.detach().to("cpu", copy=True)
        weights[key] = value

    return weights
