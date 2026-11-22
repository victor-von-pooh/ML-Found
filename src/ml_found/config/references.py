import os
import re
from typing import Any, Union

from omegaconf import DictConfig, ListConfig
from omegaconf.errors import (
    InterpolationKeyError, InterpolationResolutionError,
    InterpolationToMissingValueError, OmegaConfBaseException,
    UnsupportedInterpolationType, ValidationError
)

# resolver が出した誤りの種類を, OmegaConf の文の先頭から読むための形
RAISED_PATTERN = re.compile(r"^(\w+) raised while resolving interpolation")

# oc.env が無い環境変数で出す文から, 環境変数の名前を読むための形
ENV_PATTERN = re.compile(
    r"^KeyError raised while resolving interpolation: "
    r"\"Environment variable '([^']*)' not found\""
)


def is_missing_env(message: str, expression: str) -> bool:
    """
    参照の誤りが, oc.env の無い環境変数によるものかを判定する関数

    Parameters
    ----------
    message: str
        OmegaConf が出した誤りの文
    expression: str
        config に書いた参照 (${...})

    Returns
    ----------
    missing: bool
        oc.env の文の形で, その名前の環境変数が本当に無い場合は True,
        自作の resolver が出した KeyError などは False
    """
    # 文の形と参照の oc.env に加え, 名前の環境変数が無いことの確認
    # 値は読まずに有無だけを確認
    matched = ENV_PATTERN.match(message)
    missing = (
        matched is not None and "oc.env:" in expression
        and matched.group(1) not in os.environ
    )

    return missing


def describe_reason(error: OmegaConfBaseException, expression: str) -> str:
    """
    参照を解決できない理由を, 解決した値を含めずに言葉にする関数

    Parameters
    ----------
    error: OmegaConfBaseException
        参照を解決したときに OmegaConf が出した誤り
    expression: str
        config に書いた参照 (${...})

    Returns
    ----------
    reason: str
        理由, OmegaConf の文は解決した値 (環境変数の値など) を含み得るので
        使わず, 誤りの種類だけで決める
    """
    # OmegaConf が種類を分ける誤り
    if isinstance(error, InterpolationKeyError):
        reason = "参照する項目がありません"
        return reason
    if isinstance(error, InterpolationToMissingValueError):
        reason = "参照する項目の値が決まっていません (???)"
        return reason
    if isinstance(error, UnsupportedInterpolationType):
        reason = (
            "登録されていない resolver です (start_experiment の前に登録)"
        )
        return reason
    if isinstance(error, ValidationError):
        reason = "解決した値の型が項目の型に合いません"
        return reason

    # 循環と resolver が出した誤りは, 文の先頭の決まった形だけから判定
    message = str(error)
    if isinstance(error, InterpolationResolutionError):
        if message.startswith("Recursive interpolation detected"):
            reason = "参照が循環しています"
            return reason
        raised = RAISED_PATTERN.match(message)
        if raised is not None:
            if is_missing_env(message, expression):
                reason = "環境変数がありません (既定値を書くか環境変数を設定)"
                return reason
            reason = f"resolver が誤り ({raised.group(1)}) を出しました"
            return reason
    reason = f"解決できません ({type(error).__name__})"

    return reason


def find_unresolved(
    node: Union[DictConfig, ListConfig], prefix: str = ""
) -> list[str]:
    """
    config の中の解決できない参照を, 項目の場所と書いた参照で集める関数

    Parameters
    ----------
    node: Union[DictConfig, ListConfig]
        調べる config かその一部
    prefix: str = ""
        node の場所 (一番上は空)

    Returns
    ----------
    problems: list[str]
        「項目の場所: 書いた参照 (理由)」の文のリスト, 無ければ空
    """
    # null と値の決まっていない (???) 入れ物は中身が無いので対象外
    problems = []
    if node._is_none() or node._is_missing():
        return problems

    # 項目ごとに, 参照なら解決を試し, 入れ物なら中に進む
    keys: Any = (
        list(node.keys()) if isinstance(node, DictConfig)
        else range(len(node))
    )
    for key in keys:
        if isinstance(node, DictConfig):
            path = f"{prefix}.{key}" if prefix else str(key)
        else:
            path = f"{prefix}[{key}]"
        child = node._get_node(key)
        if child._is_interpolation():
            expression = str(child._value())
            try:
                node[key]
            except OmegaConfBaseException as e:
                reason = describe_reason(e, expression)
                problems.append(f"{path}: {expression!r} ({reason})")
            continue
        if isinstance(child, (DictConfig, ListConfig)):
            problems += find_unresolved(child, path)

    return problems


def check_references(cfg: DictConfig, label: str = "config") -> None:
    """
    config 全体の参照 (${...}) を解決できることを確かめる関数

    Parameters
    ----------
    cfg: DictConfig
        確かめる config
    label: str = "config"
        誤りの文で config を指す名前

    Returns
    ----------
    None
    """
    # 解決できない参照を全て並べて誤り
    # 秘密の値を出さないように, 解決した値と OmegaConf の文は含めない
    problems = find_unresolved(cfg)
    if problems:
        lines = "".join(f"\n\t{problem}" for problem in problems)
        raise ValueError(
            f"{label} の参照 (${{...}}) を解決できません "
            f"(解決した値は示しません):{lines}"
        )
