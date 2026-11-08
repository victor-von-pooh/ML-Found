import copy
from pathlib import Path
from typing import Optional

import torch
from torch import nn


def check_mode(mode: str) -> None:
    """
    指標の良し悪しの向きが正しいことを確かめる関数

    Parameters
    ----------
    mode: str
        小さい方が良い場合は min, 大きい方が良い場合は max

    Returns
    -------
    None
    """
    # min と max 以外はエラーにする
    if mode not in ("min", "max"):
        raise ValueError(f"mode は min か max にしてください: {mode}")


def is_improved(
    value: float, best: Optional[float], mode: str, min_delta: float = 0.0
) -> bool:
    """
    指標がこれまでの最良値より改善したかどうかを判定する関数

    Parameters
    ----------
    value: float
        今回の指標の値
    best: Optional[float]
        これまでの最良値, まだ無い場合は None
    mode: str
        小さい方が良い場合は min, 大きい方が良い場合は max
    min_delta: float = 0.0
        これより小さい変化は改善とみなさない

    Returns
    -------
    improved: bool
        改善したかどうか
    """
    # 最良値がまだ無い場合は改善とみなす
    if best is None:
        improved = True
        return improved

    # 向きに応じて min_delta 以上良くなったかを判定する
    if mode == "min":
        improved = value < best - min_delta
    else:
        improved = value > best + min_delta

    return improved


def copy_state_dict(state_dict: dict) -> dict:
    """
    モデルの重みを学習で書き換わらないように複製する関数

    Parameters
    ----------
    state_dict: dict
        model.state_dict() で取得した重み

    Returns
    -------
    copied: dict
        CPU 上に複製した重み
    """
    # state_dict().copy() は辞書だけの複製で中身のテンソルは学習で書き換わるので,
    # テンソルごと CPU 上に複製する
    copied = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            copied[key] = value.detach().to("cpu", copy=True)
        else:
            copied[key] = copy.deepcopy(value)

    return copied


def save_checkpoint(state: dict, path: Path) -> None:
    """
    チェックポイントを保存する関数

    Parameters
    ----------
    state: dict
        保存する内容
    path: Path
        保存先のパス

    Returns
    -------
    None
    """
    # 書き込み中に止まっても壊れたファイルが残らないように一時ファイルを経由する
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    """
    チェックポイントを読み込む関数

    Parameters
    ----------
    path: str
        チェックポイントのパス
    device: torch.device
        テンソルを配置するデバイス

    Returns
    -------
    state: dict
        保存されていた内容
    """
    # 任意のコードが実行されないようにテンソルと基本的な型だけを読み込む
    state = torch.load(path, map_location=device, weights_only=True)

    return state


class EarlyStopping:
    def __init__(
        self, patience: int, min_delta: float = 0.0, mode: str = "min"
    ):
        # 指標の良し悪しの向きを確かめる
        check_mode(mode)

        # 待つ回数, 改善とみなす最小の変化, 良し悪しの向きを保存する
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        # これまでの最良値と, 改善しなかった回数を保存する
        self.best: Optional[float] = None
        self.counter = 0

    def step(self, value: float) -> bool:
        """
        指標の値を受け取って学習を止めるかどうかを判定する関数

        Parameters
        ----------
        value: float
            今回の指標の値

        Returns
        -------
        should_stop: bool
            学習を止めるかどうか
        """
        # 改善した場合は最良値を更新し, しなかった場合は回数を数える
        if is_improved(value, self.best, self.mode, self.min_delta):
            self.best = value
            self.counter = 0
        else:
            self.counter += 1

        # patience 回続けて改善しなかったら止める
        should_stop = self.counter >= self.patience

        return should_stop

    def state_dict(self) -> dict:
        """
        チェックポイントに保存するための状態を取得する関数

        Returns
        -------
        state: dict
            最良値と改善しなかった回数
        """
        # 最良値と回数をまとめる
        state = {"best": self.best, "counter": self.counter}

        return state

    def load_state_dict(self, state: dict) -> None:
        """
        チェックポイントから状態を戻す関数

        Parameters
        ----------
        state: dict
            state_dict で取得した状態

        Returns
        -------
        None
        """
        # 最良値と回数を戻す
        self.best = state["best"]
        self.counter = state["counter"]


class Checkpoint:
    def __init__(
        self, run_dir: Path, mode: str = "min", save_best: bool = True,
        save_last: bool = True
    ):
        # 指標の良し悪しの向きを確かめる
        check_mode(mode)

        # 保存先, 良し悪しの向き, 何を保存するかを保存する
        self.run_dir = Path(run_dir)
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last

        # これまでの最良値, そのエポック, その時の重みを保存する
        self.best_value: Optional[float] = None
        self.best_epoch = 0
        self.best_model_state: Optional[dict] = None

    def update(
        self, value: float, epoch: int, model: nn.Module, state: dict
    ) -> bool:
        """
        指標の値を受け取ってベストモデルと最後のモデルを保存する関数

        Parameters
        ----------
        value: float
            今回の指標の値
        epoch: int
            今回のエポック
        model: nn.Module
            学習中のモデル
        state: dict
            チェックポイントとして保存する内容

        Returns
        -------
        improved: bool
            ベストモデルが更新されたかどうか
        """
        # 改善した場合は最良値と重みを更新し, ベストモデルを保存する
        improved = is_improved(value, self.best_value, self.mode)
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.best_model_state = copy_state_dict(model.state_dict())
            if self.save_best:
                save_checkpoint(state, self.run_dir / "best.pt")

        # 学習を再開できるように最後のモデルを毎回上書きで保存する
        if self.save_last:
            save_checkpoint(state, self.run_dir / "last.pt")

        return improved

    def restore_best(self, model: nn.Module) -> None:
        """
        ベストモデルの重みをモデルに戻す関数

        Parameters
        ----------
        model: nn.Module
            重みを戻すモデル

        Returns
        -------
        None
        """
        # ベストモデルがまだ無い場合は何もしない
        if self.best_model_state is None:
            return

        # 保存しておいた重みを戻す
        model.load_state_dict(self.best_model_state)

    def state_dict(self) -> dict:
        """
        チェックポイントに保存するための状態を取得する関数

        Returns
        -------
        state: dict
            最良値とそのエポック
        """
        # 重みは容量が大きいので含めず, 最良値とエポックだけをまとめる
        state = {"best_value": self.best_value, "best_epoch": self.best_epoch}

        return state

    def load_state_dict(
        self, state: dict, best_model_state: Optional[dict] = None
    ) -> None:
        """
        チェックポイントから状態を戻す関数

        Parameters
        ----------
        state: dict
            state_dict で取得した状態
        best_model_state: Optional[dict] = None
            ベストモデルの重み, 再開前の best.pt から読み込んだもの

        Returns
        -------
        None
        """
        # 最良値とエポックを戻す
        self.best_value = state["best_value"]
        self.best_epoch = state["best_epoch"]

        # ベストモデルの重みがある場合は戻す
        if best_model_state is not None:
            self.best_model_state = copy_state_dict(best_model_state)
