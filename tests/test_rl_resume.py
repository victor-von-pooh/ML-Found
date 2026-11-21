from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest
import torch
from torch import nn

from ml_found.rl import (
    EvaluationLoader, Exploration, ReinforceStep, RolloutLoader, make_env
)
from ml_found.rl.rollout import env_layers
from ml_found.training.history import History
from ml_found.training.trainer import Trainer
from test_rl import Space
from test_trainer import make_experiment

# 学習と検証のエピソードの指標
METRICS = (
    "train/loss", "train/reward", "train/length", "valid/reward",
    "valid/entropy"
)


class CarryEnv:
    def __init__(self):
        # 観測と行動の空間, 環境の乱数, エピソードを越えて続く数を用意
        self.observation_space = Space(low=[-10.0, -10.0], high=[10.0, 10.0])
        self.action_space = Space(n=2)
        self.np_random = np.random.default_rng()
        self.episodes = 0
        self.t = 0
        self.length = 3

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        """
        環境の乱数で長さを決めてエピソードを始める関数

        Parameters
        ----------
        seed: Optional[int] = None
            乱数の種, None の場合は環境の乱数の続き

        Returns
        ----------
        observation: np.ndarray
            エピソードの数と乱数の値の観測
        info: dict
            空の辞書
        """
        # 種があれば乱数を作り直し, エピソードの数を進めて長さを決定
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.episodes += 1
        self.t = 0
        self.length = int(self.np_random.integers(2, 6))
        observation = self.observation()
        info: dict = {}

        return observation, info

    def observation(self) -> np.ndarray:
        """
        エピソードの数と歩数から観測を作る関数

        Parameters
        ----------
        None

        Returns
        ----------
        observation: np.ndarray
            エピソードの数を 10 で割った余りと, 歩数
        """
        # エピソードを越えて続く数が観測に出るように
        observation = np.array(
            [float(self.episodes % 10), float(self.t)], dtype=np.float32
        )

        return observation

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        行動 1 で報酬 1 の 1 歩を進める関数

        Parameters
        ----------
        action: int
            行動

        Returns
        ----------
        result: tuple[np.ndarray, float, bool, bool, dict]
            観測, 報酬, 終了, 打ち切り, info
        """
        # 歩数を進めて, 決めた長さで終了
        self.t += 1
        result = (
            self.observation(), float(action), self.t >= self.length, False,
            {}
        )

        return result

    def state_dict(self) -> dict:
        """
        エピソードを越えて続く状態を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        state: dict
            エピソードの数
        """
        # 保存する状態
        state = {"episodes": self.episodes}

        return state

    def load_state_dict(self, state: dict) -> None:
        """
        state_dict の状態に戻す関数

        Parameters
        ----------
        state: dict
            state_dict の値

        Returns
        ----------
        None
        """
        # エピソードの数の復元
        self.episodes = int(state["episodes"])


class PlainEnv(CarryEnv):
    # state_dict を持たない環境 (エピソードの数は戻らない)
    state_dict = None
    load_state_dict = None


def carry_loaders(
    model: nn.Module, env_class: type = CarryEnv
) -> tuple[RolloutLoader, EvaluationLoader]:
    """
    CarryEnv の学習と検証の loader を作る関数

    Parameters
    ----------
    model: nn.Module
        方策のモデル
    env_class: type = CarryEnv
        環境のクラス

    Returns
    ----------
    loaders: tuple[RolloutLoader, EvaluationLoader]
        学習と検証の loader
    """
    # 学習は 1 エポック 4 エピソード, 検証は 2 エピソード
    train_loader = RolloutLoader(
        env_class(), model, episodes=4, batch_episodes=2, seed=3,
        exploration=Exploration(epsilon=0.2)
    )
    valid_loader = EvaluationLoader(env_class(), model, episodes=2, seed=100)
    loaders = (train_loader, valid_loader)

    return loaders


def carry_model() -> nn.Module:
    """
    CarryEnv の方策のモデルを作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    model: nn.Module
        線形層
    """
    # 種を固定して作成
    torch.manual_seed(0)
    model = nn.Linear(2, 2)

    return model


def train_run(
    path: Path, epochs: int, make: Callable[[nn.Module], tuple],
    model_fn: Callable[[], nn.Module], resume: Optional[Path] = None
) -> tuple[History, Path]:
    """
    強化学習を決まった種で学習する関数

    Parameters
    ----------
    path: Path
        実験のディレクトリ
    epochs: int
        エポック数
    make: Callable[[nn.Module], tuple]
        学習と検証の loader を作る関数
    model_fn: Callable[[], nn.Module]
        モデルを作る関数
    resume: Optional[Path] = None
        再開元の last.pt, None の場合は最初から

    Returns
    ----------
    history: History
        エポックごとの記録
    run_dir: Path
        実験のディレクトリ
    """
    # 監視は検証の報酬, 再開元があれば resume_from を指定
    text = (
        "optimizer:\n  name: Adam\n  params:\n    lr: 0.05\n"
        f"trainer:\n  epochs: {epochs}\n  monitor: valid/reward\n"
        "  mode: max\n"
    )
    if resume is not None:
        text += f"  resume_from: {resume}\n"
    experiment = make_experiment(path, text)
    model = model_fn()
    train_loader, valid_loader = make(model)
    trainer = Trainer(model, experiment, ReinforceStep(entropy_coef=0.01))
    history = trainer.fit(train_loader, valid_loader)
    run_dir = experiment.run_dir

    return history, run_dir


def assert_same(resumed: History, full: History) -> None:
    """
    再開した学習の記録が通しの学習と一致することを確かめる関数

    Parameters
    ----------
    resumed: History
        再開した学習の記録
    full: History
        通しの学習の記録

    Returns
    ----------
    None
    """
    # 全ての指標が同じエポックで一致
    assert resumed.get("epoch") == full.get("epoch")
    for name in METRICS:
        assert resumed.get(name) == pytest.approx(full.get(name)), name


def test_resume_matches_full_with_env_state(tmp_path: Path) -> None:
    """
    環境の乱数と state_dict の状態が戻り, 再開した学習が通しの学習と
    一致することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    full, _ = train_run(tmp_path / "full", 4, carry_loaders, carry_model)
    _, first_dir = train_run(
        tmp_path / "first", 2, carry_loaders, carry_model
    )
    resumed, _ = train_run(
        tmp_path / "second", 4, carry_loaders, carry_model,
        first_dir / "last.pt"
    )
    assert_same(resumed, full)

    # チェックポイントに学習と検証の loader の状態 (検証は環境の状態だけ)
    state = torch.load(first_dir / "last.pt", weights_only=True)
    assert set(state["loaders"]) == {"train", "valid"}
    assert state["loaders"]["train"]["seeded"] is True
    assert state["loaders"]["train"]["env"] == [{
        "depth": 0, "name": "CarryEnv", "kind": "state_dict",
        "state": {"episodes": 8}
    }]
    assert state["loaders"]["valid"]["env"][0]["state"] == {"episodes": 4}


def test_resume_without_env_state_differs(tmp_path: Path) -> None:
    """
    state_dict を持たない環境では, エピソードを越えて続く状態が戻らない
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    def make(model: nn.Module) -> tuple:
        """
        state_dict を持たない環境の loader を作る関数

        Parameters
        ----------
        model: nn.Module
            方策のモデル

        Returns
        ----------
        loaders: tuple
            学習と検証の loader
        """
        # 環境のクラスだけを変更
        loaders = carry_loaders(model, PlainEnv)

        return loaders

    full, _ = train_run(tmp_path / "full", 4, make, carry_model)
    _, first_dir = train_run(tmp_path / "first", 2, make, carry_model)
    resumed, _ = train_run(
        tmp_path / "second", 4, make, carry_model, first_dir / "last.pt"
    )

    # 乱数は戻るので長さは一致し, エピソードの数の観測で行動がずれる
    assert resumed.get("train/length") == pytest.approx(
        full.get("train/length")
    )
    assert resumed.get("train/loss") != pytest.approx(full.get("train/loss"))


def cartpole_model() -> nn.Module:
    """
    CartPole の方策のモデルを作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    model: nn.Module
        2 層の全結合
    """
    # 種を固定して作成
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2))

    return model


def cartpole_loaders(model: nn.Module) -> tuple:
    """
    CartPole の学習と検証の loader を作る関数

    Parameters
    ----------
    model: nn.Module
        方策のモデル

    Returns
    ----------
    loaders: tuple
        学習と検証の loader
    """
    # 歩数で区切る学習 (エピソードの終わりまで) と, 2 エピソードの検証
    train_loader = RolloutLoader(
        make_env("CartPole-v1"), model, steps=60, batch_episodes=2, seed=0
    )
    valid_loader = EvaluationLoader(
        make_env("CartPole-v1"), model, episodes=2, seed=500
    )
    loaders = (train_loader, valid_loader)

    return loaders


def test_cartpole_resume_matches_full(tmp_path: Path) -> None:
    """
    Gymnasium の環境 (np_random を持つ) で, 再開した学習が通しの学習と
    一致することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    pytest.importorskip("gymnasium")
    full, _ = train_run(
        tmp_path / "full", 4, cartpole_loaders, cartpole_model
    )
    _, first_dir = train_run(
        tmp_path / "first", 2, cartpole_loaders, cartpole_model
    )
    resumed, _ = train_run(
        tmp_path / "second", 4, cartpole_loaders, cartpole_model,
        first_dir / "last.pt"
    )
    assert_same(resumed, full)


def test_resume_old_checkpoint_warns(tmp_path: Path) -> None:
    """
    loader の状態の無いチェックポイントからの再開で警告することを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # loader の状態を消した古い形のチェックポイント
    _, first_dir = train_run(
        tmp_path / "first", 1, carry_loaders, carry_model
    )
    path = first_dir / "last.pt"
    state = torch.load(path, weights_only=True)
    del state["loaders"]
    torch.save(state, path)

    with pytest.warns(UserWarning, match="train の loader の状態"):
        train_run(tmp_path / "second", 2, carry_loaders, carry_model, path)


def test_loader_state_round_trip() -> None:
    """
    保存した状態に戻すと, 同じエピソードと探索の設定で続くことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 1 エポック集めた後の状態を保存
    model = carry_model()
    loader, _ = carry_loaders(model)
    next(iter(loader))
    saved = loader.checkpoint_state()
    torch_state = torch.get_rng_state()
    expected = next(iter(loader))

    # 新しい loader に戻すと同じバッチ, 探索の設定の違いは警告して再開元
    other, _ = carry_loaders(model)
    other.exploration = Exploration(temperature=2.0)
    messages = other.load_checkpoint_state(saved)
    assert len(messages) == 1 and "探索の設定は再開元" in messages[0]
    torch.set_rng_state(torch_state)
    batch = next(iter(other))
    assert other.exploration.epsilon == 0.2
    for key, value in expected.items():
        assert torch.equal(batch[key], value), key


def test_mt19937_rng_and_bad_env_state() -> None:
    """
    配列を含む乱数の状態を保存して戻せ, 保存できない環境の状態はエラーに
    なることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # MT19937 の状態 (配列を含む) を保存して, 別の種類の乱数の環境に復元
    model = carry_model()
    loader, _ = carry_loaders(model)
    next(iter(loader))
    loader.env.np_random = np.random.Generator(np.random.MT19937(1))
    saved = loader.checkpoint_state()
    other, _ = carry_loaders(model)
    other.load_checkpoint_state(saved)
    assert isinstance(other.env.np_random.bit_generator, np.random.MT19937)
    assert other.env.np_random.integers(1000) == (
        loader.env.np_random.integers(1000)
    )

    # 保存できない型を返す環境はエラー
    def bad_state() -> dict:
        """
        保存できない型を含む状態を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        state: dict
            object を含む辞書
        """
        # 保存できない値
        state = {"value": object()}

        return state

    loader.env.state_dict = bad_state
    with pytest.raises(TypeError, match="環境の状態 env\\[0\\]"):
        loader.checkpoint_state()


def test_discard_epoch_restores_env(tmp_path: Path) -> None:
    """
    エポックの途中で止めて fit を呼び直すと, 環境の状態も戻って
    通しの学習と一致することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    full, _ = train_run(tmp_path / "full", 4, carry_loaders, carry_model)

    # 3 エポック目の途中で割り込み, 呼び直して残りを学習
    experiment = make_experiment(
        tmp_path / "stopped",
        "optimizer:\n  name: Adam\n  params:\n    lr: 0.05\n"
        "trainer:\n  epochs: 4\n  monitor: valid/reward\n  mode: max\n"
    )
    model = carry_model()
    train_loader, valid_loader = carry_loaders(model)
    step = ReinforceStep(entropy_coef=0.01)
    calls = {"train": 0}

    def interrupting(model: nn.Module, batch: Any, phase: str) -> dict:
        """
        6 回目の学習のバッチで割り込む step の関数

        Parameters
        ----------
        model: nn.Module
            方策のモデル
        batch: Any
            バッチ
        phase: str
            train か valid

        Returns
        ----------
        outputs: dict
            ReinforceStep の戻り値
        """
        # 学習のバッチを数え, 6 回目で割り込み
        if phase == "train":
            calls["train"] += 1
            if calls["train"] == 6:
                raise KeyboardInterrupt
        outputs = step(model, batch, phase)

        return outputs

    trainer = Trainer(model, experiment, interrupting)
    with pytest.raises(KeyboardInterrupt):
        trainer.fit(train_loader, valid_loader)
    trainer.trainer_cfg.epochs = 2
    history = trainer.fit(train_loader, valid_loader)
    assert_same(history, full)


class OwnWrapper:
    def __init__(self):
        # 中の環境 (unwrapped を返さない自分で作った Wrapper) と空間を保存
        self.env = PlainEnv()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        """
        中の環境でエピソードを始める関数

        Parameters
        ----------
        seed: Optional[int] = None
            乱数の種

        Returns
        ----------
        result: tuple[np.ndarray, dict]
            最初の観測と info
        """
        # 中の環境の初期化
        result = self.env.reset(seed=seed)

        return result

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        中の環境を 1 歩進める関数

        Parameters
        ----------
        action: int
            行動

        Returns
        ----------
        result: tuple[np.ndarray, float, bool, bool, dict]
            観測, 報酬, 終了, 打ち切り, info
        """
        # 中の環境の 1 歩
        result = self.env.step(action)

        return result


def test_env_without_rng_warns() -> None:
    """
    乱数も state_dict も見つからない環境では, 戻すときに警告することを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 乱数の状態は保存されず, 戻すときに警告
    model = carry_model()
    loader, _ = carry_loaders(model, OwnWrapper)
    next(iter(loader))
    saved = loader.checkpoint_state()
    assert saved["np_random"] is None and saved["env"] == []
    other, _ = carry_loaders(model, OwnWrapper)
    messages = other.load_checkpoint_state(saved)
    assert len(messages) == 1 and "環境の乱数は戻せません" in messages[0]


def normalized_loaders(model: nn.Module) -> tuple:
    """
    観測と報酬を正規化する Wrapper で包んだ CartPole の loader を作る関数

    Parameters
    ----------
    model: nn.Module
        方策のモデル

    Returns
    ----------
    loaders: tuple
        学習と検証の loader
    """
    # 学習は観測と報酬, 検証は観測を正規化 (統計はエピソードを越えて続く)
    from gymnasium.wrappers import NormalizeObservation, NormalizeReward

    train_env = NormalizeReward(NormalizeObservation(make_env("CartPole-v1")))
    valid_env = NormalizeObservation(make_env("CartPole-v1"))
    train_loader = RolloutLoader(
        train_env, model, steps=60, batch_episodes=2, seed=0
    )
    valid_loader = EvaluationLoader(valid_env, model, episodes=2, seed=500)
    loaders = (train_loader, valid_loader)

    return loaders


def test_normalize_wrappers_resume_matches_full(tmp_path: Path) -> None:
    """
    正規化の Wrapper の統計も戻り, 再開した学習が通しの学習と一致する
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    pytest.importorskip("gymnasium")
    full, _ = train_run(
        tmp_path / "full", 4, normalized_loaders, cartpole_model
    )
    _, first_dir = train_run(
        tmp_path / "first", 2, normalized_loaders, cartpole_model
    )
    resumed, _ = train_run(
        tmp_path / "second", 4, normalized_loaders, cartpole_model,
        first_dir / "last.pt"
    )
    assert_same(resumed, full)

    # 外側から報酬と観測の正規化の統計を保存
    state = torch.load(first_dir / "last.pt", weights_only=True)
    names = [entry["name"] for entry in state["loaders"]["train"]["env"]]
    assert names == ["NormalizeReward", "NormalizeObservation"]


def test_unsaved_and_mismatched_wrappers_warn() -> None:
    """
    戻せない Wrapper と, 再開元と違う Wrapper で包んだ環境では警告の文を
    返すことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    pytest.importorskip("gymnasium")
    from gymnasium.wrappers import (
        NormalizeObservation, RecordEpisodeStatistics
    )

    # 正規化で包んだ環境の状態を保存
    model = cartpole_model()
    env = NormalizeObservation(make_env("CartPole-v1"))
    loader = RolloutLoader(env, model, episodes=1, seed=0)
    next(iter(loader))
    saved = loader.checkpoint_state()

    # 統計の記録で包むと, 戻せない統計と, 層の違いを通知
    other_env = RecordEpisodeStatistics(make_env("CartPole-v1"))
    other = RolloutLoader(other_env, model, episodes=1, seed=0)
    messages = other.load_checkpoint_state(saved)
    assert any("NormalizeObservation の状態を受け取る層" in m for m in messages)
    assert any("RecordEpisodeStatistics の" in m for m in messages)
    assert env_layers(other_env)[0] is other_env
