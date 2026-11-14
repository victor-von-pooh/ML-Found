from typing import Any, Callable, Iterator, Optional

from torch import nn, optim

from ml_found.training.optim import step_scheduler


class NamedOptimizers:
    """
    config の optimizers に名前付きで書いた複数の Optimizer を, 1 つの
    Optimizer のように扱う入れ物

    Trainer の中の Optimizer と同じ使い方 (param_groups, zero_grad, step,
    state_dict, load_state_dict) ができる. 名前ごとの Optimizer は
    optimizers[名前] で取り出す

    Parameters
    ----------
    optimizers: dict[str, optim.Optimizer]
        名前と Optimizer (config に書いた順)
    """

    def __init__(self, optimizers: dict[str, optim.Optimizer]):
        # 名前と Optimizer を書いた順に保存
        self.optimizers = dict(optimizers)

    @property
    def param_groups(self) -> list[dict]:
        """
        全ての Optimizer のパラメータグループを, 名前の順につないで返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        groups: list[dict]
            パラメータグループ (元の辞書なので, 書き換えると Optimizer に反映)
        """
        groups = [
            group for optimizer in self.optimizers.values()
            for group in optimizer.param_groups
        ]

        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        全ての Optimizer の勾配を初期化する関数

        Parameters
        ----------
        set_to_none: bool = True
            勾配を 0 にする代わりに None にするかどうか

        Returns
        ----------
        None
        """
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Optional[Callable] = None) -> None:
        """
        全ての Optimizer で, 名前の順にパラメータを更新する関数

        Parameters
        ----------
        closure: Optional[Callable] = None
            使えないので常に None (closure が必要な Optimizer は不可)

        Returns
        ----------
        None
        """
        # closure は Optimizer ごとに損失が違うので受け付けない
        if closure is not None:
            raise ValueError("optimizers では closure を使えません")
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self) -> dict:
        """
        名前ごとの Optimizer の状態をまとめる関数

        Parameters
        ----------
        None

        Returns
        ----------
        state: dict
            {"optimizers": {名前: 状態}}
        """
        state = {
            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizers.items()
            }
        }

        return state

    def load_state_dict(self, state: dict) -> None:
        """
        名前ごとの Optimizer の状態を戻す関数

        Parameters
        ----------
        state: dict
            state_dict の戻り値

        Returns
        ----------
        None
        """
        # 名前の揃った状態だけを受け付け, 1 つの Optimizer の状態は不可
        states = check_named_state(state, "optimizers", self.optimizers)
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(states[name])

    def __getitem__(self, name: str) -> optim.Optimizer:
        """
        名前から Optimizer を取り出す関数

        Parameters
        ----------
        name: str
            config の optimizers に書いた名前

        Returns
        ----------
        optimizer: optim.Optimizer
            その名前の Optimizer
        """
        optimizer = self.optimizers[name]

        return optimizer

    def __iter__(self) -> Iterator[str]:
        """
        名前を書いた順に返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        names: Iterator[str]
            Optimizer の名前
        """
        return iter(self.optimizers)

    def items(self) -> list[tuple[str, optim.Optimizer]]:
        """
        名前と Optimizer の組を書いた順に返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        items: list[tuple[str, optim.Optimizer]]
            名前と Optimizer の組
        """
        items = list(self.optimizers.items())

        return items

    def __repr__(self) -> str:
        """
        名前ごとの Optimizer の表示をつないだ文字列を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        text: str
            「名前: Optimizer の表示」を名前の順に並べた文字列
        """
        text = "\n".join(
            f"{name}: {optimizer}"
            for name, optimizer in self.optimizers.items()
        )

        return text


class NamedSchedulers:
    """
    名前付きの Optimizer ごとの Scheduler を, 1 つの Scheduler のように扱う
    入れ物

    Scheduler を付けない Optimizer の名前は None を持つ. 学習率を更新する
    間隔は全ての Scheduler で同じ (config の確認で保証)

    Parameters
    ----------
    schedulers: dict[str, Optional[optim.lr_scheduler.LRScheduler]]
        Optimizer の名前と Scheduler (付けない場合は None)
    """

    def __init__(
        self,
        schedulers: dict[str, Optional[optim.lr_scheduler.LRScheduler]]
    ):
        # 名前と Scheduler を書いた順に保存
        self.schedulers = dict(schedulers)

    def step(self, metric: Optional[float] = None) -> None:
        """
        全ての Scheduler の学習率を 1 回更新する関数

        Parameters
        ----------
        metric: Optional[float] = None
            ReduceLROnPlateau が改善を判定するための指標の値

        Returns
        ----------
        None
        """
        for scheduler in self.schedulers.values():
            step_scheduler(scheduler, metric)

    def state_dict(self) -> dict:
        """
        名前ごとの Scheduler の状態をまとめる関数

        Parameters
        ----------
        None

        Returns
        ----------
        state: dict
            {"schedulers": {名前: 状態 (付けない場合は None)}}
        """
        state = {
            "schedulers": {
                name: None if scheduler is None else scheduler.state_dict()
                for name, scheduler in self.schedulers.items()
            }
        }

        return state

    def load_state_dict(self, state: dict) -> None:
        """
        名前ごとの Scheduler の状態を戻す関数

        Parameters
        ----------
        state: dict
            state_dict の戻り値

        Returns
        ----------
        None
        """
        # 名前の揃った状態だけを受け付け, Scheduler の有無の違いはエラー
        states = check_named_state(state, "schedulers", self.schedulers)
        for name, scheduler in self.schedulers.items():
            if (scheduler is None) != (states[name] is None):
                raise ValueError(
                    f"optimizers.{name} の Scheduler の有無が再開元と違います"
                )
            if scheduler is not None:
                scheduler.load_state_dict(states[name])

    def __getitem__(
        self, name: str
    ) -> Optional[optim.lr_scheduler.LRScheduler]:
        """
        名前から Scheduler を取り出す関数

        Parameters
        ----------
        name: str
            config の optimizers に書いた名前

        Returns
        ----------
        scheduler: Optional[optim.lr_scheduler.LRScheduler]
            その名前の Optimizer の Scheduler, 付けない場合は None
        """
        scheduler = self.schedulers[name]

        return scheduler

    def items(
        self
    ) -> list[tuple[str, Optional[optim.lr_scheduler.LRScheduler]]]:
        """
        名前と Scheduler の組を書いた順に返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        items: list[tuple[str, Optional[LRScheduler]]]
            名前と Scheduler の組
        """
        items = list(self.schedulers.items())

        return items


def check_named_state(state: Any, key: str, expected: dict) -> dict:
    """
    名前付きの状態が, 今の名前と揃っていることを確かめる関数

    Parameters
    ----------
    state: Any
        読み込む状態
    key: str
        optimizers か schedulers
    expected: dict
        今の名前ごとの入れ物

    Returns
    ----------
    states: dict
        名前ごとの状態
    """
    # 1 つの Optimizer の状態 (optimizer の設定で保存したもの) は読めない
    if not isinstance(state, dict) or key not in state:
        raise ValueError(
            "再開元は optimizers を使わない (Optimizer が 1 つの) 学習なので, "
            "optimizers の学習の状態として読み込めません"
        )
    states = state[key]
    if set(states) != set(expected):
        raise ValueError(
            f"再開元と optimizers の名前が違います: 再開元は {sorted(states)}, "
            f"今回は {sorted(expected)}"
        )

    return states


def split_parameters(
    model: nn.Module, modules: dict[str, list[str]]
) -> dict[str, list[nn.Parameter]]:
    """
    モデルの学習するパラメータを, 名前付きの Optimizer ごとに分ける関数

    学習するパラメータ (requires_grad) は, ちょうど 1 つの Optimizer が
    受け持つ必要がある. どれにも入らないものと, 2 つ以上に入るものはエラー

    Parameters
    ----------
    model: nn.Module
        学習するモデル
    modules: dict[str, list[str]]
        Optimizer の名前と, 受け持つ部分 (model の属性の名前, . で区切って
        入れ子も可) のリスト

    Returns
    ----------
    params: dict[str, list[nn.Parameter]]
        Optimizer の名前ごとのパラメータ (学習するものだけ)
    """
    # 名前で部分を探し, その中の学習するパラメータを集める
    owners: dict[int, str] = {}
    params: dict[str, list[nn.Parameter]] = {}
    for name, paths in modules.items():
        params[name] = []
        for path in paths:
            try:
                part = model.get_submodule(path)
            except AttributeError as error:
                raise ValueError(
                    f"optimizers.{name}.modules の {path} がモデルにありません"
                ) from error
            for parameter in part.parameters():
                if not parameter.requires_grad:
                    continue
                owner = owners.get(id(parameter))
                if owner == name:
                    continue
                if owner is not None:
                    raise ValueError(
                        f"optimizers.{owner} と {name} が同じパラメータを"
                        f"受け持っています ({path} の中)"
                    )
                owners[id(parameter)] = name
                params[name].append(parameter)
        if not params[name]:
            raise ValueError(
                f"optimizers.{name} が受け持つ学習するパラメータがありません: "
                f"{paths}"
            )

    # どの Optimizer にも入らない学習するパラメータはエラー
    missing = [
        param_name for param_name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in owners
    ]
    if missing:
        raise ValueError(
            "どの optimizers も受け持たない学習するパラメータがあります: "
            f"{missing[:5]}{' ほか' if len(missing) > 5 else ''}"
        )

    return params
