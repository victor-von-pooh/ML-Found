from typing import Any

from ml_found.utils.requirements import require


def make_env(env_id: str, **kwargs: Any) -> Any:
    """
    Gymnasium の環境を作る関数 (Gymnasium が無ければ入れ方を言って止める)

    Parameters
    ----------
    env_id: str
        Gymnasium に登録された環境の名前 (CartPole-v1 など)
    **kwargs: Any
        gymnasium.make に渡す引数 (max_episode_steps など)

    Returns
    ----------
    env: Any
        作った環境
    """
    # 追加の依存の確認の後に読み込み
    require("rl")
    import gymnasium

    env = gymnasium.make(env_id, **kwargs)

    return env
