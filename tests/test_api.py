import subprocess
import sys

import pytest

import ml_found


@pytest.mark.parametrize(
    "module",
    [
        "ml_found", "ml_found.config", "ml_found.experiment",
        "ml_found.training", "ml_found.utils", "ml_found.utils.plotting",
        "ml_found.training.trainer", "ml_found.experiment.setup"
    ]
)
def test_import_in_fresh_process(module: str) -> None:
    """
    どのモジュールから import しても循環 import にならないことを確かめる関数

    Parameters
    ----------
    module: str
        import するモジュールの名前

    Returns
    -------
    None
    """
    # import の順番で結果が変わらないように新しいプロセスで import する
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_public_api() -> None:
    """
    学習に必要なものがトップレベルから使えることを確かめる関数

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # 公開している名前が全て取得できる
    for name in ml_found.__all__:
        assert hasattr(ml_found, name)

    assert {"Trainer", "load_config", "start_experiment"} <= set(
        ml_found.__all__
    )
