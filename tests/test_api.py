import importlib
import subprocess
import sys

import pytest

import ml_found


@pytest.mark.parametrize(
    "module",
    [
        "ml_found", "ml_found.config", "ml_found.experiment",
        "ml_found.training", "ml_found.utils", "ml_found.utils.plotting",
        "ml_found.rl",
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
    ----------
    None
    """
    # import の順番で結果が変わらないように新しいプロセスで import
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
    ----------
    None
    """
    # 公開している名前が全て取得可能
    for name in ml_found.__all__:
        assert hasattr(ml_found, name)

    assert {"Trainer", "load_config", "start_experiment"} <= set(
        ml_found.__all__
    )


# 画面 (ML-Dashboard) が使う確かめと読み込みの名前と, 公開する場所
SCREEN_NAMES = {
    "ml_found.config": ["load_config", "params_sections", "parse_overrides"],
    "ml_found.experiment": [
        "MAX_EVALUATION_BYTES", "check_experiment_name", "check_resume_path",
        "check_trainer_config", "describe_origin", "load_evaluations",
        "load_samples", "load_task", "original_coordinates",
        "read_samples_index"
    ],
    "ml_found.training": [
        "check_optim_config", "check_optim_params", "check_param_values",
        "describe_optimizers", "describe_schedulers",
        "missing_data_size_params", "named_views", "resolve_optimizer_class",
        "resolve_scheduler_class", "try_build_optim", "uses_named_optimizers"
    ],
    "ml_found.rl": ["file_sha256", "read_policy_spec"],
    "ml_found.utils": ["describe_requirements", "environment_packages",
                       "get_device"]
}


@pytest.mark.parametrize("module", sorted(SCREEN_NAMES))
def test_screen_names_are_public(module: str) -> None:
    """
    画面が使う確かめと読み込みの名前が, 公開の __all__ にあることを確かめる関数

    Parameters
    ----------
    module: str
        公開する場所のモジュールの名前

    Returns
    ----------
    None
    """
    # __all__ にあり, 取り出せること
    package = importlib.import_module(module)
    for name in SCREEN_NAMES[module]:
        assert name in package.__all__, name
        assert hasattr(package, name), name

    # 各 __all__ の名前は全て取り出せること
    for name in package.__all__:
        assert hasattr(package, name), name


def test_max_evaluation_bytes_matches_reading() -> None:
    """
    公開した評価の大きさの上限が, load_evaluations の使う値と同じことを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    from ml_found.experiment import MAX_EVALUATION_BYTES, results

    assert MAX_EVALUATION_BYTES == results.MAX_EVALUATION_BYTES
    assert MAX_EVALUATION_BYTES == 20 * 1024 * 1024
