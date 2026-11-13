from importlib import metadata

from ml_found.config import load_config, load_project_config
from ml_found.experiment import (
    Experiment, register_rng, start_experiment, unregister_rng
)
from ml_found.training import History, Trainer

# バージョンは pyproject.toml の値を使用
try:
    __version__ = metadata.version("ml-found")
except metadata.PackageNotFoundError:
    __version__ = "不明"

__all__ = [
    "Experiment", "History", "Trainer", "load_config", "load_project_config",
    "register_rng", "start_experiment", "unregister_rng"
]
