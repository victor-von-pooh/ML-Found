from importlib import metadata

from ml_found.config import load_config, load_project_config
from ml_found.experiment import (
    Experiment, register_rng, start_experiment, unregister_rng
)
from ml_found.training import History, Trainer
from ml_found.utils.requirements import describe_requirements, require

# バージョンは pyproject.toml の値を使用
try:
    __version__ = metadata.version("ml-found")
except metadata.PackageNotFoundError:
    __version__ = "不明"

__all__ = [
    "Experiment", "History", "Trainer", "describe_requirements",
    "load_config", "load_project_config", "register_rng", "require",
    "start_experiment", "unregister_rng"
]
