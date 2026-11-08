from importlib import metadata

from ml_found.config import load_config
from ml_found.experiment import Experiment, start_experiment
from ml_found.training import History, Trainer

# バージョンは pyproject.toml の値を使う
try:
    __version__ = metadata.version("ml-found")
except metadata.PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "Experiment", "History", "Trainer", "load_config", "start_experiment"
]
