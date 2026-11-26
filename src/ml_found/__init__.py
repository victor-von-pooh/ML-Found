from importlib import metadata

from ml_found.config import load_config, load_project_config
from ml_found.experiment import (
    Experiment, describe_origin, load_evaluations, load_samples, load_task,
    original_coordinates, register_rng, save_evaluation, save_sample,
    start_experiment, task_from_dataset, unregister_rng, write_task
)
from ml_found.training import (
    FrechetInceptionDistance, History, Trainer, average_precision,
    frechet_distance, precision_at_k, price_elasticity, profit_lift,
    qini_coefficient, qini_curve, uplift_at_k
)
from ml_found.utils.requirements import (
    MissingRequirementError, describe_requirements, environment_packages,
    require
)

# バージョンは pyproject.toml の値を使用
try:
    __version__ = metadata.version("ml-found")
except metadata.PackageNotFoundError:
    __version__ = "不明"

__all__ = [
    "Experiment", "FrechetInceptionDistance", "History",
    "MissingRequirementError", "Trainer", "average_precision",
    "describe_origin", "describe_requirements", "environment_packages",
    "frechet_distance", "load_config", "load_evaluations",
    "load_project_config", "load_samples", "load_task",
    "original_coordinates", "precision_at_k", "price_elasticity",
    "profit_lift", "qini_coefficient", "qini_curve", "register_rng",
    "require", "save_evaluation", "save_sample", "start_experiment",
    "task_from_dataset", "unregister_rng", "uplift_at_k", "write_task"
]
