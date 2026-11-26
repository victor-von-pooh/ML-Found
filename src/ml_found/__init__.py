from importlib import metadata

from ml_found.config import load_config, load_project_config
from ml_found.experiment import (
    Experiment, describe_origin, load_evaluations, load_samples, load_task,
    original_coordinates, register_rng, save_evaluation, save_sample,
    start_experiment, task_from_dataset, unregister_rng, write_task
)
from ml_found.training import (
    FrechetInceptionDistance, History, PinballLoss, Trainer,
    average_precision, frechet_distance, ndcg_at_k, order_cost, pinball_loss,
    precision_at_k, price_elasticity, profit_lift, qini_coefficient,
    qini_curve, recall_at_k, uplift_at_k
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
    "MissingRequirementError", "PinballLoss", "Trainer", "average_precision",
    "describe_origin", "describe_requirements", "environment_packages",
    "frechet_distance", "load_config", "load_evaluations",
    "load_project_config", "load_samples", "load_task", "ndcg_at_k",
    "order_cost", "original_coordinates", "pinball_loss", "precision_at_k",
    "price_elasticity", "profit_lift", "qini_coefficient", "qini_curve",
    "recall_at_k", "register_rng", "require", "save_evaluation",
    "save_sample", "start_experiment", "task_from_dataset", "unregister_rng",
    "uplift_at_k", "write_task"
]
