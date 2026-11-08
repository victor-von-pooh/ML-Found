from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExperimentConfig:
    # 出力先は <output_dir>/<name>/<日付>/<時刻>/
    name: str = "experiment"
    output_dir: str = "outputs"

    # 乱数シードと, 速度より再現性を優先するかどうか
    seed: int = 42
    deterministic: bool = True

    # 学習に使うデバイス, auto の場合は cuda, mps, cpu の順に選択
    device: str = "auto"

    # ログの書式
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class OptimizerConfig:
    # torch.optim のクラス名と, そのクラスに渡す引数
    name: str = "Adam"
    params: dict[str, Any] = field(default_factory=lambda: {"lr": 1e-3})


@dataclass
class SchedulerConfig:
    # torch.optim.lr_scheduler のクラス名, null の場合は不使用
    name: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)

    # 学習率を更新する間隔, epoch か step
    interval: str = "epoch"


@dataclass
class EarlyStoppingConfig:
    # 監視する指標が patience 回続けて改善しなければ学習の停止
    use: bool = False
    patience: int = 10

    # 改善とみなす最小の変化
    min_delta: float = 0.0


@dataclass
class CheckpointConfig:
    # 監視する指標が最も良いモデルと, 最後のモデルを保存するかどうか
    save_best: bool = True
    save_last: bool = True


@dataclass
class TrainerConfig:
    # エポック数
    epochs: int = 100

    # ベストモデルの判定に使う指標と, 小さい方と大きい方のどちらが良いか
    # null の場合は valid があれば valid/loss, なければ train/loss を使用
    monitor: Optional[str] = None
    mode: str = "min"

    # 勾配のノルムの上限, null の場合はクリッピングなし
    grad_clip_norm: Optional[float] = None

    # 学習を再開するチェックポイントのパス, null の場合は最初から学習
    resume_from: Optional[str] = None

    # 進捗バーを表示するかどうか
    progress_bar: bool = True

    # Early Stopping とチェックポイントの設定
    early_stopping: EarlyStoppingConfig = field(
        default_factory=EarlyStoppingConfig
    )
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass
class Config:
    # ML-Found が使う設定
    # model や data などプロジェクト固有の設定は default の config に自由に追加
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
