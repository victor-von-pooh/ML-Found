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

    # 学習の始めに学習率を線形に上げる回数 (interval の単位), null の場合は無し
    # warmup_ratio は学習全体のステップ数に対する割合で, 学習を始めるときに
    # warmup_steps に直す. 学習率は最初の値の warmup_start_factor 倍から上げる
    warmup_steps: Optional[int] = None
    warmup_ratio: Optional[float] = None
    warmup_start_factor: float = 0.01


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

    # last.pt を保存するエポックの間隔, 最後のエポックとエポックの区切りで
    # 止めたエポックは必ず保存
    every_n_epochs: int = 1

    # best.pt にモデルの重みだけを保存するか (軽くなるが, 再開には使えない)
    best_weights_only: bool = False

    # last.pt にベストモデルの重みも入れるか. false にすると軽くなり,
    # 再開時のベストモデルの重みは再開元の best.pt からの復元
    last_with_best: bool = True


@dataclass
class TrainerConfig:
    # エポック数
    epochs: int = 100

    # ベストモデルの判定に使う指標と, 小さい方と大きい方のどちらが良いか
    # null の場合は valid があれば valid/loss, なければ train/loss を使用
    monitor: Optional[str] = None
    mode: str = "min"

    # 監視する指標が最良値と同じ値の場合に比べる指標と, その良し悪しの向き
    # null の場合は同じ値を改善とみなさない
    monitor_tiebreak: Optional[str] = None
    monitor_tiebreak_mode: str = "min"

    # 勾配のノルムの上限, null の場合はクリッピングなし
    grad_clip_norm: Optional[float] = None

    # クリップ前の勾配のノルムを train/grad_norm として記録するかどうか
    log_grad_norm: bool = False

    # 学習後にベストモデルの重みに戻すかどうか, false の場合は最後の重みのまま
    restore_best: bool = True

    # 監視する指標でベストモデルを選ぶかどうか. GAN のように損失で良し悪しを
    # 決められない学習では false にし, best.pt を作らず最後の重みを残す
    select_best: bool = True

    # 検証時にも勾配を計算するかどうか (入力に対する微分を損失に使う場合など)
    valid_grad: bool = False

    # 検証の前に乱数を固定するシード, null の場合は固定なし
    # 検証の後は学習の乱数の状態に復帰
    valid_seed: Optional[int] = None

    # 学習の後に学習用データを推論モードで評価し直し, train_eval/ で記録するか
    # (Dropout などが効いた状態の train/ の指標を補う. 検証と同じ設定で評価)
    eval_train: bool = False

    # true の場合は逆伝播とパラメータの更新を ML-Found ではなく step の中で実行
    manual_optimization: bool = False

    # 学習を再開するチェックポイントのパス, null の場合は最初から学習
    resume_from: Optional[str] = None

    # 再開時に Scheduler の引数が config と違う場合に, どちらを使うか
    # checkpoint は再開元の状態のまま, config は config の引数で作り直して
    # 再開元と同じステップ数まで進めた状態 (再開した後の学習率は, config の
    # 引数で通しで学習した場合と同じ)
    resume_scheduler: str = "checkpoint"

    # 進捗バーを表示するかどうかと, エポックの中のバッチ単位の進捗バーも出すかどうか
    progress_bar: bool = True
    progress_bar_batches: bool = False

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
