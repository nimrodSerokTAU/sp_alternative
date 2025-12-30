from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple


TrueScoreName = Literal["ssp_from_true", "dseq_from_true", "dpos_from_true", "RF_phangorn_norm", "class_label"]
ScalerType = Literal["standard", "rank", "zscore"]

LossName = Literal[
    "mse",
    "custom_mse",
    "mse_with_topk_rank_loss",
    "ranknet_loss",
    "hybrid_mse_ranknet_loss",
    "hybrid_mse_ranknet_dynamic",
    "kendall_loss",
    "listnet_loss",
    "approx_ndcg_loss",
    "hybrid_mse_approx_ndcg_loss",
    "weighted_ranknet_loss",
    "hybrid_weighted_ranknet_loss",
]


@dataclass(frozen=True)
class DataConfig:
    features_file: str
    true_score_name: TrueScoreName
    test_size: float = 0.2
    random_state: int = 42

    # dataset knobs
    deduplicated: bool = False
    empirical: bool = False

    min_rows_per_code_after_dedup: int = 1100
    number_of_msas_threshold_simulated: int = 1600
    number_of_msas_threshold_empirical: int = 1600


@dataclass(frozen=True)
class FeatureConfig:
    mode: int = 1
    remove_correlated_features: bool = False
    corr_threshold: float = 0.90

    scaler_type_features: ScalerType = "standard"
    scaler_type_labels: ScalerType = "standard"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.2
    learning_rate: float = 1e-2

    neurons: Tuple[int, int, int, int] = (0, 128, 64, 16)
    dropout_rate: float = 0.2

    regularizer_name: Literal["l1", "l2", "l1_l2"] = "l2"
    l1: float = 1e-5
    l2: float = 1e-5

    loss_fn: LossName = "mse"

    # loss params
    top_k: int = 4
    mse_weight: float = 1.0
    ranking_weight: float = 0.3
    alpha: float = 0.5
    eps: float = 1e-6
    margin: float = 0.0
    tau: float = 1.0             # for soft_kendall_loss
    topk_weight_decay: float = 0.3  # for weighted_ranknet_loss / hybrid_weighted_ranknet_loss
    beta: float = 0.2            # for hybrid_weighted_ranknet_loss

    batch_generation: Literal["standard", "custom"] = "standard"
    repeats: int = 1
    mixed_portion: float = 0.0
    per_aligner: bool = False


@dataclass(frozen=True)
class ShapExplainConfig:
    enabled: bool = False
    sample_n: int = 500


@dataclass(frozen=True)
class OutputConfig:
    out_dir: str = "../out"
    run_id: str = "0"
    verbose: bool = True

    save_model: bool = True
    save_scaled_csv: bool = True
    save_predictions_csv: bool = True
    save_plots: bool = True
    save_features_csv: bool = True
    save_scaler: bool = True

@dataclass(frozen=True)
class PickBestConfig:
    features_file: str
    true_score_name: TrueScoreName
    prediction_file: str
    error: float = 0.0
    subset: Optional[str] = None
    out_dir: str = "../out"
    num_trials: int = 1

@dataclass(frozen=True)
class OptunaConfig:
    n_trials: int = 50
    n_jobs: int = 7
    study_mode: Literal[
        "maximize_correlation",
        "minimize_val_loss",
        "maximize_correlation_and_capture_best",
        "maximize_val_kendall",
        "maximize_correlation_of_top_k"
    ] = "minimize_val_loss"
    loss_fn: str = "mse"
    seed: int = 42