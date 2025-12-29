from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Tuple
import optuna
import optuna.visualization as vis
import numpy as np

from dl_model.pipeline.experiment import RegressionExperiment
from dl_model.config.config import (
    DataConfig,
    FeatureConfig,
    TrainConfig,
    OutputConfig,
    OptunaConfig)
from dl_model.config.constants import CSV_HEADERS

def ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(",".join(CSV_HEADERS) + "\n")


def append_trial_row(path: Path, row: Dict[str, Any]) -> None:
    values = [row.get(h, "") for h in CSV_HEADERS]
    with path.open("a") as f:
        f.write(",".join(map(str, values)) + "\n")

def suggest_hparams(
    trial: optuna.Trial,
    base_train_cfg: TrainConfig,
    study_mode: str,
    loss_fn: str,
    scaler_type_features: str,
    scaler_type_labels: str,
) -> Tuple[TrainConfig, Dict[str, Any]]:
    """
    Returns:
      - train_cfg for this trial
      - flat dict of hyperparams for logging
    """

    if scaler_type_labels == "standard" and scaler_type_features == "standard":
        batch_generation = "standard"
    else:
        batch_generation = "custom"

    if study_mode != "maximize_correlation_of_top_k":
        neurons = (
            trial.suggest_categorical("neurons_1", [8, 16, 32, 64, 128, 256, 512]),
            trial.suggest_categorical("neurons_2", [8, 16, 32, 64, 128, 256, 512]),
            trial.suggest_categorical("neurons_3", [0, 8, 16, 32, 64, 128, 256, 512]),
            trial.suggest_categorical("neurons_4", [0, 8, 16, 32, 64, 128, 256, 512]),
        )
        dropout_rate = trial.suggest_float("dropout", 0.1, 0.4)
        learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 32, 64, 128, 256])
        regularizer_name = trial.suggest_categorical("regularizer", ["l1", "l2", "l1_l2"])

        if regularizer_name == "l1":
            l1 = trial.suggest_float("l1", 1e-7, 1e-4, log=True)
            l2 = 0.0
        elif regularizer_name == "l2":
            l1 = 0.0
            l2 = trial.suggest_float("l2", 1e-7, 1e-4, log=True)
        else:  # l1_l2
            l1 = trial.suggest_float("l1", 1e-7, 1e-4, log=True)
            l2 = trial.suggest_float("l2", 1e-7, 1e-4, log=True)

        # Loss-specific
        top_k = 0
        mse_weight = 0.0
        ranking_weight = 0.0
        alpha = 0.0
        eps = 0.0
        margin = 0.0
        tau = getattr(base_train_cfg, "tau", 1.0)
        topk_weight_decay = getattr(base_train_cfg, "topk_weight_decay", 0.3)
        beta = getattr(base_train_cfg, "beta", 0.2)

        if loss_fn == "custom_mse":
            top_k = 8
            mse_weight = 1.0
            ranking_weight = 0.5

        elif loss_fn in {"mse", "kendall_loss"}:
            pass

        elif loss_fn in {"hybrid_mse_ranknet_loss", "hybrid_mse_ranknet_dynamic"}:
            alpha = trial.suggest_float("alpha", 0.90, 0.995)
            eps = trial.suggest_float("eps", 1e-6, 1e-3, log=True)

        elif loss_fn == "ranknet_loss":
            eps = trial.suggest_float("eps", 1e-6, 1e-3, log=True)
            margin = trial.suggest_float("margin", 0.0, 0.5)

        elif loss_fn == "approx_ndcg_loss":
            eps = 1e-10

        elif loss_fn == "hybrid_mse_approx_ndcg_loss":
            alpha = trial.suggest_float("alpha", 0.1, 0.9)
            eps = 1e-10

        elif loss_fn == "mse_with_topk_rank_loss":
            top_k = trial.suggest_int("top_k", 2, 10)
            ranking_weight = trial.suggest_float("ranking_weight", 0.1, 10.0)
            margin = trial.suggest_float("margin", 0.0, 0.5)

        elif loss_fn == "weighted_ranknet_loss":
            topk_weight_decay = trial.suggest_float("topk_weight_decay", 0.05, 1.0)
            eps = trial.suggest_float("eps", 1e-6, 1e-3, log=True)

        elif loss_fn == "hybrid_weighted_ranknet_loss":
            topk_weight_decay = trial.suggest_float("topk_weight_decay", 0.05, 1.0)
            alpha = trial.suggest_float("alpha", 0.1, 1.0)
            beta = trial.suggest_float("beta", 0.0, 1.0)
            margin = trial.suggest_float("margin", 0.0, 0.5)
            eps = trial.suggest_float("eps", 1e-6, 1e-3, log=True)

    else:
        neurons = (256, 512, 128, 256)
        dropout_rate = 0.205948598666589
        learning_rate = 0.0000362102118239549
        batch_size = 8
        regularizer_name = "l2"
        l1 = 0.0
        l2 = 1.25386213021539e-07
        batch_generation = "custom"

        top_k = trial.suggest_int("top_k", 2, 10)
        mse_weight = 1.0
        ranking_weight = trial.suggest_float("ranking_weight", 0.1, 10.0)

        alpha = 0.0
        eps = 0.0
        margin = 0.0
        tau = getattr(base_train_cfg, "tau", 1.0)
        topk_weight_decay = getattr(base_train_cfg, "topk_weight_decay", 0.3)
        beta = getattr(base_train_cfg, "beta", 0.2)

    train_cfg = replace(
        base_train_cfg,
        learning_rate=learning_rate,
        batch_size=batch_size,
        neurons=neurons,
        dropout_rate=dropout_rate,
        regularizer_name=regularizer_name,
        l1=l1,
        l2=l2,
        loss_fn=loss_fn,
        top_k=top_k,
        mse_weight=mse_weight,
        ranking_weight=ranking_weight,
        alpha=alpha,
        eps=eps,
        margin=margin,
        tau=tau,
        topk_weight_decay=topk_weight_decay,
        beta=beta,
        batch_generation=batch_generation,
    )

    flat = {
        "neurons_1": neurons[0],
        "neurons_2": neurons[1],
        "neurons_3": neurons[2],
        "neurons_4": neurons[3],
        "dropout": dropout_rate,
        "lr": learning_rate,
        "batch_size": batch_size,
        "regularizer": regularizer_name,
        "l1": l1,
        "l2": l2,
        "loss_fn": loss_fn,
        "top_k": top_k,
        "mse_weight": mse_weight,
        "ranking_weight": ranking_weight,
        "alpha": alpha,
        "eps": eps,
        "margin": margin,
        "tau": tau,
        "topk_weight_decay": topk_weight_decay,
        "beta": beta,
        "batch_generation": batch_generation,
        "scaler_features": scaler_type_features,
        "scaler_labels": scaler_type_labels,
    }
    return train_cfg, flat

def build_objective(
    *,
    base_data_cfg: DataConfig,
    base_feat_cfg: FeatureConfig,
    base_train_cfg: TrainConfig,
    out_cfg: OutputConfig,
    opt_cfg: OptunaConfig,
    log_path: Path,
) :
    ensure_csv(log_path)

    def objective(trial: optuna.Trial) -> float:
        train_cfg, flat = suggest_hparams(
            trial=trial,
            base_train_cfg=base_train_cfg,
            study_mode=opt_cfg.study_mode,
            loss_fn=opt_cfg.loss_fn,
            scaler_type_features=base_feat_cfg.scaler_type_features,
            scaler_type_labels=base_feat_cfg.scaler_type_labels,
        )

        trial_out_cfg = replace(out_cfg, run_id=f"trial_{trial.number}")

        try:
            exp = RegressionExperiment(
                data_cfg=base_data_cfg,
                feat_cfg=base_feat_cfg,
                train_cfg=train_cfg,
                out_cfg=trial_out_cfg,
            )
            run_output = exp.run()
            results = run_output["results"]
            metrics = run_output["metrics"]

            loss = float(results.get("test_loss", np.nan))
            val_loss = float(results.get("val_loss", np.nan))
            corr = float(results.get("pearson_r", np.nan))
            avg_per_msa_corr = float(metrics.get("avg_per_msa_corr", np.nan))
            avg_per_msa_topk_corr = float(metrics.get("avg_per_msa_topk_corr", np.nan))
            top50 = float(metrics.get("top50_percentage", np.nan))
            # val_kendall = results.get("val_kendall", "")
            # val_spearman = results.get("val_spearman", "")

            if not np.isfinite(corr):
                raise ValueError("Non-finite corr encountered")

            append_trial_row(log_path, {
                "trial_number": trial.number,
                "loss": loss,
                "val_loss": val_loss,
                "corr_coefficient": corr,
                "avg_per_msa_corr": avg_per_msa_corr,
                "avg_per_msa_topk_corr": avg_per_msa_topk_corr,
                "top50_percentage": top50,
                # "val_kendall": val_kendall,
                # "val_spearman": val_spearman,
                **flat
            })

            if opt_cfg.study_mode == "maximize_correlation":
                return avg_per_msa_corr
            if opt_cfg.study_mode == "minimize_val_loss":
                return val_loss
            if opt_cfg.study_mode == "maximize_correlation_and_capture_best":
                return 0.5 * avg_per_msa_corr + (top50 / 100.0)
            # if opt_cfg.study_mode == "maximize_val_kendall":
                # if val_kendall is None, penalize
                # return float(val_kendall) if val_kendall not in (None, "") else -1e9
            if opt_cfg.study_mode == "maximize_correlation_of_top_k":
                return avg_per_msa_topk_corr

            raise ValueError(f"Unknown study_mode: {opt_cfg.study_mode}")

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            raise optuna.exceptions.TrialPruned()

    return objective


def main() -> None:
    data_cfg = DataConfig(
        features_file="../out/ortho12_distant_features_121125.csv",
        true_score_name="dseq_from_true",
        test_size=0.2,
        empirical=False,
    )

    feat_cfg = FeatureConfig(
        mode=1,
        remove_correlated_features=False,
        scaler_type_features="rank",
        scaler_type_labels="rank"
    )

    train_cfg = TrainConfig( # overwritten per trial
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        learning_rate=0.01,
        neurons=(0, 128, 64, 16),
        dropout_rate=0.2,
        regularizer_name="l2",
        l1=1e-5,
        l2=1e-5,
        loss_fn="mse",
        top_k=4,
        mse_weight=1.0,
        ranking_weight=0.3,
        alpha=0.5,
        eps=1e-6,
        margin=0.0,
        tau=1.0,
        topk_weight_decay=0.3,
        beta=0.2,
        batch_generation="custom",
        repeats=1,
        mixed_portion=0.0,
        per_aligner=False,
    )

    out_cfg = OutputConfig(
        out_dir="../out",
        run_id="optuna",
        save_model=False,
        save_scaled_csv=False,
        save_predictions_csv=False,
        save_plots=False,
        save_features_scv=False,
        save_scaler=False,
    )

    opt_cfg = OptunaConfig(
        n_trials=3,
        n_jobs=7,
        study_mode="minimize_val_loss",
        loss_fn="custom_mse",
        seed=42,
    )

    log_path = Path(
        f"{out_cfg.out_dir}/optuna_results_{opt_cfg.study_mode}_{opt_cfg.loss_fn}"
        f"_featSc_{feat_cfg.scaler_type_features}_labSc_{feat_cfg.scaler_type_labels}"
        f"_empirical_{data_cfg.empirical}.csv"
    )

    direction = "maximize" if opt_cfg.study_mode in {
        "maximize_correlation",
        "maximize_correlation_and_capture_best",
        "maximize_val_kendall",
        "maximize_correlation_of_top_k",
    } else "minimize"

    sampler = optuna.samplers.TPESampler(seed=opt_cfg.seed)

    study = optuna.create_study(sampler=sampler, direction=direction)
    objective = build_objective(
        base_data_cfg=data_cfg,
        base_feat_cfg=feat_cfg,
        base_train_cfg=train_cfg,
        out_cfg=out_cfg,
        opt_cfg=opt_cfg,
        log_path=log_path,
    )

    study.optimize(objective, n_trials=opt_cfg.n_trials, n_jobs=opt_cfg.n_jobs)

    print("Best Trial:")
    print(study.best_trial.params)

    df = study.trials_dataframe()
    df.to_csv(f"{out_cfg.out_dir}/optuna_study_all_trials_{opt_cfg.study_mode}_{opt_cfg.loss_fn}.csv", index=False)

    vis.plot_optimization_history(study).show()
    vis.plot_param_importances(study).show()
    vis.plot_slice(study).show()
    vis.plot_parallel_coordinate(study).show()
    vis.plot_contour(study).show()
    vis.plot_edf(study).show()


if __name__ == "__main__":
    main()
