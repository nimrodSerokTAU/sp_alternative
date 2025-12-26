import logging

from dl_model.config.config import DataConfig, FeatureConfig, TrainConfig, OutputConfig, ShapExplainConfig
from dl_model.pipeline.experiment import RegressionExperiment

logging.basicConfig(level=logging.INFO)

data_cfg = DataConfig(
    features_file="../input/ortho12_distant_features_121125.csv",
    true_score_name="dseq_from_true",
    test_size=0.2,
    deduplicated=False,
    empirical=False,
)

feat_cfg = FeatureConfig(
    mode=1,
    remove_correlated_features=False,
    scaler_type_features="rank",
    scaler_type_labels="rank",
)

train_cfg = TrainConfig(
    epochs=30,
    batch_size=32,
    learning_rate=1e-2,
    neurons=(0, 128, 64, 16),
    dropout_rate=0.2,
    regularizer_name="l2",
    l2=1e-5,

    loss_fn="custom_mse",
    alpha=0.5,
    eps=1e-6,
    top_k=4,
    ranking_weight=0.3,
    margin=0.0,

    batch_generation="custom",   # or "standard"
    repeats=1,
    mixed_portion=0.0,
    per_aligner=False,
)

out_cfg = OutputConfig(
    out_dir="../out",
    run_id="0",
    verbose=True,
    save_model=True,
    save_scaled_csv=True,
    save_predictions_csv=True,
    save_plots=True,
    save_scaler=True,
)

explain_cfg = ShapExplainConfig(
    enabled=True,
    sample_n=500
)

exp = RegressionExperiment(data_cfg, feat_cfg, train_cfg, out_cfg, explain_cfg)
out = exp.run()
print(out["metrics"])
