from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def run_shap_keras(model, X_test: pd.DataFrame, out_dir: str, sample_n: int = 500, run_id: str = "0",
                   test_df: pd.DataFrame | None = None):
    X_subset = X_test.sample(n=min(sample_n, len(X_test)), random_state=42)

    explainer = shap.Explainer(model, X_subset)
    shap_values = explainer(X_subset)

    joblib.dump(explainer, f"{out_dir}/explainer_{run_id}.pkl")
    joblib.dump(shap_values, f"{out_dir}/shap_values_{run_id}.pkl")

    shap.summary_plot(shap_values, X_subset, max_display=40, show=False)
    plt.savefig(f"{out_dir}/shap_summary_{run_id}.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.plots.bar(shap_values, max_display=40, show=False)
    plt.savefig(f"{out_dir}/shap_bar_{run_id}.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.plots.waterfall(shap_values[0], max_display=40, show=False)
    plt.savefig(f"{out_dir}/shap_waterfall_{run_id}.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap_vals_array = shap_values.values  # shape: (n_samples, n_features)
    feature_names = X_subset.columns.tolist()

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_vals_array).mean(axis=0),
        "mean_shap": shap_vals_array.mean(axis=0),
        "std_shap": shap_vals_array.std(axis=0),
        "max_abs_shap": np.abs(shap_vals_array).max(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    shap_df["rank"] = range(1, len(shap_df) + 1)
    shap_df.to_csv(f"{out_dir}/shap_{run_id}.csv", index=False)

    # Raw SHAP matrix: one row per sample, one column per feature, with code/code1 identifiers
    raw_df = pd.DataFrame(shap_vals_array, columns=feature_names, index=X_subset.index)
    if test_df is not None and "code" in test_df.columns and "code1" in test_df.columns:
        ids = test_df[["code", "code1"]].reset_index(drop=True)
        ids_subset = ids.loc[X_subset.index]
        raw_df.insert(0, "code1", ids_subset["code1"].values)
        raw_df.insert(0, "code", ids_subset["code"].values)
    raw_df.to_csv(f"{out_dir}/shap_raw_{run_id}.csv", index=False)


