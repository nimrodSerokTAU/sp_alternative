from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def run_shap_keras(model, X_test: pd.DataFrame, out_dir: str, sample_n: int = 500, run_id: str = "0"):
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


