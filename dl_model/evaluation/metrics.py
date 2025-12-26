from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def per_msa_correlations(groups: pd.Series, y_true: pd.Series, y_pred: np.ndarray, top_k: int) -> dict:
    df = pd.DataFrame({"msa_code": groups.values, "y_true": y_true.values, "y_pred": y_pred})
    per_corr = []
    per_topk_corr = []

    for msa_id, g in df.groupby("msa_code"):
        if g["y_true"].nunique() > 1 and g["y_pred"].nunique() > 1:
            r, _ = pearsonr(g["y_true"], g["y_pred"])
            per_corr.append(r)

        topg = g.nsmallest(top_k, "y_pred")  # smaller predicted = better
        if topg["y_true"].nunique() > 1 and topg["y_pred"].nunique() > 1:
            r2, _ = pearsonr(topg["y_true"], topg["y_pred"])
            per_topk_corr.append(r2)

    return {
        "avg_per_msa_corr": float(np.mean(per_corr)) if per_corr else float("nan"),
        "median_per_msa_corr": float(np.median(per_corr)) if per_corr else float("nan"),
        "avg_per_msa_topk_corr": float(np.mean(per_topk_corr)) if per_topk_corr else float("nan"),
    }


def top50_percentage(groups: pd.Series, file_codes: pd.Series, y_true: pd.Series, y_pred: np.ndarray, top_n: int = 50) -> float:
    df = pd.DataFrame({
        "msa_code": groups.values,
        "file_code": file_codes.values,
        "true_score": y_true.values,
        "predicted_score": y_pred,
    })

    hits = []
    for msa_id, g in df.groupby("msa_code"):
        best_pred_idx = g["predicted_score"].idxmin()
        best_true_score = g.loc[best_pred_idx, "true_score"]

        g_sorted = g.sort_values("true_score", ascending=True).reset_index(drop=True)
        g_sorted["true_rank"] = np.arange(1, len(g_sorted) + 1)
        best_rank = g_sorted.loc[g_sorted["true_score"] == best_true_score, "true_rank"].values[0]

        hits.append(1 if best_rank <= min(top_n, len(g_sorted)) else 0)

    return float(100.0 * np.mean(hits)) if hits else 0.0
