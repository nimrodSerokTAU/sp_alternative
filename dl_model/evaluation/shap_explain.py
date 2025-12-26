from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from dl_model.scripts.predict_with_pretrained import construct_test_set, use_test_from_origin

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

def run_shap_pretrained(out_dir: str, features_file="./out/balibase_RV10-50_features_080125_w_foldmason_scores.csv",
                        predictions_file=f'./out/BaliBase_ALL_10-50/DL8_w_foldmason_features/prediction_DL_0_mode1_msa_distance.csv',
                        mode=1, portion=0.1, run_id: str = "0"):
    X_test, test_df, true_score_name, main_codes_test, file_codes_test = use_test_from_origin(features_file=features_file,
                                                                                              predictions_file=predictions_file,
                                                                                              mode=mode, portion=portion)

    scaler = joblib.load(
        f'/dl_model/out/BaliBase_ALL_10-50/DL8_w_foldmason_features/scaler_{run_id}_mode1_msa_distance.pkl')
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = X_test_scaled.astype('float64')
    X_test_scaled_with_names = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Set train and test Labels
    y_test = test_df[true_score_name]
    y_test = y_test.astype('float64')

    # Check the size of each set
    print(f"Test set size (final): {X_test_scaled.shape}")

    print(np.any(np.isnan(X_test_scaled)))
    print(np.any(np.isinf(X_test_scaled)))
    print(np.any(np.isnan(y_test)))
    print(np.any(np.isinf(y_test)))

    # Load the saved model
    model = load_model(
        f'/Users/kpolonsky/Documents/sp_alternative/dl_model/out/BaliBase_ALL_10-50/DL8_w_foldmason_features/regressor_model_{run_id}_mode1_msa_distance.keras')

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred = np.ravel(y_pred)  # flatten multi-dimensional array into one-dimensional

    # try to explain the features
    explainer = shap.Explainer(model, X_test_scaled_with_names)
    shap_values = explainer(X_test_scaled_with_names)
    joblib.dump(explainer,
                f'/dl_model/out/explainer_0_mode1_msa_distance.pkl')
    joblib.dump(shap_values,
                f'/dl_model/out/shap_values_0_mode1_msa_distance.pkl')

    matplotlib.use('Agg')
    # shap_values = joblib.load(f'/Users/kpolonsky/Documents/sp_alternative/dl_model/out/shap_values_0_mode1_msa_distance.pkl')

    feature_names = [
        a + ": " + str(b) for a, b in zip(X_test.columns, np.abs(shap_values.values).mean(0).round(3))
    ]

    shap.summary_plot(shap_values, X_test_scaled_with_names, max_display=40, feature_names=feature_names)
    # shap.summary_plot(shap_values, X_test_scaled_with_names, max_display=30, feature_names=feature_names)
    plt.savefig('/Users/kpolonsky/Documents/sp_alternative/dl_model/out/summary_plot.png', dpi=300,
                bbox_inches='tight')
    plt.close()

    shap.plots.waterfall(shap_values[0], max_display=40)
    plt.savefig('/Users/kpolonsky/Documents/sp_alternative/dl_model/out/waterfall_0_plot.png', dpi=300,
                bbox_inches='tight')
    plt.close()

    shap.force_plot(shap_values[0], X_test_scaled[0], matplotlib=True, show=False)
    plt.savefig('/Users/kpolonsky/Documents/sp_alternative/dl_model/out/force_0_plot.png')
    plt.close()

    shap.plots.bar(shap_values, max_display=40)
    plt.savefig('/Users/kpolonsky/Documents/sp_alternative/dl_model/out/bar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

