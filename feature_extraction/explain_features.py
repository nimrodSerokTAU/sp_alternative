import numpy as np
from classes.regressor import Regressor
from scipy import stats
import joblib
import shap
import pandas as pd
from tensorflow.keras.models import load_model
from feature_extraction.predict_with_pretrained import construct_test_set

if __name__ == '__main__':
    n = 1
    for i in range(n):
        X_test, test_df, true_score_name, main_codes_test, file_codes_test = construct_test_set(
            "/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/balibase_features_73.csv", mode=1)
        scaler = joblib.load(
            f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/BaliBase/DL4_w_SoP_and_importance/scaler_{i}_mode1_msa_distance.pkl')
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
            f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/BaliBase/DL4_w_SoP_and_importance/regressor_model_{i}_mode1_msa_distance.keras')

        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred = np.ravel(y_pred)  # flatten multi-dimensional array into one-dimensional

        # try to explain the features
        explainer = shap.Explainer(model, X_test_scaled_with_names)
        shap_values = explainer(X_test_scaled_with_names)
        joblib.dump(explainer,
                    f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/explainer_0_mode1_msa_distance.pkl')
        joblib.dump(shap_values,
                    f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/shap_values_0_mode1_msa_distance.pkl')
