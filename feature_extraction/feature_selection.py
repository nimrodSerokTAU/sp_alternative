
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.feature_selection import RFE


features_file = '/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/orthomam_features_240125.csv'

# X = pd.read_csv(features_file)
# features = X.drop(columns=['',''])

df = pd.read_csv(features_file)
# to make sure that all dataset codes are read as strings and not integers
df['code1'] = df['code1'].astype(str)

# add normalized_rf
df["normalized_rf"] = df['rf_from_true'] / (df['taxa_num'] - 1)
df["class_label"] = np.where(df['dpos_dist_from_true'] <= 0.02, 0, 1)
df["class_label2"] = np.where(df['dpos_dist_from_true'] <= 0.01, 0, np.where(df['dpos_dist_from_true'] <= 0.05, 1, 2))
df = df.dropna()
# df['MEAN_RES_PAIR_SCORE'] = df['MEAN_RES_PAIR_SCORE'].astype(float)
# df['MEAN_COL_SCORE'] = df['MEAN_COL_SCORE'].astype(float)
true_score_name = "dpos_dist_from_true"

y = df[true_score_name]
X = df.drop(
        columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty',
                 'class_label', 'class_label2'])

# Assuming X is your feature matrix as a pandas DataFrame
correlation_matrix = X.corr().abs()

# Create a mask to select upper triangle of the correlation matrix
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Get columns to drop based on correlation threshold
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]

print(to_drop)

# Drop correlated columns
X_cleaned = X.drop(columns=to_drop)

print(X_cleaned)

unique_code1 = df['code1'].unique()

# Split the unique 'code1' into training and test sets
train_code1, test_code1 = train_test_split(unique_code1, test_size=0.2)
print(f"the training set is: {train_code1} \n")
print(f"the testing set is: {test_code1} \n")

# Create training and test DataFrames by filtering based on 'code1'
train_df = df[df['code1'].isin(train_code1)]
test_df = df[df['code1'].isin(test_code1)]

X_train = train_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty','class_label', 'class_label2'])
X_test = test_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty','class_label', 'class_label2'])

y_train = train_df[true_score_name]
y_test = test_df[true_score_name]

scaler = joblib.load(
    f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/orthomam_all_w_balify_no_ancestors/DL8_new_features_w_SoP/scaler_0_mode1_msa_distance.pkl')
X_test_scaled = scaler.transform(X_test)
X_test_scaled = X_test_scaled.astype('float64')
X_test_scaled_with_names = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Set train and test Labels
y_test = test_df[true_score_name]
y_test = y_test.astype('float64')

# Initialize model
model = load_model(
        f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/orthomam_all_w_balify_no_ancestors/DL8_new_features_w_SoP/regressor_model_0_mode1_msa_distance.keras')

# Apply RFE for feature selection
rfe = RFE(model, n_features_to_select=30)  # Choose the number of features you want to keep
X_rfe = rfe.fit_transform(X_train, y_train)

# Get selected features
selected_features = X_train.columns[rfe.support_]
print(selected_features)


