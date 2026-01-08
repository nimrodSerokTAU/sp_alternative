
# A deep-learning-based score to evaluate multiple sequence alignments

Nimrod Serok(1*), Ksenia Polonsky(1*), Haim Ashkenazy(2), Itay Mayrose(3), Jeffrey L. Thorne(4,5), Tal Pupko(1,†)

1 The Shmunis School of Biomedicine and Cancer Research, George S. Wise Faculty of Life Sciences, Tel Aviv University, Tel Aviv 69978, Israel.

2 Department of Molecular Biology, Max Planck Institute for Biology Tübingen, Tübingen, Germany.

3 The School of Plant Sciences and Food Security, George S. Wise Faculty of Life Sciences, Tel Aviv University, Tel Aviv 69978, Israel

4 Department of Biological Sciences, North Carolina State University, Raleigh, NC 27695, USA.

5 Department of Statistics, North Carolina State University, Raleigh, NC 27695, USA.

† To whom correspondence should be addressed. Email: talp@tauex.tau.ac.il

* These two authors equally contributed to this work

## Description

This repository provides a framework for predicting MSA quality through feature computation and the use of one of two deep learning models. 
It is structured into three stages: feature computation, model training, and results analysis.
### Features calculation
In order to calculate the features for a directory of MSAs, use the `multiple_msa_calc_features_and_labels` function from the `multi_msa_service.py`
The feature calculation is divided into 7 categories that can be calculated separately: 
1. Unaligned sequence attributes
2. MSA attributes (`BasicStats` class)
3. SoP-related features (`SopStats` class and `WSopStats` class)
4. Gap related features (`GapStats` class)
5. Tree related features (`TreeStats` class)
6. Entropy related features (`EntropyStats` class)
7. kmer related features (`KMerStats` class)

The `Configuration` class control the configuration of the feature calculation, as well as which features to calculate.

The unit-tests on `tests/sp_alt_spec.py` provide examples for running and testing the features' creation.

### Results analysis
The `results_analyzer` directory includes the code for the results analysis described in the paper, which can be executed and configured through the `analyzer.py` file.

### Ranking prediction with the pre-trained model (MODEL2)
To generate ranking predictions using the pre-trained model, run the script  `predict_pretrained_main.py` 
(located in `dl_model/scripts`) with the appropriate command-line arguments.

#### Example

The following example shows how to use MODEL2, pre-trained on OrthoMaM-based simulated data:
   ```bash
   python predict_pretrained_main.py \
   --features-file <path_to_features_file> \
   --scaler-type-features rank \
   --scaler-type-labels rank \
   --model-path ./dl_model/input/orthomam_model2/regressor_model_0_mode1_dseq_from_true.keras \
   --scaler-path ./dl_model/input/orthomam_model2/scaler_0_mode1_dseq_from_true.pkl \
   --out-dir <output_directory>
   ```

#### Arguments

1. `--features-file <path_to_features_file>`
Path to the input features file (.csv format) for which predictions will be generated.
The file must contain 153 feature columns with names matching those used during model training 
and described in the paper for the simulated model. 
The first two columns must be:
- `code1` dataset identifier
- `code` MSA identifier

2. `--scaler-type-features` 
Type of scaling applied to the input features. For MODEL2, this should be set to `rank`

3. `--scaler-type-labels` 
Type of scaling applied to the labels during training. For MODEL2, this should be set to `rank`

4. `--model-path` 
Path to the pre-trained deep learning model file (`.keras` format).

5. `--scaler-path` 
Path to the scaler used for feature normalization (`.pkl` format).

6. `--out-dir <output_directory>`
Directory where the prediction results will be saved.

### Training a new model 

This section describes how to train a new deep learning model using the provided training pipeline. 
The `run_experiment_main.py` script located in the `dl_model/scripts` orchestrates data loading, feature preprocessing, model configuration, 
training, evaluation, and optional explainability in a single experiment run. 
Training behavior is fully controlled through configuration objects, allowing you to easily reproduce experiments 
or adjust hyperparameters without modifying the core code.
The example below shows how to define the required configuration blocks for data handling, 
feature processing, model training, output management, and SHAP-based model explanation.
To train MODEL1, 
set `loss_fn="mse"` and `batch_generation="standard"` in `TrainConfig`, 
and use the appropriate features file for `data_cfg` and hyperparameters (as described in the paper).
To train MODEL2, 
set `loss_fn="custom_mse"` and `batch_generation="custom"` in `TrainConfig`, 
and use the appropriate features file for `data_cfg` and hyperparameters (as described in the paper).
Adjust empirical=False/True in `data_cfg` according to the features file used (simulated/empirical).
Finally, call the `run_experiment_main.py` script with the defined configurations to start training.

#### Example

```python
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
    epochs=50,
    batch_size=32,
    learning_rate=0.0022,
    neurons=(64, 128, 64, 512),
    dropout_rate=0.24,
    regularizer_name="l2",
    l2=1.65e-5,
    loss_fn="custom_mse",
    alpha=0,
    eps=0,
    top_k=8,
    ranking_weight=1.33,
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
```

#### Arguments

1. `data_cfg` (DataConfig):
    `features_file`: Path to the input features file (.csv format) for training.
    `true_score_name`: Name of the column in the features file representing the true MSA quality score (label) to predict.
    `empirical`: Whether to use empirical data for training (155 features) or simulated data (153 features).
2. `feat_cfg` (FeatureConfig):
    `mode`: Feature processing mode (1 uses all features, 2 uses short list of features).
    `remove_correlated_features`: Whether to remove highly correlated features.
    `scaler_type_features`: Type of scaling for features ('rank', 'standard', 'zscore').
    `scaler_type_labels`: Type of scaling for labels ('rank', 'standard', 'zscore').
3. `train_cfg` (TrainConfig):
    `loss_fn`: Loss function to use ('mse', 'custom_mse', 'ranknet_loss', 'hybrid_mse_ranknet_loss', 'kendall_loss', etc.).
    `batch_generation`: Method for generating training batches ('standard', 'custom'). Custom uses a specialized batch generation strategy where all samples in the same mini-batch come from the same MSA-batch.
    `regularizer_name`: Type of regularization to apply ('l1', 'l2', 'l1_l2').
4.  `out_cfg` (OutputConfig):
    `out_dir`: Directory to save output files.
    `run_id`: Identifier for the training run.
    `save_model`: Whether to save the trained model (.keras format).
    `save_scaled_csv`: Whether to save the scaled features and labels to a CSV file.
    `save_predictions_csv`: Whether to save the model's predictions to a CSV file.
    `save_plots`: Whether to generate and save training plots.
    `save_scaler`: Whether to save the feature scaler used during training (.pkl format).
5. `explain_cfg` (ShapExplainConfig):
    `enabled`: Whether to perform SHAP-based model explainability analysis.
    `sample_n`: Number of samples to use for SHAP analysis.