
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

This repository provides a framework for predicting MSA quality and ranking alternative MSAs through feature
computation and the use of one of two deep learning models.

The prediction pipeline is structured into three stages:
1. [Alternative MSAs generation](#alternative-msas-generation)
2. [Features extraction for the set of alternative MSAs](#features-extraction-for-the-set-of-alternative-msas)
3. [Ranking prediction using a pre-trained deep learning model](#ranking-prediction-using-a-pre-trained-deep-learning-model)

The repository also includes code for training a new deep learning model using custom datasets
as well as analysing the results.


### Alternative MSAs generation

To generate a set of alternative MSAs for a given set of unaligned sequences, you can use popular MSA tools such as
GUIDANCE2, or MUSCLE with different parameters. GUIDANCE2 allows to generate alternative MSAs while running it
with MAFFT or PRANK as the underlying MSA tool.

The steps we followed for generating 1600 alternative MSAs:
- run GUIDANCE2 with MAFFT or PRANK as the MSA tool:
  - Using the webserver at http://guidance.tau.ac.il/ with default settings, or
  - Running locally the Perl version via command line: `perl <guidance directory>/www/Guidance/guidance.pl --seqFile <fasta> --msaProgram [MAFFT|PRANK] --seqType aa --outDir <out_dir> --program GUIDANCE2 --bootstraps 100`, or
  - Running locally the Python version via command line: `guidance_main.py --seqFile <fasta> --msaProgram [MAFFT|PRANK] --seqType aa --outDir <out_dir> --program GUIDANCE2 --bootstraps 100 --proc_num 4`
  - Refer to the [GUIDANCE2 official web-site](http://guidance.tau.ac.il/) for more details.
  - This will generate 400 alternative MSAs for each MSA tool.
- run MUSCLE with the `-diversified` option to generate diversified replicates:
    - Example command line for generating 400 diversified replicates using 8 threads:
  `muscle -align <fasta> -output <out_dir>/diversified_replicate.@.afa -diversified -replicates 400 -threads 8`
    - Refer to the [MUSCLE official web-site](https://www.drive5.com/muscle/) for more details.
- run BaliPhy:
  - Example command line for generating 400 alternative MSAs using 8 threads:
    `./bali-phy <fasta> --iter=4000 -n <code_name>`
  - Refer to the [BaliPhy official web-site](https://www.bali-phy.org/README.xhtml) for more details.



The resulting MSAs from these runs should then be collected into a single directory for feature extraction
and ranking prediction.
An example set of 1600 alternative MSAs for a single set of unaligned sequences (MSA-batch) is provided
in the `example_data/alternative_msas/126014/` directory for testing purposes.

### Features extraction for the set of alternative MSAs

This module provides tools for computing a comprehensive set of features for a directory containing multiple MSA files.

#### Running Feature Extraction

To compute features for a directory of MSAs, use the `multiple_msa_calc_features_and_labels` function from the `multi_msa_service.py`.

Feature computation is organized into seven feature categories, each of which can be calculated independently:

1. Unaligned sequence attributes
2. MSA-level attributes (`BasicStats` class)
3. Sum-of-Pairs (SoP) features (`SopStats` and `WSopStats` classes)
4. Gap-related features (`GapStats` class)
5. Tree-based features (`TreeStats` class)
6. Entropy-based features (`EntropyStats` class)
7. k-mer-based features (`KMerStats` class)

#### Configuration

Feature extraction is controlled via the `Configuration` class, which defines both how features are computed and which features are included.

Key configuration parameters:

- `models_list`
	A list of evolutionary models (`EvoModel`) to use during feature computation.
	Each model specifies:

	 - substitution matrix
	 - gap opening penalty
	 - gap extension penalty

 - `sop_clac_type`
	Selects the algorithm used to compute the Sum-of-Pairs (SoP) score:
	 - an efficient implementation, or
	 - a naive (baseline) implementation
 - `input_files_dir_name`
	Path to the directory containing the MSA files to process.

 - `additional_weights`
	A set of weighting schemes to apply when computing weighted SoP features.

 - `k_values`
	A list of k values used for k-mer feature computation.
 - `stats_output`
	Specifies which feature groups to compute (or all groups at once).



#### Example
The following example demonstrates how to compute the 153 features used in the paper:

```
configuration: Configuration = Configuration(
		models_list=[
			EvoModel(-10, -0.5, 'BLOSUM62'),
			EvoModel(-6, -0.5, 'BLOSUM62'),
            EvoModel(-10, -1, 'BLOSUM62'),
			EvoModel(-6, -1, 'BLOSUM62'),
            EvoModel(-10, -0.2, 'BLOSUM62'),
			EvoModel(-6, -0.2, 'BLOSUM62'),
            EvoModel(-10, -0.5, 'PAM250'),
			EvoModel(-6, -0.5, 'PAM250'),
            EvoModel(-10, -1, 'PAM250'),
			EvoModel(-6, -1, 'PAM250'),
            EvoModel(-10, -0.2, 'PAM250'),
			EvoModel(-6, -0.2, 'PAM250'),
		],
		sop_clac_type=SopCalcTypes.EFFICIENT,
		input_files_dir_name=<ENTER_YOUR_MSA_FOLDER_PATH>,
		additional_weights={
			WeightMethods.HENIKOFF_WG,
			WeightMethods.HENIKOFF_WOG,
			WeightMethods.CLUSTAL_MID_ROOT,
			WeightMethods.CLUSTAL_DIFFERENTIAL_SUM
		},
		k_values=[5, 10, 20],
		stats_output={StatsOutput.ALL}
)

multiple_msa_calc_features_and_labels(configuration)
```
This code generates a collection of CSV files that together contain all requested features.

#### Testing and Examples

Unit tests located in `tests/sp_alt_spec.py` provide concrete examples for running and validating feature extraction.

A unified features.csv file containing 153 features for a single MSA batch is provided in
example_data/features/ for testing and reference purposes.


### Ranking prediction using a pre-trained deep learning model

The paper describes two deep learning models for MSA ranking prediction (MODEL1 and MODEL2).
To generate ranking predictions using the pre-trained model, run the script `predict_pretrained_main.py`
(located in `dl_model/scripts`) with the appropriate command-line arguments (see below).

The example of pre-trained MODEL2 can be found in the `dl_model/input/orthomam_model2/` directory:
- the model file: `regressor_model_0_mode1_dseq_from_true.keras`
- the scaler file: `scaler_0_mode1_dseq_from_true.pkl`
The model was trained on OrthoMaM-based simulated data with 153 features.

The resulting predictions example file is provided in the `example_data/predictions/` directory for testing purposes.
The resulting predictions file `example_data/predictions/predictions_pretrained_126014_mode1_dseq_from_true.csv`
was generated using the features file located in `example_data/features/126014_features.csv`.

#### Example

The following example shows how to use MODEL2, pre-trained on OrthoMaM-based simulated data with the
above-mentioned features file, model file, and scaler file via the command line (activate the environment first,
adjust paths as needed, and run from the repository root):
First create and activate the environment:
```bash
  cd dl_model/
  conda env create -f environment_uni.yml
  conda activate dl_model_env
```

Then run the prediction script:
   ```bash
   cd dl_model/scripts/
   python predict_pretrained_main.py \
   --features-file ../../example_data/features/126014_features.csv \
   --true-score-name dseq_from_true \
   --scaler-type-features rank \
   --scaler-type-labels rank \
   --model-path ../input/orthomam_model2/regressor_model_0_mode1_dseq_from_true.keras \
   --scaler-path ../input/orthomam_model2/scaler_0_mode1_dseq_from_true.pkl \
   --out-dir ../../example_data/predictions/ \
    --run-id 126014
   ```

While running in IDEs like PyCharm you can set the script parameters in the "Run/Debug Configurations" window as follows (assuming the working directory is set to `dl_model/scripts/`):
```
--features-file ../../example_data/features/126014_features.csv  --true-score-name dseq_from_true  --mode 1   --scaler-type-features rank     --scaler-type-labels rank  --scaler-path ../input/orthomam_model2/scaler_0_mode1_dseq_from_true.pkl    --model-path  ../input/orthomam_model2/regressor_model_0_mode1_dseq_from_true.keras    --out-dir ../../example_data/predictions/  --run-id 126014
```

#### Arguments

1. `--features-file <path_to_features_file>`
Path to the input features file (.csv or .parquet format) for which predictions will be generated.
The file must contain 153 feature columns with names matching those used during model training
and described in the paper for the simulated model, or 155 features for the empirical model.
The first two columns in the file must be:
- `code1` dataset identifier
- `code` MSA identifier

2. `--scaler-type-features`
Type of scaling applied to the input features. For MODEL2, this should be set to `rank`.
For MODEL1, this should be set to `standard`.

3. `--scaler-type-labels`
Type of scaling applied to the labels during training. For MODEL2, this should be set to `rank`.
For MODEL1, this should be set to `standard`.

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

### Results analysis
The `results_analyzer` directory includes the code for the results analysis described in the paper,
which can be executed and configured through the `analyzer.py` file.