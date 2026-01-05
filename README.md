
# A deep-learning-based score to evaluate multiple sequence alignments

This repository enables predicting MSA quality by computing features and using one of two deep learning models.

### Features calculation
In order to calculate the features for a directory of MSAs, use the `multiple_msa_calc_features_and_labels` function from the `multi_msa_service.py`

The unit-tests on `tests/sp_alt_spec.py` provide examples for running and testing the features creation.
