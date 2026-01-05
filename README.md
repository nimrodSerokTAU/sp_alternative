
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