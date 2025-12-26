CODE_COL = "code"
GROUP_COL = "code1"
ALIGNER_COL = "aligner"
TAXA_COL = "taxa_num"

DROP_LABEL_COLS = ["ssp_from_true", "dseq_from_true", "dpos_from_true", "RF_phangorn_norm", "class_label"]
DROP_COLS_EXTENDED = ["ssp_from_true", "dseq_from_true", "dpos_from_true", "RF_phangorn_norm",
                "class_label", "code", "code1", "aligner"]

COLUMNS_TO_CHOOSE_MODE3 = [
    "sp_mismatch_norm_PAM250", "sp_mismatch_norm_BLOSUM62",
    "sp_norm_PAM250_GO_-6_GE_-0.2", "sp_PAM250_GO_-6_GE_-0.2",
    "constant_sites_pct", "sp_match_count", "sp_match_count_norm",
    "sp_mismatch_count", "sp_mismatch_count_norm",
    "msa_length", "taxa_num",
    "entropy_sum", "entropy_mean", "entropy_75_pct",
    "sp_go_norm", "sp_ge_norm",
    "parsimony_sum", "parsimony_mean", "parsimony_25_pct", "parsimony_75_pct", "parsimony_max",
    "k_mer_average_K5", "k_mer_90_pct_K5", "single_char_count",
    "num_cols_no_gaps", "num_cols_2_gaps", "gaps_len_four_plus",
    "gaps_all_except_1_len4plus", "gaps_2seq_len1",
    "num_cols_all_gaps_except1", "av_gap_segment_length", "num_unique_gaps",
    "bl_mean", "bl_max", "bl_25_pct", "n_unique_sites"
]

SOP_SCORE_NAME = 'sp_BLOSUM62_GO_-10_GE_-0.5'
PREDICTED_SCORE_NAME = 'predicted_score'
PREDICTED_CLASS_PROB = 'predicted_class_prob'

CSV_HEADERS = [
    "trial_number",
    "loss",
    "val_loss",
    "corr_coefficient",
    "avg_per_msa_corr",
    "avg_per_msa_topk_corr",
    "top50_percentage",
    "neurons_1",
    "neurons_2",
    "neurons_3",
    "neurons_4",
    "dropout",
    "lr",
    "l1",
    "l2",
    "batch_size",
    "regularizer",
    "top_k",
    "mse_weight",
    "ranking_weight",
    "batch_generation",
    "loss_fn",
    "alpha",
    "eps",
    "margin",
    "tau",
    "topk_weight_decay",
    "beta",
    "scaler_features",
    "scaler_labels",
    # "val_kendall",
    # "val_spearman",
]