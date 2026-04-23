from dl_model.evaluation.pick_best import PickBest
from dl_model.config.config import PickBestConfig

pickbest_cfg = PickBestConfig(
    features_file="/Users/kpolonsky/Documents/sp_alternative/dl_model/out/nucleotide_features_200426.csv", #replace with your features file
    # features_file="/Users/kpolonsky/Documents/sp_alternative/dl_model/out/large_trees_5000S1_0.5_features.csv",
    true_score_name="dseq_from_true",
    prediction_file=f"/Users/kpolonsky/Documents/sp_alternative/dl_model/out/prediction_DL_0_mode1_dseq_from_true.csv", #replace with your prediction file
    # prediction_file=f"/Users/kpolonsky/Documents/sp_alternative/dl_model/out/new_model2_3_340/pretrained_5000/prediction_pretrained_5000_0.5_mode1_dseq_from_true.csv", #replace with your prediction file
    error=0.0,
    subset=None,
    out_dir="/Users/kpolonsky/Documents/sp_alternative/dl_model/out/",
    num_trials=1
)

if __name__ == '__main__':
    data_dict = {}
    sop_data_dict = {}
    for i in range(pickbest_cfg.num_trials):
        pickme = PickBest(features_file=pickbest_cfg.features_file,
                            prediction_file=pickbest_cfg.prediction_file,
                            true_score_name=pickbest_cfg.true_score_name,
                            # sum_of_pairs_score='sp_Nucleotides_GO_-5_GE_-1', #TODO - remove for AA
                            sum_of_pairs_score="sp_NucleotidesPAM250_GO_-1.5_GE_0",
                            error=pickbest_cfg.error,
                            subset=pickbest_cfg.subset,
                            output_dir=pickbest_cfg.out_dir)
        pickme.run(i)
        pickme.summarize()
        pickme.save_to_csv(i)
        pickme.plot_results(i)
        pickme.plot_overall_results(i)

