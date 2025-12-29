from dl_model.evaluation.pick_best import PickBest
from dl_model.config.config import PickBestConfig

pickbest_cfg = PickBestConfig(
    features_file="../input/ortho12_distant_features_121125.csv",
    true_score_name="dseq_from_true",
    prediction_file=f"../out/prediction_DL_0_mode1_dseq_from_true.csv",
    error=0.0,
    subset=None,
    out_dir="../out",
    num_trials=1
)

if __name__ == '__main__':
    data_dict = {}
    sop_data_dict = {}
    for i in range(pickbest_cfg.num_trials):
        pickme = PickBest(features_file=pickbest_cfg.features_file,
                            prediction_file=pickbest_cfg.prediction_file,
                            true_score_name=pickbest_cfg.true_score_name,
                            error=pickbest_cfg.error,
                            subset=pickbest_cfg.subset,
                            output_dir=pickbest_cfg.out_dir)
        pickme.run(i)
        pickme.summarize()
        pickme.save_to_csv(i)
        pickme.plot_results(i)
        pickme.plot_overall_results(i)

