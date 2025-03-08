import numpy as np
from classes.pick_me import PickMeGame
from classes.pick_me_per_program import PickMeGameProgram
from classes.pick_me_per_program_max import PickMeGameProgramMax
from classes.pick_me_trio import PickMeGameTrio, PickMeAggregator
from classes.pick_me_average import PickMeGameAverage
from scipy import stats
# from classes.pick_me_classification import PickMeGameClassification
from classes.pick_me_mixed import PickMeGameMixed



if __name__ == '__main__':
    # aggregator = PickMeAggregator()
    n = 1
    data_dict = {}
    sop_data_dict = {}
    for i in range(n):
        # pickme = PickMeGameProgram(features_file='./out/balibase_RV10-50_features_080125_w_foldmason_scores.csv',
        #                            prediction_file=f'./out/BaliBase_ALL_10-50/DL8_w_foldmason_features/prediction_DL_{i}_mode1_msa_distance.csv',
        #                            error=0)
        # pickme = PickMeGameProgram(features_file='./out/orthomam_features_251224.csv',
        #                     prediction_file=f'./out/orthomam_all_w_balify_no_ancestors/DL7_new_features_w_SoP/prediction_DL_{i}_mode1_msa_distance.csv',
        #                     error=0)

        # pickme = PickMeGameTrio(features_file='./out/balibase_RV10-50_features_080125_w_foldmason_scores.csv',
        #                            prediction_file=f'./out/BaliBase_ALL_10-50/DL8_w_foldmason_features/prediction_DL_{i}_mode1_msa_distance.csv',
        #                            error=0)
        #

        # pickme = PickMeGameTrio(features_file='./out/orthomam_features_260225_with_NS_300Alt.csv',
        #                         prediction_file=f'./out/prediction_DL_0_mode1_msa_distance.csv',
        #                         error=0, subset = 64)
        pickme = PickMeGameTrio(features_file='./out/orthomam_features_260225_with_NS_300Alt.csv',
                                prediction_file=f'./out/1/prediction_DL_{i}_mode1_msa_distance.csv',
                                error=0, subset = None)

        # pickme = PickMeGameClassification(features_file='./out/orthomam_features_260225_with_NS_300Alt.csv',
        #                         prediction_file=f'./out/orthomam_NEW/RF_classifier_all_features_threshold_0.5_NS300+KP_alt/rf_prediction_0_mode1_class_label.csv',
        #                         error=0, subset=None)

        # pickme = PickMeGameMixed(features_file='./out/orthomam_features_260225_with_NS_300Alt.csv',
        #                                                           prediction_file1=f'./out/combined_classification_DL/prediction_DL_0_mode1_msa_distance.csv', prediction_file2=f'./out/combined_classification_DL/prediction_DL_0_mode1_class_label.csv',
        #                                                           error=0, subset=None)

        pickme.run(i)

        # pickme = PickMeGameAverage(features_file='./out/orthomam_features_260225.csv',
        #                         prediction_file=f'./out/prediction_DL_{i}_mode3_msa_distance.csv',
        #                         error=0)
        #

        # aggregator.add(pickme)

        pickme.summarize()
        pickme.save_to_csv(i)
        pickme.plot_results(i)
        pickme.plot_overall_results(i)

    # aggregator.summarize()
    # aggregator.save_to_csv()
    # aggregator.plot_results()
    # aggregator.plot_overall_results()



    #     pickme.plot_SoP_results(i)
    #     for condition in pickme.accumulated_sop_data:
    #         if condition not in sop_data_dict:
    #             sop_data_dict[condition] = {'sum': 0, 'count': 0}
    #
    #         sop_data_dict[condition]['sum'] += pickme.accumulated_sop_data[condition]['sum']
    #         sop_data_dict[condition]['count'] += pickme.accumulated_sop_data[condition]['count']
    #     for condition in pickme.accumulated_data:
    #         if condition not in data_dict:
    #             data_dict[condition] = {'sum': 0, 'count': 0}
    #
    #         data_dict[condition]['sum'] += pickme.accumulated_data[condition]['sum']
    #         data_dict[condition]['count'] += pickme.accumulated_data[condition]['count']
    # if n > 1:
    #     pickme.accumulated_sop_data = sop_data_dict
    #     pickme.accumulated_data = data_dict
    #     pickme.plot_average_results(n)
