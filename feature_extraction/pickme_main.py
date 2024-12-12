import numpy as np
from classes.pick_me import PickMeGame
from classes.pick_me_per_program import PickMeGameProgram
from scipy import stats

if __name__ == '__main__':
    n = 1
    data_dict = {}
    sop_data_dict = {}
    for i in range(n):
        # pickme = PickMeGameProgram(features_file='./out/balibase_features_full_74.csv',
        #                            prediction_file=f'./out/BaliBase/DL2/prediction_DL_{i}_mode2_msa_distance.csv',
        #                            error=0)
        pickme = PickMeGameProgram(features_file='./out/orthomam_all_w_balify_no_ancestors_67.csv',
                            prediction_file=f'./out/prediction_DL_{i}_mode1_msa_distance.csv',
                            error=0)

        # pickme = PickMeGame(features_file='./out/orthomam_all_w_balify_no_ancestors_67.csv',
        #                     prediction_file=f'./out/orthomam_all_w_balify_no_ancestors/DL1/prediction_DL_{i}_mode2_msa_distance.csv',
        #                     error=0)
        # pickme = PickMeGame(features_file='./out/TreeBASE_incl_missing_prank.csv',
        #                     prediction_file=f'./out/TreeBase_incl_missing_prank/DL/prediction_DL_{i}_mode2_msa_distance.csv',
        #                     error=0)
        # pickme = PickMeGame(features_file='./out/orthomam_treebase_w_balify_no_ancestors_67.csv',
        #                     prediction_file=f'./out/orthomam_treebase_w_balify_no_ancestors_67/DL/prediction_DL_{i}_mode2_msa_distance.csv',
        #                     error=0)
        # pickme = PickMeGame(features_file='./out/orthomam_features_w_balify.csv',
        #                     prediction_file=f'./out/orthomam_w_BaliPhy/DL/prediction_DL_{i}_mode2_msa_distance.csv',
        #                     error=0)
        # pickme = PickMeGame(features_file='./out/TreeBASE_incl_missing_prank.csv',
        #                     prediction_file=f'./out/TreeBase_incl_missing_prank/RF/rf_prediction_{i}_mode2_msa_distance.csv',
        #                     error=0)
        # pickme = PickMeGame(features_file='/Users/kpolonsky/Downloads/balibase_features.csv',
        #                     prediction_file=f'./out/Balibase_trained/RF1/rf_prediction_{i}_mode2_msa_distance.csv',
        #                     error=0)
        # pickme = PickMeGame(features_file='/Users/kpolonsky/Downloads/balibase_features.csv',
        #                     prediction_file=f'./out/Balibase_trained/DL1/prediction_DL_{i}_mode2_msa_distance.csv',
        #                     error=0)
        # pickme = PickMeGame(features_file='/Users/kpolonsky/Downloads/BaliBase4/balibase_features.csv',
        #                     prediction_file=f'./out/BaliBase/RF/prediction_RF_balibase_msa_distance.csv',
        #                     error=0)
        # pickme = PickMeGame(features_file='./out/balibase_features_full_74.csv',
        #                     prediction_file=f'./out/BaliBase/DL2/prediction_DL_{i}_mode2_msa_distance.csv',
        #                     error=0)
        # pickme = PickMeGame(features_file='./out/orthomam_treebase_combined_features2.csv',
        #                     prediction_file=f'./out/treebase_orthomam_PRANK_combined/DL_new_sigmoid_transformed/prediction_DL_{i}_mode2_msa_distance.csv',
        #                     error=0)
        # pickme = PickMeGame(features_file='./out/orthomam_treebase_combined_features2.csv',
        #                     prediction_file=f'./out/treebase_orthomam_PRANK_combined/RF_new_transformed2/rf_prediction_{i}_mode2_msa_distance.csv',
        #                     error=0)

        # pickme = PickMeGame(features_file='./out/3M_features.csv',
        #                     prediction_file=f'./out/treeBase_4000/MSA_dist_RESULTS/RF_all_extept_all_SP/rf_prediction_{i}_mode2_msa_distance.csv',
        #                     error=0)
        # pickme = PickMeGame(features_file='./out/500k_features.csv',
        #                     prediction_file=f'./out/MSA_dist_RESULTS/RF_all_except2/rf_prediction_{i}_mode2_msa_distance.csv',
        #                     error=0)
        # pickme = PickMeGame(features_file='./out/500k_features.csv', prediction_file=f'./out/MSA_dist_RESULTS/DL_all_except2_batch64_32_32_LR.0.001.fixed/prediction_DL_{i}_mode2_msa_distance.csv', error=0)
        # pickme = PickMeGame(features_file='./out/500k_features.csv', prediction_file=f'./out/Tree_Dist_RESULTS/DL_model2_64_16_32_LeakyRelu_LeakyRelu_Elu_bs.32.epochs.30_LR.0.0001_validspl.0.2/prediction_DL_{i}_mode2_tree_distance.csv',
        #                     error=0, predicted_measure='tree_distance')
        pickme.save_to_csv(i)
        pickme.plot_results(i)
        pickme.plot_SoP_results(i)
        for condition in pickme.accumulated_sop_data:
            if condition not in sop_data_dict:
                sop_data_dict[condition] = {'sum': 0, 'count': 0}

            sop_data_dict[condition]['sum'] += pickme.accumulated_sop_data[condition]['sum']
            sop_data_dict[condition]['count'] += pickme.accumulated_sop_data[condition]['count']
        for condition in pickme.accumulated_data:
            if condition not in data_dict:
                data_dict[condition] = {'sum': 0, 'count': 0}

            data_dict[condition]['sum'] += pickme.accumulated_data[condition]['sum']
            data_dict[condition]['count'] += pickme.accumulated_data[condition]['count']
    if n > 1:
        pickme.accumulated_sop_data = sop_data_dict
        pickme.accumulated_data = data_dict
        pickme.plot_average_results(n)
