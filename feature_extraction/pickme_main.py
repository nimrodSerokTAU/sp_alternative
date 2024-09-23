import numpy as np
from classes.pick_me import PickMeGame
from scipy import stats

if __name__ == '__main__':
    for i in range(5):
        pickme = PickMeGame(features_file='./out/500k_features.csv',
                            prediction_file=f'./out/MSA_dist_RESULTS/RF_all_except2/rf_prediction_{i}_mode2_msa_distance.csv',
                            error=0)
        # pickme = PickMeGame(features_file='./out/500k_features.csv', prediction_file=f'./out/MSA_dist_RESULTS/DL_all_except2_batch64_32_32_LR.0.001.fixed/prediction_DL_{i}_mode2_msa_distance.csv', error=0)
        # pickme = PickMeGame(features_file='./out/500k_features.csv', prediction_file=f'./out/Tree_Dist_RESULTS/DL_model2_64_16_32_LeakyRelu_LeakyRelu_Elu_bs.32.epochs.30_LR.0.0001_validspl.0.2/prediction_DL_{i}_mode2_tree_distance.csv',
        #                     error=0, predicted_measure='tree_distance')
        pickme.save_to_csv(i)
        pickme.plot_results(i)
