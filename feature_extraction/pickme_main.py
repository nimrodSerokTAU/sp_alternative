import numpy as np
from classes.pick_me import PickMeGame
from scipy import stats

if __name__ == '__main__':
    for i in range(5):
        # pickme = PickMeGame(features_file='./out/500k_features.csv', prediction_file=f'./out/prediction_DL_{i}_mode1_msa_distance.csv', error=0)
        pickme = PickMeGame(features_file='./out/500k_features.csv', prediction_file=f'./out/rf_prediction_{i}_mode2_msa_distance.csv',
                            error=0)
        pickme.save_to_csv(i)
        pickme.plot_results(i)
