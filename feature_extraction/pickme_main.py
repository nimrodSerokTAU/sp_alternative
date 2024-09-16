import numpy as np
from classes.pick_me import PickMeGame
from scipy import stats

if __name__ == '__main__':
    for i in range(5):
        pickme = PickMeGame(features_file='/Users/kpolonsky/Downloads/TEST/500k_features.csv', prediction_file=f'/Users/kpolonsky/Downloads/TEST/Features/prediction_DL_{i}.csv', error=0)
        pickme.save_to_csv(i)
        pickme.plot_results(i)
