
from classes import regressor


if __name__ == '__main__':
    model = regressor.Regressor("feature_extraction/all_features_ALL_NEW_541.csv",0.1)
    model.random_forest()
    model.plot_results("rf")
    model.gradient_boost()
    model.plot_results("gbr")
    model.support_vector()
    model.plot_results("svr")
    model.k_nearest_neighbors()
    model.plot_results("knn-r")