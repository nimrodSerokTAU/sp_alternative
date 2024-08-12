
from classes import regressor


if __name__ == '__main__':
    model = regressor.Regressor("/Users/kpolonsky/PycharmProjects/SOP/feature_extraction/all_features_ALL_NEW_no_pythia.csv",0.1)
    model.random_forest()
    model.plot_results("Random Forest Regression")
    model.gradient_boost()
    model.plot_results("Gradient Booster Regression")
    model.support_vector()
    model.plot_results("Support Vector Regression")
    model.k_nearest_neighbors()
    model.plot_results("K-Nearest Neighbors Regression")