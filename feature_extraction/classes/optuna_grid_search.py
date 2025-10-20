import optuna
from regressor import Regressor
import optuna.visualization as vis
import numpy as np
import pandas as pd
import os



######################## SETTINGS / CONFUGURATION ###############################
i: int = 0
mode: int = 1
true_score_name: str = 'dseq_from_true'

# features_file: str = '../out/ortho12_distant_features_260825.csv'
# empirical: bool = False

features_file: str = '../out/balibase_features_with_foldmason_161025.csv'
empirical: bool = True

# scaler_type="standard"
scaler_type="rank"

######################## SETTINGS ###############################


log_file = "../out/optuna_results.csv"

if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        headers = [
            "trial_number", "loss", "val_loss", "corr_coefficient",
            "neurons_1", "neurons_2", "neurons_3", "neurons_4",
            "dropout", "lr", "l1", "l2", "batch_size", "regularizer","top_k", "mse_weight", "ranking_weight",
            "batch_generation", "loss_fn"
        ]
        f.write(",".join(headers) + "\n")



def objective(trial):
    try:
        # Suggest hyperparameters
        neurons = [
            trial.suggest_int("neurons_1", 32, 512),
            trial.suggest_int("neurons_2", 32, 256),
            # trial.suggest_int("neurons_3", 0, 256),  # 0 means skip this layer
            # trial.suggest_int("neurons_4", 0, 256) # 0 means skip this layer
            trial.suggest_categorical("neurons_3", [0, 8, 16, 32, 64, 128, 256]),
            trial.suggest_categorical("neurons_4", [0, 8, 16, 32, 64, 128, 256])
        ]
        dropout_rate = trial.suggest_float("dropout", 0.1, 0.4)
        learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        l1 = trial.suggest_float("l1", 1e-7, 1e-4, log=True)
        l2 = trial.suggest_float("l2", 1e-7, 1e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 32, 64, 128, 256])
        regularizer_name = trial.suggest_categorical("regularizer", ["l1", "l2", "l1_l2"])
        # loss_fn = trial.suggest_categorical("loss_fn", ["mse", "custom_mse"])

        # For MODEL2 only:
        if scaler_type == "rank":
            top_k = trial.suggest_int("top_k", 2, 10)
            # mse_weight = trial.suggest_float("mse_weight", 1.0, 2.0)
            mse_weight = 1
            ranking_weight = trial.suggest_float("ranking_weight", 0.1, 10.0)
            batch_generation = "custom"
            loss_fn = 'custom_mse'

        # For MODEL1 only:
        elif scaler_type == "standard":
            top_k = 0
            mse_weight = 0
            ranking_weight = 0
            batch_generation = 'standard'
            loss_fn = 'mse'

        # model_instance = Regressor(features_file=features_file,
        #                                  verbose=0, # turn off during tuning
        #                                  test_size=0.2,
        #                                  mode=mode, i=i, remove_correlated_features=False,
        #                                  empirical=empirical, scaler_type=scaler_type, true_score_name=true_score_name)
        model_instance = Regressor(features_file=features_file,
                                   verbose=0,  # turn off during tuning
                                   test_size=0.2,
                                   mode=mode, i=i, remove_correlated_features=False,
                                   empirical=empirical, scaler_type=scaler_type, true_score_name=true_score_name)

        loss, val_loss, corr_coefficient = model_instance.deep_learning(
            epochs=50,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0,  # turn off during tuning
            learning_rate=learning_rate,
            neurons=neurons,
            dropout_rate=dropout_rate,
            l1=l1,
            l2=l2,
            i=trial.number,  # to name files per trial
            regularizer_name=regularizer_name,
            loss_fn=loss_fn,
            top_k=top_k,
            mse_weight=mse_weight,
            ranking_weight=ranking_weight,
            batch_generation=batch_generation
        )

        if not np.isfinite(corr_coefficient):
            raise ValueError("Non-finite score encountered")

        with open(log_file, "a") as f:
            f.write(f"{trial.number},{loss},{val_loss},{corr_coefficient},"
                    f"{neurons[0]},{neurons[1]},{neurons[2]},{neurons[3]},"
                    f"{dropout_rate},{learning_rate},{l1},{l2},{batch_size},{regularizer_name},"
                    f"{top_k},{mse_weight},{ranking_weight},{batch_generation},{loss_fn}\n")
    # except Exception as e:
    #     print(f"Trial {trial.number} failed: {e}")
    #     return float('inf')  # or a very bad score to discard trial
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.exceptions.TrialPruned()

    if scaler_type == "standard": #MODEL1
        return corr_coefficient  # Optuna will minimize this
    elif scaler_type == "rank": #MODEL2
        return val_loss  # Optuna will minimize this

if scaler_type == "standard": #MODEL1
    study = optuna.create_study(direction="maximize")
elif scaler_type == "rank": #MODEL2
    study = optuna.create_study(direction = "minimize")

study.optimize(objective, n_trials=50)

# Print best results
print("Best Trial:")
print(study.best_trial.params)

df = study.trials_dataframe()
df.to_csv("../out/optuna_study_all_trials.csv", index=False)

plot = vis.plot_param_importances(study)
plot.write_html("../out/param_importances.html")

vis.plot_optimization_history(study).show()
vis.plot_param_importances(study).show()
vis.plot_slice(study).show()
vis.plot_parallel_coordinate(study).show()
vis.plot_contour(study).show()
vis.plot_edf(study).show()

# fig1 = vis.plot_optimization_history(study)
# fig1.write_image("../out/optimization_history.png")
# fig2 = vis.plot_param_importances(study)
# fig2.write_image("../out/param_importances.png")
# fig3 = vis.plot_slice(study)
# fig3.write_image("../out/param_slice.png")
# fig4 = vis.plot_parallel_coordinate(study)
# fig4.write_image("../out/param_parallel_coordinate.png")
# fig5 = vis.plot_contour(study)
# fig5.write_image("../out/param_contour.png")
# fig6 = vis.plot_edf(study)
# fig6.write_image("../out/param_edf.png")