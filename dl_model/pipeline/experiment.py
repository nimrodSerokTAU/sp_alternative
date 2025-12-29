from __future__ import annotations
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from dl_model.config.config import DataConfig, FeatureConfig, TrainConfig, OutputConfig, ShapExplainConfig
from dl_model.config.constants import CODE_COL, GROUP_COL, ALIGNER_COL
from dl_model.data_processing.io import read_features
from dl_model.data_processing.preprocess import DatasetPreprocessor
from dl_model.data_processing.features_selector import FeatureSelector
from dl_model.data_processing.scaling import FeatureScaler
from dl_model.modeling.model_train import Trainer
from dl_model.evaluation.metrics import per_msa_correlations, top50_percentage
from dl_model.export.writer import OutputsWriter
from dl_model.export.writer import save_loss_plot, save_correlation_plot
from dl_model.evaluation.shap_explain import run_shap_keras

logger = logging.getLogger(__name__)


class RegressionExperiment:
    def __init__(
        self,
        data_cfg: DataConfig,
        feat_cfg: FeatureConfig,
        train_cfg: TrainConfig,
        out_cfg: OutputConfig,
        explain_cfg: ShapExplainConfig = ShapExplainConfig(enabled=False),
    ):
        self.data_cfg = data_cfg
        self.feat_cfg = feat_cfg
        self.train_cfg = train_cfg
        self.out_cfg = out_cfg
        self.explain_cfg = explain_cfg

        self.prep = DatasetPreprocessor(data_cfg)
        self.selector = FeatureSelector(feat_cfg, target_col=data_cfg.true_score_name)
        self.scaler = FeatureScaler(feat_cfg)
        self.trainer = Trainer(out_cfg)
        self.writer = OutputsWriter(out_cfg)

    def run(self) -> dict:
        df = read_features(self.data_cfg.features_file)
        df = self.prep.preprocess(df)

        if self.out_cfg.save_features_scv:
            self.writer.save_features_csv(df, filename=f"features_w_aligner_{self.feat_cfg.mode}_{self.data_cfg.true_score_name}.csv")

        codes = df[GROUP_COL].unique()
        train_codes, test_codes = train_test_split(
            codes, test_size=self.data_cfg.test_size, random_state=self.data_cfg.random_state
        )

        train_df = df[df[GROUP_COL].isin(train_codes)].copy()
        test_df = df[df[GROUP_COL].isin(test_codes)].copy()

        X_train_df, y_train = self.selector.select(train_df)
        X_test_df, y_test = self.selector.select(test_df)

        X_train, X_test = self.scaler.fit_transform_X(train_df, test_df, X_train_df, X_test_df)

        groups_train = train_df[GROUP_COL]
        groups_test = test_df[GROUP_COL]
        y_train_s, y_test_s = self.scaler.transform_y(y_train, y_test, groups_train, groups_test)

        if self.data_cfg.true_score_name != "class_label":
            y_train_s = y_train_s.astype("float64")
            y_test_s = y_test_s.astype("float64")
        else:
            y_train_s = y_train_s.astype("int")
            y_test_s = y_test_s.astype("int")

        results = self.trainer.fit_predict(
            X_train=X_train, y_train=y_train_s,
            X_test=X_test, y_test=y_test_s,
            train_cfg=self.train_cfg,
            train_df=train_df,
            groups_train=groups_train,
            aligners_train=train_df[ALIGNER_COL],
        )

        # metrics
        corr = per_msa_correlations(groups_test, y_test_s, results["y_pred"], top_k=self.train_cfg.top_k)
        top50 = top50_percentage(groups_test, test_df[CODE_COL], y_test_s, results["y_pred"], top_n=50)

        metrics = {
            "mse": results["mse"],
            "test_loss": results["test_loss"],
            "val_loss": results["val_loss"],
            "pearson_r": results["pearson_r"],
            "pearson_p": results["pearson_p"],
            **corr,
            "top50_percentage": top50,
        }

        logger.info("Final metrics: %s", metrics)

        # save artifacts
        rid = self.out_cfg.run_id

        if self.out_cfg.save_plots:
            save_loss_plot(results["history"], f"{self.out_cfg.out_dir}/loss_graph_{rid}_mode{self.feat_cfg.mode}_{self.data_cfg.true_score_name}.png")
            save_correlation_plot(y_test_s, results["y_pred"], self.train_cfg.loss_fn, self.feat_cfg.mode, self.data_cfg.true_score_name, results["mse"], rid, out_path=self.out_cfg.out_dir)

        if self.out_cfg.save_model:
            self.writer.save_model(results["model"], f"regressor_model_{rid}_mode{self.feat_cfg.mode}_{self.data_cfg.true_score_name}.keras")

        if self.out_cfg.save_scaled_csv:
            self.writer.save_scaled_csv(
                X_train, train_df[CODE_COL], train_df[GROUP_COL], y_train_s,
                filename=f"train_scaled_{rid}.csv",
                feature_names=self.scaler.feature_names_out
            )
            self.writer.save_scaled_csv(
                X_test, test_df[CODE_COL], test_df[GROUP_COL], y_test_s,
                filename=f"test_scaled_{rid}.csv",
                feature_names=self.scaler.feature_names_out
            )

        # save scaler
        if self.scaler.scaler is not None and self.out_cfg.save_scaler:
            self.writer.save_scaler(self.scaler.scaler, f"scaler_{rid}_mode{self.feat_cfg.mode}_{self.data_cfg.true_score_name}.pkl")

        if self.out_cfg.save_predictions_csv:
            self.writer.save_predictions(groups_test, test_df[CODE_COL], results["y_pred"],
                                         filename=f"prediction_DL_{rid}_mode{self.feat_cfg.mode}_{self.data_cfg.true_score_name}.csv")

        # SHAP explainability
        if self.explain_cfg.enabled:
            X_test_named = pd.DataFrame(X_test, columns=self.scaler.feature_names_out or list(X_test_df.columns))
            run_shap_keras(results["model"], X_test_named, out_dir=self.out_cfg.out_dir,
                           sample_n=self.explain_cfg.sample_n, run_id=rid)

        return {"metrics": metrics, "results": results}
