from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from dl_model.config.config import TrainConfig, OutputConfig
from dl_model.modeling.model_builder import build_model
from dl_model.modeling.losses import make_loss
from dl_model.batching.batch_generator import BatchGenerator

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, out_cfg: OutputConfig):
        self.out_cfg = out_cfg

    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series,
        train_cfg: TrainConfig,
        train_df: pd.DataFrame | None = None,
        groups_train: pd.Series | None = None,
        aligners_train: pd.Series | None = None,
    ):
        tf.config.set_visible_devices([], "GPU")

        model = build_model(X_train.shape[1], train_cfg)
        # loss = make_loss(train_cfg)
        loss = make_loss(
            train_cfg.loss_fn,
            top_k=train_cfg.top_k,
            mse_weight=train_cfg.mse_weight,
            ranking_weight=train_cfg.ranking_weight,
            alpha=train_cfg.alpha,
            beta=train_cfg.beta,
            eps=train_cfg.eps,
            margin=train_cfg.margin,
            tau=train_cfg.tau,
            topk_weight_decay=train_cfg.topk_weight_decay,
        )

        model.compile(optimizer=Adam(learning_rate=train_cfg.learning_rate), loss=loss)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6, verbose=1),
        ]

        if train_cfg.batch_generation == "custom":
            if train_df is None or groups_train is None:
                raise ValueError("custom batch_generation requires train_df and groups_train")

            unique_codes = groups_train.unique()
            train_ids, val_ids = train_test_split(unique_codes, test_size=0.2, random_state=42)

            batch_gen = BatchGenerator(
                features=X_train, true_labels=y_train, true_msa_ids=groups_train,
                train_msa_ids=train_ids, val_msa_ids=val_ids, aligners=aligners_train,
                batch_size=train_cfg.batch_size, validation_split=train_cfg.validation_split,
                is_validation=False, repeats=train_cfg.repeats, mixed_portion=train_cfg.mixed_portion,
                per_aligner=train_cfg.per_aligner, features_w_names=train_df
            )

            val_gen = BatchGenerator(
                features=X_train, true_labels=y_train, true_msa_ids=groups_train,
                train_msa_ids=train_ids, val_msa_ids=val_ids, aligners=aligners_train,
                batch_size=train_cfg.batch_size, validation_split=train_cfg.validation_split,
                is_validation=True, repeats=train_cfg.repeats, mixed_portion=train_cfg.mixed_portion,
                per_aligner=train_cfg.per_aligner, features_w_names=train_df
            )

            history = model.fit(batch_gen, validation_data=val_gen, epochs=train_cfg.epochs,
                                verbose=self.out_cfg.verbose, callbacks=callbacks)

        else:
            history = model.fit(
                X_train, y_train,
                epochs=train_cfg.epochs,
                batch_size=train_cfg.batch_size,
                validation_split=train_cfg.validation_split,
                verbose=self.out_cfg.verbose,
                callbacks=callbacks,
            )

        test_loss = float(model.evaluate(X_test, y_test, verbose=0))
        y_pred = model.predict(X_test, verbose=0).reshape(-1).astype(np.float64)

        mse = float(mean_squared_error(y_test, y_pred))
        r, p = pearsonr(y_test.astype(np.float64), y_pred)

        val_loss = float(history.history["val_loss"][-1])

        return {
            "model": model,
            "history": history,
            "test_loss": test_loss,
            "val_loss": val_loss,
            "mse": mse,
            "pearson_r": float(r),
            "pearson_p": float(p),
            "y_pred": y_pred,
        }
