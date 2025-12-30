import numpy as np
import tensorflow as tf
from scipy.stats import kendalltau, spearmanr
from tensorflow.keras.callbacks import Callback

class PerBatchRankingMetrics(Callback):
    def __init__(self, val_generator, metric: str = "kendall", verbose: int = 1):
        """
        val_generator: custom validation BatchGenerator (must yield (X, y))
        metric: 'kendall' or 'spearman'
        """
        super().__init__()
        self.val_generator = val_generator
        self.metric = metric
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        correlations = []

        for X_batch, y_batch in self.val_generator:
            y_pred = self.model.predict(X_batch, verbose=0).flatten()
            y_true = y_batch.flatten()

            # Compute correlation only if > 1 unique label
            if np.std(y_true) == 0 or len(np.unique(y_true)) < 2:
                continue

            if self.metric == "kendall":
                corr, _ = kendalltau(y_true, y_pred)
            elif self.metric == "spearman":
                corr, _ = spearmanr(y_true, y_pred)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            if not np.isnan(corr):
                correlations.append(corr)

        mean_corr = np.mean(correlations) if correlations else np.nan

        logs[f"val_{self.metric}"] = mean_corr

        # if self.verbose:
        #     print(f"\nEpoch {epoch+1}: mean val_{self.metric} = {mean_corr:.4f}")

        # save in self.params so Keras logs it
        self.model.history.history[f"val_{self.metric}"] = \
            self.model.history.history.get(f"val_{self.metric}", []) + [mean_corr]
