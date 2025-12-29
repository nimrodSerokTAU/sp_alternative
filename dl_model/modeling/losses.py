from __future__ import annotations
import tensorflow as tf

def _flatten(y_true: tf.Tensor, y_pred: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Flatten to shape (n,) and cast to float32."""
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    return y_true, y_pred


def _rank_loss_topk(y_true: tf.Tensor, y_pred: tf.Tensor, top_k: int) -> tf.Tensor:
    """
    sort by y_true (ascending), take top_k (best),
    penalize squared difference between true and predicted values in that slice
    """
    y_true, y_pred = _flatten(y_true, y_pred)
    paired = tf.stack([y_true, y_pred], axis=1)  # (n,2)
    idx = tf.argsort(paired[:, 0], axis=0, direction="ASCENDING")
    sorted_paired = tf.gather(paired, idx, axis=0)
    k = tf.minimum(tf.cast(top_k, tf.int32), tf.shape(sorted_paired)[0])
    true_top = sorted_paired[:k, 0]
    pred_top = sorted_paired[:k, 1]
    return tf.reduce_mean(tf.square(true_top - pred_top))


def _topk_ranking_hinge(y_true: tf.Tensor, y_pred: tf.Tensor, top_k: int, margin: float) -> tf.Tensor:
    """
    - Find true top-k by y_true ascending (best are smallest)
    - Encourage their y_pred to be smaller than non-topk by margin (hinge)
    """
    y_true, y_pred = _flatten(y_true, y_pred)
    n = tf.shape(y_true)[0]
    k = tf.minimum(tf.cast(top_k, tf.int32), n)

    true_order = tf.argsort(y_true, direction="ASCENDING")
    true_topk = true_order[:k]                       # (k,)
    pred_all = y_pred                                # (n,)
    pred_topk = tf.gather(y_pred, true_topk)          # (k,)

    # pairwise hinge: relu(margin + pred_topk[i] - pred_all[j])
    loss_matrix = tf.nn.relu(tf.cast(margin, tf.float32) + pred_topk[:, None] - pred_all[None, :])  # (k,n)

    # mask out comparisons among topk itself
    topk_mask = tf.scatter_nd(true_topk[:, None], tf.ones([k], tf.float32), [n])  # (n,)
    non_topk_mask = (1.0 - topk_mask)[None, :]                                    # (1,n)
    loss_matrix = loss_matrix * non_topk_mask

    return tf.reduce_mean(loss_matrix)


def _ranknet_pairwise(y_true: tf.Tensor, y_pred: tf.Tensor, margin: float, eps: float) -> tf.Tensor:
    """
    S=1 where (y_true_i < y_true_j) => i should rank higher (smaller true is better)
    P = sigmoid(diff_pred - margin)
    """
    y_true, y_pred = _flatten(y_true, y_pred)

    diff_true = tf.expand_dims(y_true, 1) - tf.expand_dims(y_true, 0)
    diff_pred = tf.expand_dims(y_pred, 1) - tf.expand_dims(y_pred, 0)

    S = tf.cast(diff_true < 0, tf.float32)
    P = tf.nn.sigmoid(diff_pred - tf.cast(margin, tf.float32))

    eps = tf.cast(eps, tf.float32)
    loss = -(S * tf.math.log(P + eps) + (1.0 - S) * tf.math.log(1.0 - P + eps))
    return tf.reduce_mean(loss)


def _kendall_smooth(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true, y_pred = _flatten(y_true, y_pred)
    diff_true = tf.expand_dims(y_true, 1) - tf.expand_dims(y_true, 0)
    diff_pred = tf.expand_dims(y_pred, 1) - tf.expand_dims(y_pred, 0)
    sign_true = tf.sign(diff_true)
    sign_pred = tf.tanh(diff_pred) # smooth sign
    tau = tf.reduce_mean(sign_true * sign_pred)
    return 1.0 - tau # maximize Kendall's τ --> minimize 1 - τ


def _listnet(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true, y_pred = _flatten(y_true, y_pred)
    true_prob = tf.nn.softmax(y_true, axis=-1)
    pred_prob = tf.nn.softmax(y_pred, axis=-1)
    return -tf.reduce_sum(true_prob * tf.math.log(pred_prob + 1e-10))


def _approx_ndcg(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float) -> tf.Tensor:
    y_true, y_pred = _flatten(y_true, y_pred)
    eps = tf.cast(eps, tf.float32)

    pred_diffs = tf.expand_dims(y_pred, 1) - tf.expand_dims(y_pred, 0)
    soft_rank = tf.reduce_sum(tf.nn.sigmoid(-pred_diffs), axis=-1) + 1.0  # (n,)

    gains = tf.pow(2.0, y_true) - 1.0
    discounts = 1.0 / (tf.math.log(1.0 + soft_rank) / tf.math.log(2.0))
    dcg = tf.reduce_sum(gains * discounts)

    sorted_true = tf.sort(y_true, direction="DESCENDING")
    ideal_gains = tf.pow(2.0, sorted_true) - 1.0
    ideal_discounts = 1.0 / (tf.math.log(1.0 + tf.range(1, tf.size(y_true) + 1, dtype=tf.float32)) / tf.math.log(2.0))
    idcg = tf.reduce_sum(ideal_gains * ideal_discounts)

    ndcg = dcg / (idcg + eps)
    return 1.0 - ndcg # maximize NDCG --> minimize 1 - NDCG


def _weighted_ranknet(y_true: tf.Tensor, y_pred: tf.Tensor, topk_weight_decay: float, eps: float) -> tf.Tensor:
    y_true, y_pred = _flatten(y_true, y_pred)
    eps = tf.cast(eps, tf.float32)
    decay = tf.cast(topk_weight_decay, tf.float32)

    diff_true = tf.expand_dims(y_true, 1) - tf.expand_dims(y_true, 0)
    diff_pred = tf.expand_dims(y_pred, 1) - tf.expand_dims(y_pred, 0)

    S = tf.cast(diff_true < 0, tf.float32)
    P = tf.nn.sigmoid(-diff_pred)

    ranks = tf.argsort(tf.argsort(y_true))  # smaller y_true => smaller rank index
    ranks = tf.cast(ranks, tf.float32)
    item_w = tf.exp(-decay * ranks)
    pair_w = (tf.expand_dims(item_w, 1) + tf.expand_dims(item_w, 0)) / 2.0

    loss = -pair_w * (S * tf.math.log(P + eps) + (1.0 - S) * tf.math.log(1.0 - P + eps))
    return tf.reduce_sum(loss) / (tf.reduce_sum(pair_w) + eps)


def _hybrid_weighted_ranknet(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    topk_weight_decay: float,
    alpha: float,
    beta: float,
    margin: float,
    eps: float,
) -> tf.Tensor:
    y_true, y_pred = _flatten(y_true, y_pred)
    eps = tf.cast(eps, tf.float32)
    decay = tf.cast(topk_weight_decay, tf.float32)
    alpha = tf.cast(alpha, tf.float32)
    beta = tf.cast(beta, tf.float32)
    margin = tf.cast(margin, tf.float32)

    diff_true = tf.expand_dims(y_true, 1) - tf.expand_dims(y_true, 0)
    diff_pred = tf.expand_dims(y_pred, 1) - tf.expand_dims(y_pred, 0)

    S = tf.cast(diff_true < 0, tf.float32)
    P = tf.nn.sigmoid(-diff_pred)

    ranks = tf.argsort(tf.argsort(y_true))
    ranks = tf.cast(ranks, tf.float32)
    item_w = tf.exp(-decay * ranks)
    pair_w = (tf.expand_dims(item_w, 1) + tf.expand_dims(item_w, 0)) / 2.0

    pairwise = -pair_w * (S * tf.math.log(P + eps) + (1.0 - S) * tf.math.log(1.0 - P + eps))
    pairwise = tf.reduce_sum(pairwise) / (tf.reduce_sum(pair_w) + eps)

    min_val = tf.reduce_min(y_true)
    best_mask = tf.cast(tf.equal(y_true, min_val), tf.float32)
    best_class = -tf.reduce_mean(best_mask * tf.math.log(tf.nn.sigmoid(-y_pred) + eps))

    non_best = 1.0 - best_mask
    best_pred = tf.reduce_min(y_pred)
    margin_loss = tf.reduce_mean(non_best * tf.nn.relu(margin - (y_pred - best_pred)))

    return pairwise + alpha * best_class + beta * margin_loss


# ============================================================
# Serializable Loss Classes
# ============================================================

@tf.keras.utils.register_keras_serializable(package="msa_regression")
class MSEWithRankLoss(tf.keras.losses.Loss):
    def __init__(self, top_k: int = 4, mse_weight: float = 1.0, ranking_weight: float = 0.3, name: str = "mse_with_rank_loss"):
        super().__init__(name=name)
        self.top_k = int(top_k)
        self.mse_weight = float(mse_weight)
        self.ranking_weight = float(ranking_weight)

    def call(self, y_true, y_pred):
        y_true, y_pred = _flatten(y_true, y_pred)
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        r = _rank_loss_topk(y_true, y_pred, self.top_k)
        return tf.cast(self.mse_weight, tf.float32) * mse + tf.cast(self.ranking_weight, tf.float32) * r

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"top_k": self.top_k, "mse_weight": self.mse_weight, "ranking_weight": self.ranking_weight})
        return cfg


@tf.keras.utils.register_keras_serializable(package="msa_regression")
class MSEWithTopKRankLoss(tf.keras.losses.Loss):
    def __init__(self, top_k: int = 4, ranking_weight: float = 0.3, margin: float = 0.0, name: str = "mse_with_topk_rank_loss"):
        super().__init__(name=name)
        self.top_k = int(top_k)
        self.ranking_weight = float(ranking_weight)
        self.margin = float(margin)

    def call(self, y_true, y_pred):
        y_true, y_pred = _flatten(y_true, y_pred)
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        r = _topk_ranking_hinge(y_true, y_pred, self.top_k, self.margin)
        return mse + tf.cast(self.ranking_weight, tf.float32) * r

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"top_k": self.top_k, "ranking_weight": self.ranking_weight, "margin": self.margin})
        return cfg


@tf.keras.utils.register_keras_serializable(package="msa_regression")
class RankNetLoss(tf.keras.losses.Loss):
    def __init__(self, margin: float = 0.0, eps: float = 1e-6, name: str = "ranknet_loss"):
        super().__init__(name=name)
        self.margin = float(margin)
        self.eps = float(eps)

    def call(self, y_true, y_pred):
        return _ranknet_pairwise(y_true, y_pred, self.margin, self.eps)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"margin": self.margin, "eps": self.eps})
        return cfg


@tf.keras.utils.register_keras_serializable(package="msa_regression")
class HybridMSERankNetLoss(tf.keras.losses.Loss):
    def __init__(self, alpha: float = 0.5, margin: float = 0.0, eps: float = 1e-6, name: str = "hybrid_mse_ranknet_loss"):
        super().__init__(name=name)
        self.alpha = float(alpha)
        self.margin = float(margin)
        self.eps = float(eps)

    def call(self, y_true, y_pred):
        y_true, y_pred = _flatten(y_true, y_pred)
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        r = _ranknet_pairwise(y_true, y_pred, self.margin, self.eps)
        alpha = tf.cast(self.alpha, tf.float32)
        return alpha * mse + (1.0 - alpha) * r

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "margin": self.margin, "eps": self.eps})
        return cfg


@tf.keras.utils.register_keras_serializable(package="msa_regression")
class HybridMSERankNetDynamicLoss(tf.keras.losses.Loss):
    def __init__(self, alpha_base: float = 0.98, margin: float = 0.0, eps: float = 1e-6, name: str = "hybrid_mse_ranknet_dynamic"):
        super().__init__(name=name)
        self.alpha_base = float(alpha_base)
        self.margin = float(margin)
        self.eps = float(eps)

    def call(self, y_true, y_pred):
        y_true, y_pred = _flatten(y_true, y_pred)
        eps = tf.cast(self.eps, tf.float32)
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        r = _ranknet_pairwise(y_true, y_pred, self.margin, self.eps)
        ratio = tf.stop_gradient(r / (mse + eps))
        alpha = tf.clip_by_value(tf.cast(self.alpha_base, tf.float32) / (1.0 + ratio), 0.9, 0.995)
        return alpha * mse + (1.0 - alpha) * r

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha_base": self.alpha_base, "margin": self.margin, "eps": self.eps})
        return cfg


@tf.keras.utils.register_keras_serializable(package="msa_regression")
class KendallLoss(tf.keras.losses.Loss):
    def __init__(self, name: str = "kendall_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return _kendall_smooth(y_true, y_pred)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="msa_regression")
class ListNetLoss(tf.keras.losses.Loss):
    def __init__(self, name: str = "listnet_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return _listnet(y_true, y_pred)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="msa_regression")
class ApproxNDCGLoss(tf.keras.losses.Loss):
    def __init__(self, eps: float = 1e-10, name: str = "approx_ndcg_loss"):
        super().__init__(name=name)
        self.eps = float(eps)

    def call(self, y_true, y_pred):
        return _approx_ndcg(y_true, y_pred, self.eps)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"eps": self.eps})
        return cfg


@tf.keras.utils.register_keras_serializable(package="msa_regression")
class HybridMSEApproxNDCGLoss(tf.keras.losses.Loss):
    def __init__(self, alpha: float = 0.5, eps: float = 1e-10, name: str = "hybrid_mse_approx_ndcg_loss"):
        super().__init__(name=name)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def call(self, y_true, y_pred):
        y_true, y_pred = _flatten(y_true, y_pred)
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        ndcg_l = _approx_ndcg(y_true, y_pred, self.eps)
        alpha = tf.cast(self.alpha, tf.float32)
        return alpha * mse + (1.0 - alpha) * ndcg_l

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "eps": self.eps})
        return cfg

@tf.keras.utils.register_keras_serializable(package="msa_regression")
class WeightedRankNetLoss(tf.keras.losses.Loss):
    def __init__(self, topk_weight_decay: float = 0.3, eps: float = 1e-6, name: str = "weighted_ranknet_loss"):
        super().__init__(name=name)
        self.topk_weight_decay = float(topk_weight_decay)
        self.eps = float(eps)

    def call(self, y_true, y_pred):
        return _weighted_ranknet(y_true, y_pred, self.topk_weight_decay, self.eps)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"topk_weight_decay": self.topk_weight_decay, "eps": self.eps})
        return cfg


@tf.keras.utils.register_keras_serializable(package="msa_regression")
class HybridWeightedRankNetLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        topk_weight_decay: float = 0.3,
        alpha: float = 0.5,
        beta: float = 0.2,
        margin: float = 0.2,
        eps: float = 1e-6,
        name: str = "hybrid_weighted_ranknet_loss",
    ):
        super().__init__(name=name)
        self.topk_weight_decay = float(topk_weight_decay)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.margin = float(margin)
        self.eps = float(eps)

    def call(self, y_true, y_pred):
        return _hybrid_weighted_ranknet(
            y_true, y_pred,
            topk_weight_decay=self.topk_weight_decay,
            alpha=self.alpha,
            beta=self.beta,
            margin=self.margin,
            eps=self.eps,
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "topk_weight_decay": self.topk_weight_decay,
            "alpha": self.alpha,
            "beta": self.beta,
            "margin": self.margin,
            "eps": self.eps,
        })
        return cfg


def make_loss(loss_fn: str, **kwargs):
    """
    Returns a serializable loss object (need for Keras to save the model).
    Args:
        loss_fn: name of the loss function
        **kwargs: parameters for the loss function
    Returns:
        A loss function object or string identifier.
    """
    if loss_fn == "mse":
        return "mean_squared_error"

    if loss_fn == "custom_mse":
        return MSEWithRankLoss(
            top_k=kwargs.get("top_k", 4),
            mse_weight=kwargs.get("mse_weight", 1.0),
            ranking_weight=kwargs.get("ranking_weight", 0.3),
        )

    # if loss_fn == "mse_with_topk_rank_loss":
    #     return MSEWithTopKRankLoss(
    #         top_k=kwargs.get("top_k", 4),
    #         ranking_weight=kwargs.get("ranking_weight", 0.3),
    #         margin=kwargs.get("margin", 0.0),
    #     )

    if loss_fn == "ranknet_loss":
        return RankNetLoss(
            margin=kwargs.get("margin", 0.0),
            eps=kwargs.get("eps", 1e-6),
        )

    if loss_fn == "hybrid_mse_ranknet_loss":
        return HybridMSERankNetLoss(
            alpha=kwargs.get("alpha", 0.5),
            margin=kwargs.get("margin", 0.0),
            eps=kwargs.get("eps", 1e-6),
        )

    if loss_fn == "hybrid_mse_ranknet_dynamic":
        return HybridMSERankNetDynamicLoss(
            alpha_base=kwargs.get("alpha", 0.98),
            margin=kwargs.get("margin", 0.0),
            eps=kwargs.get("eps", 1e-6),
        )

    if loss_fn == "kendall_loss":
        return KendallLoss()

    # if loss_fn == "soft_kendall_loss":
    #     return SoftKendallLoss(tau=kwargs.get("tau", 1.0))

    if loss_fn == "listnet_loss":
        return ListNetLoss()

    if loss_fn == "approx_ndcg_loss":
        return ApproxNDCGLoss(eps=kwargs.get("eps", 1e-10))

    if loss_fn == "hybrid_mse_approx_ndcg_loss":
        return HybridMSEApproxNDCGLoss(alpha=kwargs.get("alpha", 0.5), eps=kwargs.get("eps", 1e-10))

    # if loss_fn == "lambda_rank_loss":
    #     return LambdaRankLoss(eps=kwargs.get("eps", 1e-10))

    if loss_fn == "weighted_ranknet_loss":
        return WeightedRankNetLoss(
            topk_weight_decay=kwargs.get("topk_weight_decay", 0.3),
            eps=kwargs.get("eps", 1e-6),
        )

    if loss_fn == "hybrid_weighted_ranknet_loss":
        return HybridWeightedRankNetLoss(
            topk_weight_decay=kwargs.get("topk_weight_decay", 0.3),
            alpha=kwargs.get("alpha", 0.5),
            beta=kwargs.get("beta", 0.2),
            margin=kwargs.get("margin", 0.2),
            eps=kwargs.get("eps", 1e-6),
        )

    raise ValueError(f"Unknown loss_fn: {loss_fn}")