from types import SimpleNamespace

import tensorflow as tf


def enable_tf1_compat():
    tf.get_logger().setLevel("ERROR")
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.disable_v2_behavior()

    for name in (
        "AUTO_REUSE",
        "ConfigProto",
        "Graph",
        "GPUOptions",
        "RunMetadata",
        "RunOptions",
        "Session",
        "Summary",
        "assign",
        "get_collection",
        "get_default_graph",
        "get_variable",
        "global_variables",
        "global_variables_initializer",
        "gradients",
        "placeholder",
        "random_normal",
        "random_uniform",
        "reset_default_graph",
        "set_random_seed",
        "summary",
        "train",
        "trainable_variables",
        "variable_scope",
    ):
        if not hasattr(tf, name):
            setattr(tf, name, getattr(tf.compat.v1, name))

    tf.layers = tf.compat.v1.layers
    tf.summary = tf.compat.v1.summary
    tf.train = tf.compat.v1.train

    if not hasattr(tf, "log"):
        tf.log = tf.math.log

    if not hasattr(tf.nn, "softmax_cross_entropy_with_logits_v2"):
        tf.nn.softmax_cross_entropy_with_logits_v2 = (
            tf.nn.softmax_cross_entropy_with_logits
        )

    if not hasattr(tf, "contrib"):
        tf.contrib = SimpleNamespace(layers=SimpleNamespace(layer_norm=_layer_norm))

    return tf


def _layer_norm(inputs, center=True, scale=True, epsilon=1e-5, **_kwargs):
    return tf.keras.layers.LayerNormalization(
        center=center,
        scale=scale,
        epsilon=epsilon,
    )(inputs)
