"""
PointNet implementation in tensorflow.keras
Look at the bottom __main__ section to see how to use PointNet architecture.
See paper: https://arxiv.org/abs/1612.00593
"""
import tensorflow as tf
import numpy as np
from .layers import LayerStack
from snoop import pp

kr = tf.keras
kl = kr.layers


def conv_stack(filters, kernel_size=1, batch_norm=True):
    layers = []
    for f in filters:
        layers.append(kl.Conv1D(f, kernel_size, activation="relu"))
        if batch_norm:
            layers.append(kl.BatchNormalization())
    return LayerStack(layers)


def dense_stack(units, batch_norm=True):
    layers = []
    for unit in units:
        layers.append(kl.Dense(unit, activation="relu"))
        if batch_norm:
            layers.append(kl.BatchNormalization())
    return LayerStack(layers)


def build_tnet(input_dim, output_dim, n_points=None, name="t_net", **kwargs):
    """
    T-Net learns a transformation matrix to multiply to the points to obtain input
    permutation invariance. Look at figure 2 in the paper to see where it fits
    in the entire PointNet architecture.

    The input is shape (batch_size, n_points, input_dim).
    The output is shape (batch_size, output_dim, output_dim).
    """
    inp = kr.Input(shape=(n_points, input_dim))
    x = conv_stack([64, 128, 1024])(inp)
    x = kl.GlobalMaxPooling1D()(x)
    x = dense_stack([512, 256])(x)
    output_dim_sqr = output_dim * output_dim
    x = kl.Dense(
        output_dim_sqr,
        weights=[np.zeros([256, output_dim_sqr]), np.eye(output_dim).flatten()],
    )(x)
    x = kl.Reshape((output_dim, output_dim))(x)
    model = kr.Model(inp, x, name=name, **kwargs)
    model.input_dim = input_dim
    model.output_dim = output_dim
    return model


if __name__ == "__main__":
    print(__doc__)
    n_points = 1000
    input_tnet = build_tnet(3, 3, n_points=n_points, name="input_tnet")
    feature_tnet = build_tnet(64, 64, n_points=n_points, name="feature_tnet")

    feature_tnet.summary()
