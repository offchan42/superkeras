"""
PointNet implementation in tensorflow.keras
Look at the bottom __main__ section to see how to use PointNet architecture.
See paper: https://arxiv.org/abs/1612.00593
"""
import tensorflow as tf
import numpy as np
from snoop import pp

kr = tf.keras
kl = kr.layers


def build_tnet(input_dim, output_dim, n_points=None, name="t_net", **kwargs):
    """
    T-Net learns a transformation matrix to multiply to the points to obtain input
    permutation invariance. Look at figure 2 in the paper to see where it fits
    in the entire PointNet architecture.

    The input is shape (batch_size, n_points, input_dim).
    The output is shape (batch_size, output_dim, output_dim).
    """
    inp = kr.Input(shape=(n_points, input_dim))
    x = kl.Convolution1D(64, 1, activation="relu")(inp)
    x = kl.BatchNormalization()(x)
    x = kl.Convolution1D(128, 1, activation="relu")(x)
    x = kl.BatchNormalization()(x)
    x = kl.Convolution1D(1024, 1, activation="relu")(x)
    x = kl.BatchNormalization()(x)
    x = kl.GlobalMaxPooling1D()(x)
    x = kl.Dense(512, activation="relu")(x)
    x = kl.BatchNormalization()(x)
    x = kl.Dense(256, activation="relu")(x)
    x = kl.BatchNormalization()(x)
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
    tnet = build_tnet(3, 3, n_points=1000)
    tnet.summary()
