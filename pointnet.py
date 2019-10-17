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


def build_tnet(n_points=None, name="t_net", **kwargs):
    """
    T-Net learns a transformation matrix to multiply to the points to obtain input
    permutation invariance. Look at figure 2 in the paper to see where it fits
    in the entire PointNet architecture.

    The input is shape (batch_size, n_points, input_dim).
    The output is shape (batch_size, output_dim, output_dim).
    If `output_dim` is 0 or None, it will be set the same as `input_dim`.
    `input_dim` will be inferred from the input tensor.
    """
    inp = kr.Input(shape=(n_points, 3))
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
    x = kl.Dense(
        9,
        weights=[
            np.zeros([256, 9]),
            np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32),
        ],
    )(x)
    x = kl.Reshape((3, 3))(x)
    return kr.Model(inp, x, name=name, **kwargs)


if __name__ == "__main__":
    print(__doc__)
    tnet = build_tnet()
    tnet.summary()
