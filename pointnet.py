"""
PointNet implementation in tensorflow.keras
Look at the bottom __main__ section to see how to use PointNet architecture.
See paper: https://arxiv.org/abs/1612.00593
"""
import numpy as np
import tensorflow as tf

from .layers import LayerStack

kr = tf.keras
kl = kr.layers


def conv_stack(filters, kernel_size=1, batch_norm=True, depth=1.0):
    layers = []
    for f in filters:
        f = int(round(f * depth))
        layers.append(kl.Conv1D(f, kernel_size, activation="relu"))
        if batch_norm:
            layers.append(kl.BatchNormalization())
    return LayerStack(layers)


def dense_stack(units, batch_norm=True, depth=1.0):
    layers = []
    for unit in units:
        unit = int(round(unit * depth))
        layers.append(kl.Dense(unit, activation="relu"))
        if batch_norm:
            layers.append(kl.BatchNormalization())
    return LayerStack(layers)


def build_tnet(input_dim, output_dim, n_points=None, depth=1.0, name="t_net", **kwargs):
    """
    T-Net learns a transformation matrix to multiply to the points to obtain input
    permutation invariance. Look at figure 2 in the paper to see where it fits
    in the entire PointNet architecture.

    The input is shape (batch_size, n_points, input_dim).
    The output is shape (batch_size, output_dim, output_dim).
    """
    inp = kr.Input(shape=(n_points, input_dim))
    x = conv_stack([64, 128, 1024], depth=depth)(inp)
    x = kl.GlobalMaxPooling1D()(x)
    x = dense_stack([512, 256], depth=depth)(x)
    x_dims = x.shape[-1]
    output_dim_sqr = output_dim * output_dim
    x = kl.Dense(
        output_dim_sqr,
        weights=[np.zeros([x_dims, output_dim_sqr]), np.eye(output_dim).flatten()],
    )(x)
    x = kl.Reshape((output_dim, output_dim))(x)
    model = kr.Model(inp, x, name=name, **kwargs)
    model.input_dim = input_dim
    model.output_dim = output_dim
    return model


def transform(points, tnet, name="transform"):
    """
    Multiply `points` tensor by `tnet` output matrix.
    Return multiplied tensor with same shape as `points`"""
    return tf.matmul(points, tnet(points), name=name)


def ExpandAndRepeat(dim, repeats):
    """Create layer to expand dimension=`dim` and repeat dimension `repeats` time."""

    def expand(x):
        x = tf.expand_dims(x, dim)
        n_dims = len(x.shape)
        multiples = [1] * n_dims
        multiples[dim] = repeats
        return tf.tile(x, multiples)

    return kl.Lambda(expand, name="expand_and_repeat")


def build_pointnet(
    n_dims,
    n_classes,
    n_points=None,
    predict_first_n=None,
    predict_last_n=None,
    depth=1.0,
    mode="segmentation",
):
    """Build the PointNet model that accept `n_points` inputs and output
    `n_points * n_classes` softmax value.
    # Args
        n_dims: Number of input dimensions e.g. 3 for 3 dimensional points
        n_classes: How many classes to predict for each point.
        n_points: Number of input points and output points, can be None for
            variable number of points.
        predict_first_n: If specified, will predict only first N points.
            Used when the other points are for assistance only.
        predict_last_n: If specified, will predict only last N points.
            Used when the other points are for assistance only.
        depth: Multiplier to apply to the number of filters in each layer.
        mode: Either segmentation or classification. The currently allowed mode
            is "segmentation" only. Classification will be supported in the future.
    """
    if mode != "segmentation":
        raise ValueError("Currently support only 'segmentation' mode.")
    if predict_first_n and predict_last_n:
        raise ValueError("Please slice first XOR slice last only.")

    # the following operations mimic closely to figure 2 of the paper
    inp_pts = kr.Input(shape=(n_points, n_dims))
    input_tnet = build_tnet(
        n_dims, n_dims, n_points=n_points, depth=depth, name="input_tnet"
    )
    tmp = transform(inp_pts, input_tnet)  # input transform
    tmp = conv_stack([64, 64], depth=depth)(tmp)  # shared MLP
    tmp_dims = tmp.shape[-1]
    feature_tnet = build_tnet(
        tmp_dims, tmp_dims, n_points=n_points, depth=depth, name="feature_tnet"
    )
    feature = transform(tmp, feature_tnet, name="feature")  # feature transform

    # convert feature to global feature (can be for classifier later)
    tmp = conv_stack([64, 128, 1024], depth=depth)(feature)  # shared MLP
    global_feature = kl.GlobalMaxPooling1D()(tmp)

    # combine feature and global feature to make segmentation head
    if not predict_first_n and not predict_last_n:
        repeats = n_points if n_points else tf.shape(inp_pts)[1]
    elif predict_first_n:
        feature = feature[:, :predict_first_n, :]
        repeats = predict_first_n if n_points else tf.shape(feature)[1]
    elif predict_last_n:
        feature = feature[:, -predict_last_n:, :]
        repeats = predict_last_n if n_points else tf.shape(feature)[1]
    else:
        raise ValueError("Unexpected bug, both values are not true??")
    global_feature = ExpandAndRepeat(1, repeats)(global_feature)
    combined_feature = kl.concatenate(
        [feature, global_feature], name="combined_feature"
    )
    point_feature = conv_stack([512, 256, 128, 128], depth=depth)(combined_feature)
    out = kl.Conv1D(n_classes, 1, activation="softmax")(point_feature)
    model = kr.Model(inp_pts, out, name="pointnet")
    return model


if __name__ == "__main__":
    print(__doc__)
    model = build_pointnet(3, 4, n_points=10, depth=1.0)
    model.summary()
    pred = model.predict(np.random.random((1, 10, 3)))
    print("pred.shape:", pred.shape)
    print("pred[0]:\n", pred[0])
