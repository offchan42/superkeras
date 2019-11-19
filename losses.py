import tensorflow as tf
from tensorflow.keras import backend as K


def iou_coef(y_true, y_pred, smooth=1):
    """
    Intersection-over-Union coefficient from 0 to 1.

    Can be used to evaluate image segmentation problems where the output of the
    model are images or segmentation maps.
    `y_true` and `y_pred` must have shape (num_images, height, width, classes)

    See: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    """Dice coefficient is the `2 * overlapped_area / total_area`

    Can be used to evaluate image segmentation problems where the output of the
    model are images or segmentation maps.
    `y_true` and `y_pred` must have shape (num_images, height, width, classes)

    See: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)
    return dice


def dice_loss(y_true, y_pred):
    """Loss for image segmentation problems. It is usually combined with binary
    cross entropy for robustness.

    See: https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
    """
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return tf.reshape(1 - numerator / denominator, (-1, 1, 1))


def r2_score(y_true, y_pred):
    """R-squared score from 0 to 1. Can be used for evaluating regression problem."""
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


# region Euclidean Distance
def euclidean_distance_squared(
    y_true, y_pred, axis=-1, keepdims=False, force_positive=True
):
    """Compute the distance squared along the specified axis.
    # Example
        If each input has shape (10, 3), and axis=-1, the output will have shape (10,).
        Each element in the output represent euclidean distance ** 2.
    """
    sum_square = K.sum(K.square(y_true - y_pred), axis=axis, keepdims=keepdims)
    if force_positive:
        sum_square = K.maximum(sum_square, K.epsilon())
    return sum_square


def euclidean_distance(y_true, y_pred, axis=-1, keepdims=False, force_positive=True):
    """Compute the distance along the specified axis.
    # Example
        If each input has shape (10, 3), and axis=-1, the output will have shape (10,).
        Each element in the output represent euclidean distance.
    """
    return K.sqrt(
        euclidean_distance_squared(
            y_true, y_pred, axis=axis, keepdims=keepdims, force_positive=force_positive
        )
    )


def mean_euclidean_distance_squared(
    y_true, y_pred, axis=-1, keepdims=False, force_positive=True
):
    return K.mean(
        euclidean_distance_squared(
            y_true, y_pred, axis=axis, keepdims=keepdims, force_positive=force_positive
        )
    )


class mean_euclidean_distance_squared_metric:
    """Return keras metric that computes mean euclidean distance squared along specified axis."""

    name = "mean_euclidean_distance_squared"
    __name__ = "mean_euclidean_distance_squared"

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, y_true, y_pred):
        return mean_euclidean_distance_squared(
            y_true, y_pred, axis=self.axis, keepdims=False, force_positive=True
        )


def mean_euclidean_distance(
    y_true, y_pred, axis=-1, keepdims=False, force_positive=True
):
    return K.mean(
        euclidean_distance(
            y_true, y_pred, axis=axis, keepdims=keepdims, force_positive=force_positive
        )
    )


class mean_euclidean_distance_metric:
    """Return keras metric that computes mean euclidean distance along specified axis."""

    name = "mean_euclidean_distance"
    __name__ = "mean_euclidean_distance"

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, y_true, y_pred):
        return mean_euclidean_distance(
            y_true, y_pred, axis=self.axis, keepdims=False, force_positive=True
        )


# endregion

# region Quaternion Distance and Angle
def get_quat_distance(q1, q2, K):
    """Get quaternion distance from 0 to 1. The input must be normalized."""
    return 1 - K.sum(q1 * q2, axis=-1) ** 2


def mean_quat_distance(y_true, y_pred):
    """Quaternion distance from 0 to 1 loss/metric for keras.
    Make sure that the quaternion are normalized."""
    return K.mean(get_quat_distance(y_true, y_pred, K))


def get_quat_angle(q1, q2):
    """Get quaternion angle from 0 to pi. The input must be normalized."""
    import numpy as np

    eps = np.finfo(float).eps
    return np.arccos(np.clip(2 * np.sum(q1 * q2, axis=-1) ** 2 - 1, -1 + eps, 1 - eps))


def get_quat_angle_tf(q1, q2):
    """Get quaternion angle from 0 to pi. The input must be normalized."""
    acos_input = 2 * K.sum(q1 * q2, axis=-1) ** 2 - 1
    acos_input = K.clip(
        acos_input, -1 + K.epsilon(), 1 - K.epsilon()
    )  # prevent acos from returning nan
    return tf.acos(acos_input)


def mean_quat_angle(y_true, y_pred):
    """Quaternion angle loss/metric for keras in radians.
    Make sure that the quaternion are normalized. Otherwise you might get nan."""
    return K.mean(get_quat_angle_tf(y_true, y_pred))


def mean_quat_angle_deg(y_true, y_pred):
    """Quaternion angle loss/metric for keras in degrees.
    Make sure that the quaternion are normalized. Otherwise you might get nan."""
    import numpy as np

    return mean_quat_angle(y_true, y_pred) * 180 / np.pi


def mean_sqr_quat_angle(y_true, y_pred):
    """Get mean square quaternion angle.
    Make sure that the quaternion are normalized."""
    return K.mean(get_quat_angle_tf(y_true, y_pred) ** 2)


# endregion
