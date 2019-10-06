"""
tf.data utilities
The most useful function that you can check is `create_xy_dataset_kit()`.
"""
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.data import Dataset
from collections import namedtuple


class DatasetKit(
    namedtuple(
        "DatasetKit",
        [
            "cache_path",
            "ds",
            "n",
            "dsb",
            "batch_size",
            "steps",
            "steps_float",
            "shuffle",
        ],
    )
):
    """
    DatasetKit is supposed to contain either train or test set, but not both.
    You can pass `dsb` to kr.Model.fit() along with `steps`.

    # Attributes
        cache_path: Path of the cache files
        ds: The dataset content. When iterated, give one sample at a time.
        n: The amount of iterations to complete one epoch of `ds`
        dsb: The dataset content (batched). When iterated, give a batch of samples at a time.
        batch_size: The batch size for `dsb`
        steps: The estimated amount of iterations to complete one epoch of `dsb`.
        steps_float: The exact amount of iterations to complete one epoch of `dsb`.
            If the value is close to `steps` it means precise evaluation will happen.
        shuffle: Whether the dataset was shuffled. Should be set to True for training set.
            But can be False for test set.
    """

    pass


def create_image_loader(
    channels=0,
    prep_func=None,
    width=None,
    height=None,
    resize_method=tf.image.ResizeMethod.AREA,
):
    """Create a function that receives an image file path and returns a 3D tf.Tensor
    (height, width, channels) representing an image.
    The output tensor will consist of values between 0 and 1.
    The image will be resized if `width` and `height` are provided.
    The image must have a JPEG or PNG format.
    See example code below.

    # Args
        channels: Number of color channels, e.g. 0 (automatically detected), 1, or 3
        prep_func: Preprocessing procedure to perform to the image before it is resized.
            The input to the function will be a 3D tf.Tensor with dtype=uint8.
            (because uint8 work flawlessly with OpenCV preprocessing functions)
            You must return the preprocessed uint8 image back to the caller.
            You need to convert tf.Tensor to a numpy array using `.numpy()`
            inside the function. (Eager execution must be enabled)
            See example below.
        width: Target width of the image (after resizing)
        height: Target height of the image (after resizing)
        resize_method: The interpolation method when resizing.
            default=tf.image.ResizeMethod.AREA (OpenCV said it's good for
            downsampling images)

    # Example
        >>> import cv2 as cv
        >>> def detect_edges(img):
        >>>     img = img.numpy()
        >>>     img = cv.Canny(img[...,0], 100, 200)
        >>>     img = img[...,None]
        >>>     return img
        >>> image_loader = datautil.create_image_loader(channels=1, prep_func=detect_edges, width=target_w, height=target_h)
        >>> img = image_loader('/path/to/image.jpg')  # a 3D tf.Tensor (target_h, target_w, 1)
        >>> img = img.numpy()[...,0]  # eagerly convert to numpy to plot
        >>> cv.imshow("img", img)
    """
    # if width and height are provided, we will resize the image
    hw_count = 0
    if width is not None:
        hw_count += 1
    if height is not None:
        hw_count += 1
    if hw_count == 1:
        raise ValueError("Please provide both `width` and `height` to resize image.")

    def load_and_preprocess_image(path):
        """Load image from `path` and return as 3D tf.Tensor"""
        image = tf.io.read_file(path)
        # cannot use decode_image() as it doesn't give shape information
        image = tf.cond(
            tf.image.is_jpeg(image),
            lambda: tf.image.decode_jpeg(image, channels=channels),
            lambda: tf.image.decode_png(image, channels=channels, dtype=tf.uint8),
        )
        if prep_func is not None:
            im_shape = image.shape
            [image] = tf.py_function(prep_func, [image], [tf.uint8])
            image.set_shape(im_shape)  # if we don't do this we will see unknown shape
        image = tf.image.convert_image_dtype(image, tf.float32)
        if hw_count == 2:
            image = tf.image.resize(image, (height, width))
        return image  # return 3D tensor

    return load_and_preprocess_image


def create_xy_dataset(xs, ys, xmap=None):
    """
    Create a zipped dataset with (x, y) pairs when iterated through.

    # Args
        xs: A list of x, e.g. a list of strings representing image paths.
        ys: A list of y. e.g. a list of image labels (numpy array).
            If it's None, only dataset with xs will be returned.
        xmap: A function to map to each x in xs, to obtain new xs list.
            E.g. an image loader function which reads image from a path. It can
            be created by calling `create_image_loader(...)`.

    # Returns
        The zipped dataset and its number of samples

    # Example
        Creating dataset of images and labels
        >>> paths = ['/path/img1.jpg', 'path/img2.jpg']
        >>> labels = [[0.1,0.2,0.3], [0.4,0.5,0.6]]
        >>> image_loader = datautil.create_image_loader(channels=3, width=100, height=200)
        >>> ds, n = datautil.create_xy_dataset(paths, labels, xmap=image_loader)
        >>> ds
        <DatasetV1Adapter shapes: ((200, 100, 3), (3,)), types: (tf.float32, tf.float32)>
        >>> assert n == 2
    """
    x_ds = Dataset.from_tensor_slices(xs)
    if xmap is not None:
        x_ds = x_ds.map(xmap, num_parallel_calls=AUTOTUNE)
    if ys is None:
        return x_ds
    y_ds = Dataset.from_tensor_slices(ys)
    zipped_ds = Dataset.zip((x_ds, y_ds))
    return zipped_ds, len(xs)


def create_xy_dataset_kit(xy_dataset, n_samples, cache_path, shuffle, batch_size):
    """
    Create an (x,y) DatasetKit instance (you can check its doc for how to use it)
    representing either a training set or test set, but not both at the same time.
    It's designed to work nicely with Keras model.fit().
    See the example code below.

    # Processing steps
        1. cache ds (maybe)
        2. dsb = ds
        3. shuffle dsb (maybe)
        4. repeat dsb (to work with keras.Model.fit(), it expects repeating dataset
            for both train and validation set)
        5. batch dsb
        6. prefetch dsb (to load next batch while the model is training)

    # Args
        xy_dataset: An instance of tf.data.Dataset created from `create_xy_dataset()`
        n_samples: Number of samples returned from `create_xy_dataset()`
        cache_path: Data cache file path e.g. "data/train". Can be None to not cache.
        shuffle: Whether to shuffle the dataset (Suggestion: shuffle for train, but not
            for test)
        batch_size: Batch size for batched dataset (used when training and inference)
            Choose `batch_size` that divides (or almost divides) `n_samples`
            will provide better evaluation measurement.

    # Example
        >>> ds, n = datautil.create_xy_dataset(...)
        >>> train_kit = datautil.create_xy_dataset_kit(ds, n, None, True, 32)
        >>> ds, n = datautil.create_xy_dataset(...)
        >>> test_kit = datautil.create_xy_dataset_kit(ds, n, None, False, 256)
        >>> model = keras.layers.Sequential(...)
        >>> model.fit(train_kit.dsb, steps_per_epoch=train_kit.steps, epochs=200, validation_data=test_kit.dsb, validation_steps=test_kit.steps)
        >>> loss, mae = model.evaluate(test_kit.ds.batch(256))

    # Troubleshooting
        If you see an error like `ValueError: Tensor's shape (x,) is not compatible
        with supplied shape (h, w, 1)` Make sure that you delete all cache files first.
        Error can happen easily if the cache files are broken. So make sure to
        delete them often.
    """
    ds = xy_dataset
    if cache_path is not None:
        ds = ds.cache(cache_path)
    dsb = ds
    n = n_samples
    if shuffle:
        dsb = dsb.shuffle(n)
    # keras requires that both training and validation set are repeating
    dsb = dsb.repeat().batch(batch_size).prefetch(AUTOTUNE)
    steps_f = n / batch_size
    steps = int(round(steps_f))
    if steps == 0:
        raise ValueError("steps == 0, please reduce `batch_size`!")
    return DatasetKit(cache_path, ds, n, dsb, batch_size, steps, steps_f, shuffle)
