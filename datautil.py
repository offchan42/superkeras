import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.data import Dataset
from collections import namedtuple


class DatasetKit(namedtuple('DatasetKit', ['name', 'ds', 'n', 'dsb', 'batch_size', 'steps', 'shuffle'])):
    """
    DatasetKit is supposed to contain [train data XOR test data], not both at the same time.
    
    # Attributes
        name: name of the dataset, used when naming cache files
        ds: The dataset content. When iterated, give one sample at a time.
        n: The amount of iterations to complete one epoch of `ds`
        dsb: The dataset content (batched). When iterated, give a batch of samples at a time.
        batch_size: The batch size for `dsb`
        steps: The amount of iterations to complete one epoch of `dsb`.
        shuffle: Whether the dataset was shuffled. Should be set to True for training set.
            But can be False for test set.
    """
    pass


def create_image_loader(channels=None, width=None, height=None,
                    resize_method=tf.image.ResizeMethod.AREA):
    """Create a function that receives an image file path and returns a 3D tf.Tensor
    (height, width, channels) representing an image.
    The output tensor will consist of values between 0 and 1.
    The image will be resized if `width` and `height` are provided.
    
    # Args
        channels: Number of color channels, e.g. 0, 1, or 3
        width: Target width of the image (after resizing)
        height: Target height of the image (after resizing)
        resize_method: The interpolation method when resizing
            (default=tf.image.ResizeMethod.AREA)
    """
    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=channels, dtype=tf.float32)

        # if width and height are provided, we will resize the image
        hw_count = 0
        if width is not None: hw_count += 1
        if height is not None: hw_count += 1
        elif hw_count == 1:
            raise ValueError("Please provide both `width` and `height` to resize image.")
        elif hw_count == 2:
            image = tf.image.resize(image, (height, width))

        return image  # return 3D tensor
    return load_and_preprocess_image


def create_image_label_dataset(image_paths, labels, image_loader):
    """
    Create a dataset with (image, label) pairs when iterated through.
    # Args
        labels: If it's None, only dataset with images will be returned
    """
    path_ds = Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(image_loader, num_parallel_calls=AUTOTUNE)
    if labels is None:
        return image_ds
    label_ds = Dataset.from_tensor_slices(labels)
    image_label_ds = Dataset.zip((image_ds, label_ds))
    return image_label_ds


def create_image_label_dataset_kit(name, image_paths, labels, shuffle, batch_size, drop_remainder=True):
    """
    If you see an error like `ValueError: Tensor's shape (52,) is not compatible with supplied shape (80, 80, 1)`
    Make sure that you delete the cache file first.
    # Args
        name: Data cache file name
        labels: If it's None, only dataset with images will be returned
    """
    ds = create_image_label_dataset(image_paths, labels).cache('data/' + name)
    dsb = ds
    n = len(image_paths)
    if shuffle:
        dsb = dsb.shuffle(n)
    dsb = dsb.repeat().batch(batch_size, drop_remainder=drop_remainder).prefetch(AUTOTUNE)
    return DatasetKit(name, ds, n, dsb, batch_size, n // batch_size, shuffle)