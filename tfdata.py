import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.data import Dataset
from collections import namedtuple

DatasetKit = namedtuple('DatasetKit', ['name', 'ds', 'n', 'dsb', 'batch_size', 'steps', 'shuffle'])
DSIZE = [60, 80 * 2]  # rows and columns A.K.A. height and width
print("DSIZE:", DSIZE)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, DSIZE)
    image /= 255  # normalize to [0,1] range
    return image  # return 3D tensor


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def create_image_label_dataset(image_paths, labels):
    """
    # Args
        labels: If it's None, only dataset with images will be returned
    """
    path_ds = Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
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