# superkeras

A bunch of Keras utilities and paper implementations written under TensorFlow backend

It is made to work nicely with `tensorflow.keras` and TensorFlow 2.0.

## How to use

Download this repository and put it inside your project as a folder named `superkeras`.

One of the nice option is to do `git submodule add https://github.com/offchan42/superkeras` in order to download this repo
and track it as a part of your Git repository because this repo is frequently updated.

Then simply `import superkeras` to use all the functionalities provided. See more info below.

### General utilities/helpers/extras

- `import superkeras.layers` to use use functions and classes in the file. Contains some useful functions like:
  - `repeat_layers` for creating multiple layers with the same type
  - `apply_residual_block` function for building ResNet-like architecture, making the network able to learn without
    degradation when the depth is very deep
  - `BlurPool` antialiased layer to make ConvNet shift-invariant, also increase accuracy.
    See https://github.com/adobe/antialiased-cnns
  - `Arithmetic` layer for performing simple arithmetic operations on a trainable weight
  - `NormalizeQuaternion` layer for normalizing Quaternion data to have magnitude of 1.
  - and couple more utilities
- `import superkeras.losses` to use loss functions (which can also be used as metrics)
  - `r2_score` for computing r-squared score for regression problem.
  - `mean_euclidean_distance` or `mean_euclidean_distance_squared` for computing
    distance between 2 positions.
  - `mean_quat_distance`, `mean_quat_angle`, `mean_quat_angle_deg`, and
    `mean_sqr_quat_angle` are for computing Quaternion difference, must be used
    with normalized quaternion (output of `NormalizeQuaternion` layer).
  - `dice_loss`, `iou_coef`, and `dice_coef` are losses/metrics for image segmentation problems.
- `import superkeras.datautil` to use utilities made for `tf.data` module
- `import superkeras.utils` to use some helper functions not related to `keras` e.g. `make_xy_3d` for converting
  a time-series `DataFrame` into a 3D data for ConvNets or LSTM.
- `import superkeras.imgutils` to use helper functions for manipulating images e.g. cropping with zero padding.
  Please check `Rect` class which helps you a lot in bounding box related manipulation.

### Paper implementations

- `import superkeras.permutational_layer` to use `PermutationalLayer`
  model implemented accurately following the paper [Permutation-equivariant
  neural networks applied to dynamics
  prediction](https://arxiv.org/pdf/1612.04530.pdf). This layer is for modeling
  problems that the order of the input objects are not important and that you
  can swap/permute/re-order them and the prediction should stay the same,
  preserving **Permutation Invariance** property. You can think of this as
  modeling a `Set` data structure for the inputs of neural networks.

  To use it without the need to understand too much details,
  you can use `PermutationalModule`.

  You can run the main code in the module to understand how it works intuitively.

- `import superkeras.pointnet` to use `PointNet` architecture from the
  [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593) paper.
  PointNet also allows modeling of permutation invariance problems, it's computationally cheaper than `PermutationalLayer`
  for large amount of point instances.

  You can run the main code in the module to understand how it practically works.

## Tips

You can run `pytest` to run test files that have name starting with `test_`.

For any functions that are not mentioned here but exist in the file, you can use them.
All of the functions that are supposed to be usable usually have documentation written very good on them. So check that!

## Troubleshooting

- `ValueError: Unknown metric function:mean_quat_angle_deg`:

  This error is caused by not providing the function to the model loader.
  It usually happens when you save the Keras model to disk and trying to load it
  using `keras.models.load_model` function.

  To fix this, you need to provide `custom_objects` dictionary with string key
  pointing to the function reference.
  Example:

  ```python
  from superkeras.losses import mean_quat_angle_deg
  from keras.models import load_model
  model = load_model('model_path.h5', custom_objects=dict(mean_quat_angle_deg=mean_quat_angle_deg))
  ```

  There are many possible metric functions that this error can indicate.
  Most of the functions live in `losses` and `layers` module.
  So you must provide all of the unknown functions into `custom_objects`.

## FAQ

### How do I run the example main code in `permutational_layer.py` or `pointnet.py`?

You can run each file as a module of `superkeras` package by changing directory to be above `superkeras` and then run
`python -m superkeras.pointnet` for example. You cannot run `python pointnet.py` because this kind of run is not working
with relative imports.

### Why is this library named `superkeras`?

When I was searching for a unique name for my keras helper utilities like keras helpers, keras utils, keras extras, etc,
I've found that all of those names were already chosen by someone else. I don't want to repeat the name so I thought of
something cool and just name `superkeras` for ease in remembering.
