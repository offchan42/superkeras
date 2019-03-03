# superkeras

A bunch of Keras utilities and paper implementations written under TensorFlow backend

## How to use

Download this repository and put it inside your project as a folder named `superkeras`.

- `import superkeras.layers` to use use functions and classes in the file. Contains some useful functions like:
  - `repeat_layers` for creating multiple layers with the same type
  - `apply_residual_block` function for building ResNet-like architecture, making the network able to learn without
    degradation when the depth is very deep
  - `Arithmetic` layer for performing simple arithmetic operations on a trainable weight
  - `NormalizeQuaternion` layer for normalizing Quaternion data to have magnitude of 1.
  - and couple more utilities
- `import superkeras.losses` to use loss functions (which can also be used as metrics)
  - `mean_euclidean_distance` or `mean_euclidean_distance_squared` for computing
    distance between 2 positions.
  - `mean_quat_distance`, `mean_quat_angle`, `mean_quat_angle_deg`, and
    `mean_sqr_quat_angle` are for computing Quaternion difference, must be used
    with normalized quaternion (output of `NormalizeQuaternion` layer).
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

- Run `pytest` to run test files that have name starting with `test_`.
