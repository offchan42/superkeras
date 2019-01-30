# superkeras

A bunch of Keras utilities and paper implementations written under TensorFlow backend

## How to use

- `import superkeras.layers` to use use functions and classes in the file. Contains some useful functions like:
  - `repeat_layers` for creating multiple layers with the same type
  - `apply_residual_block` function for building ResNet-like architecture, making the network able to learn without
    degradation when the depth is very deep
  - `Arithmetic` layer for performing simple arithmetic operations on a trainable weight
  - and couple more utilities
- `import superkeras.permutational_layer` to use `PermutationalLayer`
  model implemented accurately following the paper [Permutation-equivariant
  neural networks applied to dynamics
  prediction](https://arxiv.org/pdf/1612.04530.pdf). This layer is for modeling
  problems that the order of the input objects are not important and that you
  can swap/permute/re-order them and the prediction should stay the same,
  preserving **Permutation Invariance** property. You can think of this as
  modeling a `Set` data structure for the inputs of neural networks.

- Run `pytest` to run test files that have name starting with `test_`.