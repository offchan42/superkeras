"""
An implementation of the Permutational Layers as described by the paper
"Permutation-equivariant neural networks applied to dynamics prediction"
Link to the paper: https://arxiv.org/pdf/1612.04530.pdf

Accurate Implementation using Keras Functional API. Deep learning for human.
Supports multiple layers.

# Concepts introduced in this code (and some from the paper)
    properties: A vector representing an object. E.g. 4 numbers representing position XY, and velocity XY.
    object: The input to the system. 2 objects when swapped should not change the meaning of the network's predictions
        E.g. If you have 2 particles, each giving you 4 numbers (positionX, positionY, velocityX, velocityY),
        the object in this case is the particle. So you have 2 objects in the system.
    pairwise model: The model that compares 2 objects and make an encoding out of it.
    permutational encoder: The model that takes N objects, apply pairwise model N times, merge the encodings and return it.
        It needs to know the order of the object that is the main one, the output will represent the main object's encoding.
    permutational layer: The model that takes N objects, apply permutational encoder to each of the object,
        resulting in N distinct encodings. These encodings are still preserving the permutational-invariance property.
        It means that you can apply many permutational layers on top of each other.

# Structure of the concepts from concrete to abstract
    properties -> object -> pairwise model -> permutational encoder -> permutational layer
"""
from kerashelper import repeat_layers, LayerStack
from keras.layers import Input, concatenate, Dense, average
from keras.models import Model


def move_item(a_list, from_idx, to_idx):
    """Move item in the list in-place"""
    a_list.insert(to_idx, a_list.pop(from_idx))


def PairwiseModel(individual_shape, sequential_layers, **kwargs):
    """
    Create a model that accepts 2 input tensors, each having shape `individual_shape`.
    The input tensors are concatenated and fed to each layer inside `sequential_layers`.
    The output of the model will be the tensor output from the last layer of `sequential_layers`.

    # Returns
        A pairwise model which takes 2 input tensors and returns one output tensor
    """
    x1 = Input(individual_shape, name="x1")
    x2 = Input(individual_shape, name="x2")
    x_pair = [x1, x2]
    x_concat = concatenate(x_pair)
    dense_stack = LayerStack(sequential_layers)
    output_features = dense_stack(x_concat)
    return Model(x_pair, output_features, **kwargs)


def PermutationalEncoder(
    pairwise_model,
    n_inputs,
    main_index=0,
    pooling=average,
    encode_identical=True,
    **kwargs,
):
    """
    Create a model which takes a list of `n_input` tensors with identical shape
    and outputs a tensor representing the main input's encoding.
    Tensor shape of both input and output are inferred from `pairwise_model`.

    # Arguments
        pairwise_model: The model that takes 2 inputs tensor and return one input tensor
        n_inputs: Number of permutationally invariant input tensors with identical shape. The shape will be inferred from pairwise model.
        main_index: A number ranging from [0, n_inputs) to tell which input is the main one.
        pooling: A function to apply to the list of final encodings to merge them into one encoding tensor.
            If not set, the default average function will be used like in the paper.
        encode_identical: Whether to run pairwise model on the main input paired with main input or not.
            The paper sets this to True. They run pairwise model also on the same inputs.

    # Returns
        The permutational encoder model that takes a list of input tensors and returns an encoding tensor for the main input.

    """
    assert n_inputs >= 2
    assert 0 <= main_index < n_inputs

    batch_shape = pairwise_model.input_shape[0]
    inputs = [
        Input(
            batch_shape=batch_shape, name=f"x{i+1}{'_main' if i == main_index else ''}"
        )
        for i in range(n_inputs)
    ]
    main_input = inputs[main_index]
    encodings = []
    for other_index in range(n_inputs):
        if not encode_identical and main_index == other_index:
            continue
        other_input = inputs[other_index]
        encoding = pairwise_model([main_input, other_input])
        encodings.append(encoding)
    model = Model(inputs, pooling(encodings), **kwargs)
    model.main_index = main_index
    return model


def PermutationalLayer(permutational_encoder, pooling=None, **kwargs):
    """
    A model that takes in a list of N input tensors with identical shape,
    permutate the list `N` times, apply `permutational_encoder` to each permutation,
    and return a list of N output tensors from the encoder with identical shape.
    The list length and shape of the input and output is inferred from the `permutational_encoder` provided.
    Each output tensor is preserving permutational invariance property.

    If `pooling` function is set, it will be applied to the list of output tensors from `permutational_encoder`.
    Then a single tensor output will be returned from the model, instead of a list of N tensors.

    You can stack this layer on top of each other like it's a Dense layer.
    To obtain a single tensor output from this layer, just provide a pooling function
    to be applied to the list of outputs (e.g. average, sum, max, or even concatenate).
    Or you can apply the pooling function on the outputs of the model yourself.
    One of the suggested pooling function is `max` because it shows good result in the paper.

    Note: Even though it's named a layer, under the hood it's actually a Model instance.
    """
    inputs = [
        Input(batch_shape=batch_shape, name=f"x{i+1}")
        for i, batch_shape in enumerate(permutational_encoder.input_shape)
    ]
    n_inputs = len(inputs)
    encodings = []
    for main_index in range(n_inputs):
        inputs_permuted = [x for x in inputs]  # copy the list for permutation
        move_item(inputs_permuted, main_index, permutational_encoder.main_index)
        encodings.append(permutational_encoder(inputs_permuted))
    output = encodings
    if pooling:
        output = pooling(output)
    model = Model(inputs, output, **kwargs)
    return model


# test creating the network similar to the one mentioned in the paper
# look at this section for example usage and run this file to understand how it works
if __name__ == "__main__":
    pairwise_model = PairwiseModel(
        (4,), repeat_layers(Dense, [32, 8], name="hidden"), name="pairwise_model"
    )
    print("# PairwiseModel summary")
    pairwise_model.summary()

    perm_encoder = PermutationalEncoder(pairwise_model, 3, name="permutational_encoder")
    print("# PermutationalEncoder summary")
    perm_encoder.summary()

    perm_layer = PermutationalLayer(perm_encoder, name="permutational_layer")
    print("# PermutationalLayer summary")
    perm_layer.summary()

    x = [[[1, 2, 3, 4]], [[4, 3, 2, 1]], [[1, 2, 3, 4]]]
    predictions = perm_layer.predict(x)
    print("# Input Data")
    print(x)
    print("# Output Features")
    for pred in predictions:
        print(pred)
    print(
        "You will notice that output1 = output2 if input1 = input2. So the permutational layer is simply performing a function f(x) for each input x."
    )
    print("It is similar to a shared model applied to each input.")
    print(
        "The difference that makes this approach better is because f(x) is not standalone like shared model. It sees useful relationship between inputs."
    )

    ## now let's make a model with multi permutational layers to show you that it's possible!
    a = Input((4,), name="a")
    b = Input((4,), name="b")
    c = Input((4,), name="c")
    inputs = [a, b, c]
    outputs = perm_layer(inputs)
    perm_layer2 = PermutationalLayer(
        PermutationalEncoder(
            PairwiseModel((8,), repeat_layers(Dense, [16, 4], activation="relu")), 3
        ),
        name="permutational_layer2",
    )
    outputs = perm_layer2(outputs)
    perm_layer3 = PermutationalLayer(
        PermutationalEncoder(
            PairwiseModel((4,), repeat_layers(Dense, [16, 4], activation="tanh")), 3
        ),
        name="permutational_layer3",
    )
    outputs = perm_layer3(outputs)
    # I can even reuse this layer because the input shape and output shape of it is identical. but don't try this at home unless you know what you are doing.
    outputs = perm_layer3(outputs)
    output = average(outputs)  # let's average the output for single tensor
    model = Model(inputs, output)
    print("# Multi-layer model summary")
    model.summary()
    # let's predict with the big mult layers model on some similar data
    print("# Multi-layer Prediction")
    print(
        "Because the layer is preserving permutation invariance. "
        "The output will be the same no matter how many times you permute the input order."
    )
    print("## Input/Output 1")
    x = [[[0, 0, 0, 0]], [[1, 2, 3, 4]], [[4, 3, 2, 1]]]
    print(x)
    print(model.predict(x))

    print("## Input/Output 2")
    x = [[[1, 2, 3, 4]], [[0, 0, 0, 0]], [[4, 3, 2, 1]]]
    print(x)
    print(model.predict(x))

    print("## Input/Output 3")
    x = [[[4, 3, 2, 1]], [[0, 0, 0, 0]], [[1, 2, 3, 4]]]
    print(x)
    print(model.predict(x))
    print()
    print("Isn't this cool !?")
