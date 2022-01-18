"""
An implementation of the Permutational Layers as described by the paper
"Permutation-equivariant neural networks applied to dynamics prediction"
Link to the paper: https://arxiv.org/pdf/1612.04530.pdf

Accurate Implementation using Keras Functional API. Deep learning for human.
Supports multiple layers.

To get started right away, please check the bottom section and run this file.

In order to understand permutational layer, let us first understand the simplest
idea of incorporating permutation invariance property into neural networks first.
Suppose you have 3 inputs with permutation invariance property.
Create a model that sees only one input, and output an encoding.
Apply this same model (shared weight) to all the inputs so you get 3 encodings.
If you sum the encodings together, you are going to get encoding X.
If you permute the input order, you are still going to get the encoding X.
This is simple way to incorporate permutation invariance.

The problem is that the model sees only one input at a time. It can not consider
the relationship between the inputs. It does preserve invariance property but
it does not give good accuracy on the dataset because it cannot fit the data well.
If you try to make the model sees more than one input,
you are going to violate the invariance property if not done properly.

This paper presents a solution to make the model sees more than one input
and still preserve invariance property. The idea is called permutational layer.

A permutational layer is a layer that takes N inputs, and encode them individually
resulting in N outputs. if input[i] = input[j], then output[i] = output[j] also.
The encoder inside each layer compares all pairs of inputs.
The output of the layer preserves permutation invariance property like the input.
The benefit of permutational layer is that output[i] is influenced by all inputs.
This make the model relates the information between each input better.
As you add more permutational layers, the rate of input relationship grows exponentially.

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
    permutational module: The model that takes N objects, apply multiple permutational layers sequentially to the objects.
        The last permutational layer can use pooling function to reduce output to one tensor.

# Structure of the concepts from concrete to abstract
    properties -> object -> pairwise model -> permutational encoder -> permutational layer -> permutational module
"""
from inspect import signature

from tensorflow.keras.layers import Dense, Input, average, concatenate, maximum
from tensorflow.keras.models import Model
import numpy as np

from .layers import LayerStack, repeat_layers


def move_item(a_list, from_idx, to_idx):
    """Move item in the list in-place"""
    a_list.insert(to_idx, a_list.pop(from_idx))


def PairwiseModel(individual_shape, layers, concat_axis=-1, **kwargs):
    """
    Create a model that accepts 2 input tensors, each having shape `individual_shape`.
    The output of the model will be the tensor output from the last layer of `layers`.

    If `layers` is a list, the model will use it to create layers.
    Two tensor inputs will be created, concatenated, and sent to each layer one after another.

    If `layers` is a callable then two input tensors will be fed to it
    without concatenation, and the callable is expected to return a tensor output.

    # Arguments
        individual_shape: Shape of each input tensor without batch dimension
        layers: List of layers or a callable that takes 2 tensors and output a tensor.
        concat_axis: Concatenation axis of the 2 input tensors
            Will not be used if `layers` is callable.
        **kwargs: Arguments to send to Model()

    # Returns
        A pairwise model which takes 2 input tensors and returns one output tensor
    """
    x1 = Input(shape=individual_shape, name="x1")
    x2 = Input(shape=individual_shape, name="x2")
    x_pair = [x1, x2]
    if isinstance(layers, list):
        x_concat = concatenate(x_pair, axis=concat_axis)
        dense_stack = LayerStack(layers)
        output_features = dense_stack(x_concat)
    elif callable(layers):
        params = signature(layers).parameters
        n_params = len(params)
        if n_params != 2:
            raise ValueError(
                f"Detected `layers` as a callable that takes in {n_params} "
                "arguments as input. But this function requires a callable that takes "
                "in 2 input tensor as arguments."
            )
        output_features = layers(x1, x2)
    else:
        raise ValueError("`layers` must be a list or a callable.")
    return Model(x_pair, output_features, **kwargs)


def PermutationalEncoder(
    pairwise_model,
    n_inputs,
    main_index=0,
    pooling=maximum,
    encode_identical=True,
    **kwargs,
):
    """
    Create a model which takes a list of `n_inputs` tensors with identical shape
    and outputs a tensor representing the main input's encoding.
    Tensor shape of both input and output are inferred from `pairwise_model`.

    # Arguments
        pairwise_model: The model that takes 2 inputs tensor and return one input tensor
        n_inputs: Number of permutationally invariant input tensors with identical shape. The shape will be inferred from pairwise model.
        main_index: A number ranging from [0, n_inputs) to tell which input is the main one.
        pooling: A function to apply to the list of final encodings to merge them into one encoding tensor.
            If not set, the default maximum function will be used as it shows the best result in the paper.
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


def PermutationalModule(
    input_shape,
    n_inputs,
    layers_stack,
    concat_axis=-1,
    encoder_pooling=maximum,
    encode_identical=True,
    last_layer_pooling=None,
    summary=True,
    name="permutational_module",
    **kwargs,
):
    """
    Create a model that performs multiple PermutationalLayer after one another.
    The input to the model must be a list of `n_inputs` tensors.
    The output will also be a list of `n_inputs` tensor if `last_layer_pooling` is None.
    If `last_layer_pooling` is set, the output will be a single tensor.
    
    This function simplifies the entire process of using permutational layers
    as it makes the process more high level.
    PermutationalLayer, PermutationalEncoder, and PairwiseModel are automatically created for you.
    If you don't want to deal with too much detail on how to stack PermutationalLayer
    together, just use this function.
    # Arguments
        input_shape: Input shape of one individual input without batch dimension.
        n_inputs: How many inputs to setup PermutationalEncoder for
        layers_stack: A list of layers list or callables.
            layers_stack[i] = a list of layers or a callable that takes 2 input
            tensors and return an output tensor for pairwise model i.
            Read documentation of PairwiseModel for more information.
        concat_axis: Concatenation axis of the 2 input tensors inside PairwiseModel
            Will only be used for layers inside `layers_stack` that are not callable.
        encoder_pooling: The pooling function for permutational encoder to use.
            The default is maximum as it shows the best result in the paper.
        encode_identical: Whether to run PairwiseModel on duplicate pairs of inputs.
            E.g. Run PairwiseModel on input x1 and x1, x2 and x2, etc.
            See PermutationalEncoder doc for more details.
        last_layer_pooling: The pooling function for the last permutational layer.
            We only allow last layer pooling because applying pooling to
            intermediate permutational layers do not make sense as it collapses
            `n_inputs` to 1 for subsequent permutational layers.
        summary: Whether or not to print summary
    
    # Returns
        A model that consumes `n_inputs` tensors and output `n_inputs` tensors
        or output 1 tensor if `last_layer_pooling` is not None.
    
    # Example
        Create a module with 2 permutational layers, the last layer is pooled using max function.
        Each permutational layer consists of a pairwise model that has 2 Dense layers.
        >>> PermutationalModule((4,), 3, [repeat_layers(Dense, [2, 4]), repeat_layers(Dense, [8, 16])], last_layer_pooling=maximum)
        permutational_module
        __________________________________________________________________________________________________
        Layer (type)                    Output Shape         Param #     Connected to                     
        ==================================================================================================
        x1 (InputLayer)                 (None, 4)            0                                            
        __________________________________________________________________________________________________
        x2 (InputLayer)                 (None, 4)            0                                            
        __________________________________________________________________________________________________
        x3 (InputLayer)                 (None, 4)            0                                            
        __________________________________________________________________________________________________
        permutational_layer_1 (Model)   [(None, 4), (None, 4 30          x1[0][0]                         
                                                                        x2[0][0]                         
                                                                        x3[0][0]                         
        __________________________________________________________________________________________________
        permutational_layer_2 (Model)   (None, 16)           216         permutational_layer_1[1][0]      
                                                                        permutational_layer_1[1][1]      
                                                                        permutational_layer_1[1][2]      
        ==================================================================================================
        Total params: 246
        Trainable params: 246
        Non-trainable params: 0
        __________________________________________________________________________________________________
        <keras.engine.training.Model at 0x2e2089774a8>
    """
    inputs = [Input(shape=input_shape, name=f"x{i+1}") for i in range(n_inputs)]
    outputs = inputs
    n_perm_layers = len(layers_stack)
    for i, layers in enumerate(layers_stack):
        # figure out whether to use pooling or not
        is_last_layer = n_perm_layers - 1 == i
        pooling = last_layer_pooling if is_last_layer else None

        perm_layer = PermutationalLayer(
            PermutationalEncoder(
                PairwiseModel(input_shape, layers, name=f"pairwise_model_{i+1}"),
                n_inputs,
                pooling=encoder_pooling,
                encode_identical=encode_identical,
                name=f"permutational_encoder_{i+1}",
            ),
            pooling=pooling,
            name=f"permutational_layer_{i+1}",
        )

        outputs = perm_layer(outputs)
        if not is_last_layer:
            input_shape = perm_layer.output_shape[0][1:]
    model = Model(inputs, outputs, name=name, **kwargs)
    if summary:
        print(model.name)
        model.summary()
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
    print("# Input Data")
    print(x)
    x = [np.array(e) for e in x]
    print("# Output Features")
    predictions = perm_layer.predict(x)
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
    a = Input(shape=(4,), name="a")
    b = Input(shape=(4,), name="b")
    c = Input(shape=(4,), name="c")
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
    x = [np.array(e) for e in x]
    print(model.predict(x))

    print("## Input/Output 2")
    x = [[[1, 2, 3, 4]], [[0, 0, 0, 0]], [[4, 3, 2, 1]]]
    print(x)
    x = [np.array(e) for e in x]
    print(model.predict(x))

    print("## Input/Output 3")
    x = [[[4, 3, 2, 1]], [[0, 0, 0, 0]], [[1, 2, 3, 4]]]
    print(x)
    x = [np.array(e) for e in x]
    print(model.predict(x))
    print()
    print("Isn't this cool !?")
    print()
    print("After you understand the internal working of permutational layers,")
    print("you can use PermutationalModule() as a high-level function to construct")
    print("all the layer components quicker. Read its docstring to see how to use.")
