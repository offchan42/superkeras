import pytest
from .layers import *
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Sequential


def test_repeat_layers():
    layers = repeat_layers(Conv2D, [32, 64], [(3, 7), 5])
    assert [layer.kernel_size for layer in layers] == [(3, 7), (5, 5)]

    # should raise because there is no positional arguments
    with pytest.raises(TypeError):
        repeat_layers(Dense)

    assert isinstance(repeat_layers(Flatten)[0], Flatten)


def test_repeat_layers_name_indexing():
    layers = repeat_layers(Dense, [10, 20, 30], name="hello")
    assert layers[-1].name == "hello_3"


def test_repeat_layers_input_shape():
    layers = repeat_layers(Dense, [10, 20, 30], input_shape=(1, 2, 3))
    seq = Sequential(layers)
    assert seq.input_shape == (None, 1, 2, 3)
    assert seq.output_shape == (None, 1, 2, 30)


def test_repeat_layers_no_input_shape():
    layers = repeat_layers(Dense, [10])
    seq = Sequential(layers)
    # should raise because there is no input_shape
    with pytest.raises(AttributeError):
        seq.input_shape

    # should raise because there is no input_shape and the model has not been built
    with pytest.raises(ValueError):
        seq.summary()


def test_layer_stack():
    layers = repeat_layers(Dense, [10, 20], input_shape=(5,))
    stack = LayerStack(layers)
    assert stack.layers == LayerStack(Sequential(layers)).layers
    a = Input((5,))
    b = stack(a)
    assert len(stack.layers) == 2
    assert b.shape.as_list()[-1] == 20
