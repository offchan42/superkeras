import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import constraints, initializers
from tensorflow.keras.layers import Activation, Lambda, Layer, add


class BlurPool(Layer):
    """
    BlurPool allows you to antialias your model architecture, making convolutional
    networks shift-invariant again!
    The methodology is simple. First, apply convolution layer with strides=1,
    then use this BlurPool layer to do antialised downsampling.

    You can replace AveragePooling or MaxPooling with the following guideline.
    1. Max Pooling: MaxPool(strides=2) => [MaxPool(strides=1), BlurPool(strides=2)]
    2. Strided-Convolution: Conv(strides=2, 'relu') => [Conv(strides=1, 'relu'), BlurPool(strides=2)]
    3. Average Pooling: AvgPool(strides=2) => BlurPool(strides=2)

    # Benefits
        The network's accuracy will increase and the prediction probability won't
        fluctuate much when the object in the image is slightly moved.

    # See also
        https://arxiv.org/abs/1904.11486
        https://github.com/adobe/antialiased-cnns
        https://github.com/adobe/antialiased-cnns/issues/10
    """

    def __init__(self, kernel_size, strides=2, **kwargs):
        self.strides = (strides, strides)
        self.kernel_size = kernel_size
        self.padding = (
            (
                int(1.0 * (kernel_size - 1) / 2),
                int(np.ceil(1.0 * (kernel_size - 1) / 2)),
            ),
            (
                int(1.0 * (kernel_size - 1) / 2),
                int(np.ceil(1.0 * (kernel_size - 1) / 2)),
            ),
        )
        if self.kernel_size == 1:
            self.a = np.array([1.0])
        elif self.kernel_size == 2:
            self.a = np.array([1.0, 1.0])
        elif self.kernel_size == 3:
            self.a = np.array([1.0, 2.0, 1.0])
        elif self.kernel_size == 4:
            self.a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.kernel_size == 5:
            self.a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.kernel_size == 6:
            self.a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.kernel_size == 7:
            self.a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
        super(BlurPool, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        height = input_shape[1] // self.strides[0]
        width = input_shape[2] // self.strides[1]
        channels = input_shape[3]
        return (input_shape[0], height, width, channels)

    def call(self, x):
        k = self.a
        k = k[:, None] * k[None, :]
        k = k / np.sum(k)
        k = np.tile(k[:, :, None, None], (1, 1, K.int_shape(x)[-1], 1))
        k = K.constant(k, dtype=K.floatx())

        x = K.spatial_2d_padding(x, padding=self.padding)
        x = tf.nn.depthwise_conv2d(
            x, k, strides=[1, self.strides[0], self.strides[1], 1], padding="VALID"
        )
        return x


class LayerStack:
    """
    Represent sequential keras layers. Used in place of Sequential for less
    prediction latency when attempting to connect multiple Sequential together.

    An initialized LayerStack should be called with a tensor argument and get a tensor
    output similar to Sequential.

    # Example
        >>> stack = LayerStack([Dense(10), Dense(20)])
        >>> stack(input_tensor)
        <tf.Tensor 'dense_2/BiasAdd:0' shape=(?, 20) dtype=float32>

        Or in the case you have keras Model instance (encoder):
        >>> stack = LayerStack(encoder)
        >>> stack(input_tensor)
        <tf.Tensor 'hidden_5_7/BiasAdd:0' shape=(?, 50) dtype=float32>
    """

    def __init__(self, keras_layers, name=None):
        """
        # Args
            keras_layers: A list of layers or a keras Model instance
            name: The default name that will be used for the output tensor
        """
        try:
            # if the input is a keras Model instance then this would work
            keras_layers = keras_layers.layers
            self.name = keras_layers.name
        except Exception:
            pass
        if not isinstance(keras_layers, list):
            raise ValueError("`keras_layers` must be a list or a keras Model instance.")
        self.layers = keras_layers
        self.name = name

    def __call__(self, tensors, name=None):
        """Call and return the tensor with the given name.
        If given name is None, the default name will be used (if it exists)
        """
        out = call_layers(self.layers, tensors)
        if name is None:
            name = self.name
        if name:
            out = rename_tensor(out, name)
        return out


def repeat_layers(layer_class, *args, name=None, name_start_index=1, **kwargs):
    """
    Instantiate layer instances from `layer_class` and return them as a list.

    The number of layers is inferred from the length of the first positional argument.
    Each positional argument must be a list. Each element inside the list will be fed to one layer instance.
    All layer instance shares the same keyword arguments.

    # Arguments
        args: Arguments must be list of positional arguments to feed to layer_class()

    # Keyword Arguments
        input_shape: Will be fed only to the first layer. This allows you to call Sequential() on the output of this function.
        name: Will be appended with suffix index to differentiate each layer from one another.
        name_start_index: If you define the name, this value will determine the starting suffix index.

    # Example
        Create a list of 2 Dense layers, with 10 units, and 20 units consecutively.
        >>> repeat_layers(Dense, [10, 20], activation='relu')
        [<keras.layers.core.Dense at 0x1d5054b5e48>,
         <keras.layers.core.Dense at 0x1d5054b5f60>]

        Create a list of 2 Conv2D layers, and checking its kernel_size.
        >>> [layer.kernel_size for layer in repeat_layers(Conv2D, [32, 64], [(3, 7), 5])]
        [(3, 7), (5, 5)]

        Create a list of 2 LSTM layers with input_shape, then feed it to Sequential() model.
        >>> layers = repeat_layers(LSTM, [1, 2], return_sequences=True, input_shape=(3, 4), name='rnn')
        >>> Sequential(layers).summary()
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        rnn_1 (LSTM)                 (None, 3, 1)              24        
        _________________________________________________________________
        rnn_2 (LSTM)                 (None, 3, 2)              32        
        =================================================================
        Total params: 56
        Trainable params: 56
        Non-trainable params: 0
        _________________________________________________________________

    # Returns
        A list of layer instances
    """
    layers = []
    if not args:
        return [layer_class(name=name, **kwargs)]
    for i in range(len(args[0])):
        arg = []
        for j in range(len(args)):
            arg.append(args[j][i])
        if name:
            kwargs["name"] = name + "_" + str(name_start_index + i)
        layers.append(layer_class(*arg, **kwargs))
        kwargs.pop("input_shape", None)  # remove input_shape for later layers
    return layers


def call_layers(layers, tensor):
    """
    Pass `tensor` through each layer sequentially until it reaches the last layer.

    The output tensor of the last layer will be returned.
    
    This function is useful when you don't want to create a Sequential() model just to call the layers.
    One usage is for inspecting the summary() of the model which has many nested Sequential() model inside.
    Usually, if you create an inner Sequential() model, and you use it on the outer model, when you call
    summary() on the outer model, you will not see the inside of the Sequential() model.
    This function can help you expand the layers of the Sequential() model so that you can see all layers
    under the nested models.
    To see what I mean, please see the example below.
    
    # Arguments
        layers: A list of keras layers. If it's not a list, it's assumed to be
            one layer.
        tensor: Input tensor to feed to the first layer
    
    # Returns
        Output tensor from the last layer.
        
    # Example
        Create some Dense layers and call them on an input tensor.
        >>> a = Input(shape=(10,), name='input')
        >>> dense_stack = repeat_layers(Dense, [16, 32, 64], name='hidden')
        >>> b = call_layers(dense_stack, a)
        >>> b
        <tf.Tensor 'hidden_3_1/BiasAdd:0' shape=(?, 64) dtype=float32>
        >>> Model(a, b).summary()
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        input (InputLayer)           (None, 10)                0         
        _________________________________________________________________
        hidden_1 (Dense)             (None, 16)                176       
        _________________________________________________________________
        hidden_2 (Dense)             (None, 32)                544       
        _________________________________________________________________
        hidden_3 (Dense)             (None, 64)                2112      
        =================================================================
        Total params: 2,832
        Trainable params: 2,832
        Non-trainable params: 0
        _________________________________________________________________
        

        Suppose we have an encoder model in the form of Sequential() like this:
        >>> dense_stack = repeat_layers(Dense, [10, 20, 30, 40, 50], activation='relu', name='hidden', input_shape=(10,))
        >>> encoder = Sequential(dense_stack)
        
        And we also have a bigger model which uses the encoder twice on 2 inputs:
        >>> a = Input(shape=(10,))
        >>> b = Input(shape=(10,))
        
        We encode the inputs and concatenate the them. And then we create a output layer.
        We then create a model out of it.
        >>> encoding = concatenate([encoder(a), encoder(b)])
        >>> out = Dense(5)(encoding)
        >>> big_model = Model(inputs=[a, b], output=out)
        
        Let us check the summary of the model to see what it is like.
        >>> big_model.summary()
        __________________________________________________________________________________________________
        Layer (type)                    Output Shape         Param #     Connected to                     
        ==================================================================================================
        input_14 (InputLayer)           (None, 10)           0                                            
        __________________________________________________________________________________________________
        input_15 (InputLayer)           (None, 10)           0                                            
        __________________________________________________________________________________________________
        sequential_32 (Sequential)      (None, 50)           4250        input_14[0][0]                   
                                                                         input_15[0][0]                   
        __________________________________________________________________________________________________
        concatenate_2 (Concatenate)     (None, 100)          0           sequential_32[3][0]              
                                                                         sequential_32[4][0]              
        __________________________________________________________________________________________________
        dense_6 (Dense)                 (None, 5)            505         concatenate_2[0][0]              
        ==================================================================================================
        Total params: 4,755
        Trainable params: 4,755
        Non-trainable params: 0
        
        You see that the Sequential model hides all the detail. It only shows the parameter counts but it doesn't show its internal layers.
        To make it shows the internal layers, we can use `call_layers(encoder.layers, a)` instead of `encoder(a)` to expand the encoder like this:
        >>> encoding = concatenate([call_layers(encoder.layers, a), call_layers(encoder.layers, b)])
        >>> out = Dense(5)(encoding)
        >>> big_model = Model(inputs=[a, b], output=out)
        >>> big_model.summary()
        __________________________________________________________________________________________________
        Layer (type)                    Output Shape         Param #     Connected to                     
        ==================================================================================================
        input_14 (InputLayer)           (None, 10)           0                                            
        __________________________________________________________________________________________________
        input_15 (InputLayer)           (None, 10)           0                                            
        __________________________________________________________________________________________________
        hidden_1 (Dense)                (None, 10)           110         input_14[0][0]                   
                                                                         input_15[0][0]                   
        __________________________________________________________________________________________________
        hidden_2 (Dense)                (None, 20)           220         hidden_1[2][0]                   
                                                                         hidden_1[3][0]                   
        __________________________________________________________________________________________________
        hidden_3 (Dense)                (None, 30)           630         hidden_2[2][0]                   
                                                                         hidden_2[3][0]                   
        __________________________________________________________________________________________________
        hidden_4 (Dense)                (None, 40)           1240        hidden_3[2][0]                   
                                                                         hidden_3[3][0]                   
        __________________________________________________________________________________________________
        hidden_5 (Dense)                (None, 50)           2050        hidden_4[2][0]                   
                                                                         hidden_4[3][0]                   
        __________________________________________________________________________________________________
        concatenate_4 (Concatenate)     (None, 100)          0           hidden_5[2][0]                   
                                                                         hidden_5[3][0]                   
        __________________________________________________________________________________________________
        dense_8 (Dense)                 (None, 5)            505         concatenate_4[0][0]              
        ==================================================================================================
        Total params: 4,755
        Trainable params: 4,755
        Non-trainable params: 0
        __________________________________________________________________________________________________
        
        Now, you see the detail of each internal layer making up the encoder with one summary() call!

    """
    if not isinstance(layers, (tuple, list)):
        layers = [layers]
    for layer in layers:
        tensor = layer(tensor)
    return tensor


def rename_tensor(tensor, name, **kwargs):
    """Create an identity Lambda layer and call it on `tensor`, mainly to rename it."""
    return Lambda(lambda x: x, name=name, **kwargs)(tensor)


class Arithmetic(Layer):
    """
    Perform arithmetic operation like "+-*/" to the input using weight
    
    # Example

    >>> model = Sequential([Arithmetic('*', initializer=np.array([2,10]), weight_shape=(2, 1), input_shape=(3,))])
    >>> model.get_weights()
    [array([[ 2.],
            [10.]], dtype=float32)]
    >>> model.predict(np.array([[1, 2, 3],
                                [4, 5, 6]]))
    array([[ 2.,  4.,  6.],
           [40., 50., 60.]], dtype=float32)
    """

    allowed_operations = "+-*/"

    def __init__(
        self,
        operation,
        initializer=None,
        weight_shape=None,
        input_as_operand=False,
        constraint=None,
        trainable=True,
        **kwargs,
    ):
        """
        # Arguments
            operation: Operation to perform between input and the weight.
                It must be one of the allowed operations.
                Check `Arithmetic.allowed_operations` to see what operations you can use.
            initializer: Initializer of the weight.
                Accepts string, instance of Initializer, and numerical values.
                Set to None to use default initializer that performs identity function.
                E.g., if the operation is '+' or '-', default initializer will be 'zeros'.
                If the operation is '*' or '/', default initializer will be 'ones'.
            weight_shape: Default shape is for a scalar number.
                Shape will be inferred from initializer if it's numerical values.
                If weight_shape is set, it will broadcast initializer to have shape
                = weight_shape. If broadcasting fails, a ValueError will be raised.
            input_as_operand: Whether to use the input as operand or operator of the operation to the weight.
            trainable: Whether the weight is variable or fixed.
        """
        super(Arithmetic, self).__init__(trainable=trainable, **kwargs)
        if not operation or operation not in self.allowed_operations:
            raise ValueError(
                f"Operation '{operation}' is not one of the allowed operations: '{self.allowed_operations}'"
            )
        self.operation = operation
        self.weight_shape = weight_shape
        if initializer is None:
            initializer = "ones" if operation in "*/" else "zeros"
        try:
            self.initializer = initializers.get(initializer)
        except ValueError:
            initializer = tf.constant_initializer(initializer)
            self.initializer = initializers.get(initializer)
        self.input_as_operand = input_as_operand
        self.constraint = constraints.get(constraint)
        self.trainable = trainable

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.w = self.add_weight(
            name="weight",
            shape=self.weight_shape,
            initializer=self.initializer,
            constraint=self.constraint,
            trainable=self.trainable,
        )
        super(Arithmetic, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        op = self.operation
        a, b = x, self.w
        if self.input_as_operand:
            a, b = b, a
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b

    def get_config(self):
        config = dict(
            operation=self.operation,
            weight_shape=self.weight_shape,
            initializer=initializers.serialize(self.initializer),
            input_as_operand=self.input_as_operand,
            constraint=constraints.serialize(self.constraint),
        )
        base_config = super(Arithmetic, self).get_config()
        config.update(base_config)
        return config


def reinitialize_weights(
    layers, reinit_kernel=True, reinit_bias=True, initializer=None
):
    """
    Re-initialize weights on a list of `layers` using their default initializers.

    Or use other initializers like `ones` or `zeros`.

    # Example
        Re-initialize on the entire model with default initializers
        >>> reinitialize_weights(model.layers)

        Re-initialize on one layer with zeros weights
        >>> reinitialize_weights(model.layers[-1], initializer='zeros')

        Check the weights after re-initialization
        >>> model.get_weights()
    """
    # make layers as list if only one element is provided
    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    if initializer is None:
        session = K.get_session()
        for layer in layers:
            if reinit_kernel and hasattr(layer, "kernel_initializer"):
                layer.kernel.initializer.run(session=session)
            if reinit_bias and hasattr(layer, "bias_initializer"):
                layer.bias.initializer.run(session=session)
    else:
        for layer in layers:
            w = layer.get_weights()
            assert len(w) == 2
            kernel, bias = w
            if initializer in ("zero", "zeros"):
                if reinit_kernel:
                    kernel = np.zeros_like(kernel)
                if reinit_bias:
                    bias = np.zeros_like(bias)
            elif initializer in ("one", "ones"):
                if reinit_kernel:
                    kernel = np.ones_like(kernel)
                if reinit_bias:
                    bias = np.ones_like(bias)
            else:
                raise ValueError("Unsupported `initializer` value.")
            layer.set_weights([kernel, bias])


def apply_residual_block(layers, x, activation=None, name=None):
    """Wrap a normal layer or many layers using residual mechanism.

    Compute `call_layers(layers, x)` which will be considered the "residual" 
    difference to add to the input `x`.

    Use this function when you want to learn deep network but are afraid that the
    network will suffer from too much depth. If too much depth is set, the
    residual block will simply force the `layers` to learn zero function (a function
    which always returns zero no matter what input it gets).
    In practice, zero function is a lot easier to learn than an identity function.

    The high-level idea is that a residual block allows the network to do
    skip-connections, so that it can choose to skip some of the layers to reduce
    the depth when it makes sense to do so.
    The idea is first introduced in the model called ResNet which allows you to
    train a very deep network without suffering from degradation of accuracy
    as the depth increases.

    # Arguments
        layers: Can be any callable that takes `x` as input and returns output with the
            same shape as `x`. It can be a keras Layer, a keras Model, a LayerStack,
            or a list of layers.
            Normally, this layer's activation function should not be set. If there are
            multiple layers, only the last layer should not have activation function.
        x: An input tensor to the `layers` list sequentially, must be a Keras
            tensor with the same shape as `layers(x)`.
        activation: The activation function to apply on the output `h`. If None, it is
            going to be linear/identity function.
        name: Name of the last layer of the block if provided

    # Returns
        A tensor `h = activation(x + layers(x))`, with `h` having the same shape as `x`
    
    # Example
        Create a model with 1 normal conv layer, followed by 1 residual block
        with 1 conv layer.
        >>> x = Input(input_shape, name='x')
        >>> h = Conv1D(48, 3, activation='relu', padding='same')(x)
        >>> h = apply_residual_block(Conv1D(48, 3, padding='same'), h, activation='relu')
        >>> model = Model(x, h)

        Create a model with 1 normal conv layer, followed by 1 residual block
        with 2 conv layers (2 conv layers in a block is a typical setting in
        a ResNet model).
        >>> x = Input(input_shape, name='x')
        >>> h = Conv1D(48, 3, activation='relu', padding='same')(x)
        >>> h = apply_residual_block([Conv1D(48, 3, padding='same', activation='relu'),
                                      Conv1D(48, 3, padding='same')], h, activation='relu', name='block_1')
        >>> model = Model(x, h)
    """
    residual = call_layers(layers, x)

    # defining name argument
    last_layer_params = {}
    if name:
        last_layer_params["name"] = name

    add_params = last_layer_params if not activation else {}
    h = add([x, residual], **add_params)
    if activation:
        h = Activation(activation, **last_layer_params)(h)
    return h


# region Quaternion
def get_quat_magnitude(q, K, axis=-1, keepdims=False, force_positive=True):
    """
    Get quaternion magnitude. `q` must be of shape (..., 4, ...)
    The axis of the quaternion is `axis`.
    `K` could be `numpy` or `keras.backend`
    """
    sum_or_eps = K.sum(q ** 2, axis=axis, keepdims=keepdims)
    # prevent negative input to sqrt
    if force_positive:
        try:
            sum_or_eps = K.maximum(sum_or_eps, K.epsilon())
        except AttributeError:
            # assume that K is numpy
            sum_or_eps = np.clip(sum_or_eps, np.finfo(sum_or_eps.dtype).eps, None)
    return K.sqrt(sum_or_eps)


def normalize_quat(q, K):
    """
    Return normalized quaternion with magnitude 1.
    `K` could be `numpy` or `keras.backend`
    """
    mag = get_quat_magnitude(q, K, keepdims=True)
    return q / mag


def normalize_quat_keras(q):
    """Return normalized quaternion as a form of keras backend's tensor"""
    return normalize_quat(q, K)


def NormalizeQuaternion(name="normalize_quaternion", **kwargs):
    """
    Return a keras Lambda layer that normalize the quaternion with keras backend.
    The shape of the input should be (..., 4). Shape of the output will be the same.
    """
    from keras.layers import Lambda

    return Lambda(normalize_quat_keras, name=name, **kwargs)


# endregion
