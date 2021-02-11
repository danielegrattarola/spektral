import warnings
from functools import wraps

import tensorflow as tf
from tensorflow.keras.layers import Layer

from spektral.utils.keras import (
    deserialize_kwarg,
    is_keras_kwarg,
    is_layer_kwarg,
    serialize_kwarg,
)


class Conv(Layer):
    r"""
    A general class for convolutional layers.

    You can extend this class to create custom implementations of GNN layers
    that use standard matrix multiplication instead of the gather-scatter
    approach of MessagePassing.

    This is useful if you want to create layers that support dense inputs,
    batch and mixed modes, or other non-standard processing. No checks are done
    on the inputs, to allow for maximum flexibility.

    Any extension of this class must implement the `call(self, inputs)` and
    `config(self)` methods.

    **Arguments**:

    - ``**kwargs`: additional keyword arguments specific to Keras' Layers, like
    regularizers, initializers, constraints, etc.
    """

    def __init__(self, **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if is_keras_kwarg(k)})
        self.kwargs_keys = []
        for key in kwargs:
            if is_layer_kwarg(key):
                attr = kwargs[key]
                attr = deserialize_kwarg(key, attr)
                self.kwargs_keys.append(key)
                setattr(self, key, attr)
        self.call = check_dtypes_decorator(self.call)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        keras_config = {}
        for key in self.kwargs_keys:
            keras_config[key] = serialize_kwarg(key, getattr(self, key))
        return {**base_config, **keras_config, **self.config}

    @property
    def config(self):
        return {}

    @staticmethod
    def preprocess(a):
        return a


def check_dtypes_decorator(call):
    @wraps(call)
    def _inner_check_dtypes(inputs, **kwargs):
        inputs = check_dtypes(inputs)
        return call(inputs, **kwargs)

    return _inner_check_dtypes


def check_dtypes(inputs):
    if len(inputs) == 2:
        x, a = inputs
        e = None
    elif len(inputs) == 3:
        x, a, e = inputs
    else:
        return inputs

    if a.dtype in (tf.int32, tf.int64) and x.dtype in (
        tf.float16,
        tf.float32,
        tf.float64,
    ):
        warnings.warn(
            f"The adjacency matrix of dtype {a.dtype} is incompatible with the dtype "
            f"of the node features {x.dtype} and has been automatically cast to "
            f"{x.dtype}."
        )
        a = tf.cast(a, x.dtype)

    output = [_ for _ in [x, a, e] if _ is not None]
    return output
