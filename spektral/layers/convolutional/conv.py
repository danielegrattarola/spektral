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
