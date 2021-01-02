import inspect

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from spektral.layers.ops.scatter import deserialize_scatter, serialize_scatter
from spektral.utils.keras import (
    deserialize_kwarg,
    is_keras_kwarg,
    is_layer_kwarg,
    serialize_kwarg,
)


class MessagePassing(Layer):
    r"""
    A general class for message passing networks from the paper

    > [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)<br>
    > Justin Gilmer et al.

    **Mode**: single, disjoint.

    **This layer and all of its extensions expect a sparse adjacency matrix.**

    This layer computes:
    $$
        \x_i' = \gamma \left( \x_i, \square_{j \in \mathcal{N}(i)} \,
        \phi \left(\x_i, \x_j, \e_{j \rightarrow i} \right) \right),
    $$

    where \( \gamma \) is a differentiable update function, \( \phi \) is a
    differentiable message function, \( \square \) is a permutation-invariant
    function to aggregate the messages (like the sum or the average), and
    \(\E_{ij}\) is the edge attribute of edge i-j.

    By extending this class, it is possible to create any message-passing layer
    in single/disjoint mode.

    **API**

    ```python
    propagate(x, a, e=None, **kwargs)
    ```
    Propagates the messages and computes embeddings for each node in the graph. <br>
    Any `kwargs` will be forwarded as keyword arguments to `message()`,
    `aggregate()` and `update()`.

    ```python
    message(x, **kwargs)
    ```
    Computes messages, equivalent to \(\phi\) in the definition. <br>
    Any extra keyword argument of this function will be populated by
    `propagate()` if a matching keyword is found. <br>
    Use `self.get_i()` and  `self.get_j()` to gather the elements using the
    indices `i` or `j` of the adjacency matrix. Equivalently, you can access
    the indices themselves via the `index_i` and `index_j` attributes.

    ```python
    aggregate(messages, **kwargs)
    ```
    Aggregates the messages, equivalent to \(\square\) in the definition. <br>
    The behaviour of this function can also be controlled using the `aggregate`
    keyword in the constructor of the layer (supported aggregations: sum, mean,
    max, min, prod). <br>
    Any extra keyword argument of this function will be  populated by
    `propagate()` if a matching keyword is found.

    ```python
    update(embeddings, **kwargs)
    ```
    Updates the aggregated messages to obtain the final node embeddings,
    equivalent to \(\gamma\) in the definition. <br>
    Any extra keyword argument of this function will be  populated by
    `propagate()` if a matching keyword is found.

    **Arguments**:

    - `aggregate`: string or callable, an aggregation function. This flag can be
    used to control the behaviour of `aggregate()` wihtout re-implementing it.
    Supported aggregations: 'sum', 'mean', 'max', 'min', 'prod'.
    If callable, the function must have the signature `foo(updates, indices, n_nodes)`
    and return a rank 2 tensor with shape `(n_nodes, ...)`.
    - `kwargs`: additional keyword arguments specific to Keras' Layers, like
    regularizers, initializers, constraints, etc.
    """

    def __init__(self, aggregate="sum", **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if is_keras_kwarg(k)})
        self.kwargs_keys = []
        for key in kwargs:
            if is_layer_kwarg(key):
                attr = kwargs[key]
                attr = deserialize_kwarg(key, attr)
                self.kwargs_keys.append(key)
                setattr(self, key, attr)

        self.msg_signature = inspect.signature(self.message).parameters
        self.agg_signature = inspect.signature(self.aggregate).parameters
        self.upd_signature = inspect.signature(self.update).parameters
        self.agg = deserialize_scatter(aggregate)

    def call(self, inputs, **kwargs):
        x, a, e = self.get_inputs(inputs)
        return self.propagate(x, a, e)

    def build(self, input_shape):
        self.built = True

    def propagate(self, x, a, e=None, **kwargs):
        self.n_nodes = tf.shape(x)[-2]
        self.index_i = a.indices[:, 1]
        self.index_j = a.indices[:, 0]

        # Message
        msg_kwargs = self.get_kwargs(x, a, e, self.msg_signature, kwargs)
        messages = self.message(x, **msg_kwargs)

        # Aggregate
        agg_kwargs = self.get_kwargs(x, a, e, self.agg_signature, kwargs)
        embeddings = self.aggregate(messages, **agg_kwargs)

        # Update
        upd_kwargs = self.get_kwargs(x, a, e, self.upd_signature, kwargs)
        output = self.update(embeddings, **upd_kwargs)

        return output

    def message(self, x, **kwargs):
        return self.get_j(x)

    def aggregate(self, messages, **kwargs):
        return self.agg(messages, self.index_i, self.n_nodes)

    def update(self, embeddings, **kwargs):
        return embeddings

    def get_i(self, x):
        return tf.gather(x, self.index_i, axis=-2)

    def get_j(self, x):
        return tf.gather(x, self.index_j, axis=-2)

    def get_kwargs(self, x, a, e, signature, kwargs):
        output = {}
        for k in signature.keys():
            if signature[k].default is inspect.Parameter.empty or k == "kwargs":
                pass
            elif k == "x":
                output[k] = x
            elif k == "a":
                output[k] = a
            elif k == "e":
                output[k] = e
            elif k in kwargs:
                output[k] = kwargs[k]
            else:
                raise ValueError("Missing key {} for signature {}".format(k, signature))

        return output

    @staticmethod
    def get_inputs(inputs):
        if len(inputs) == 3:
            x, a, e = inputs
            assert K.ndim(e) in (2, 3), "E must have rank 2 or 3"
        elif len(inputs) == 2:
            x, a = inputs
            e = None
        else:
            raise ValueError(
                "Expected 2 or 3 inputs tensors (X, A, E), got {}.".format(len(inputs))
            )
        assert K.ndim(x) in (2, 3), "X must have rank 2 or 3"
        assert K.is_sparse(a), "A must be a SparseTensor"
        assert K.ndim(a) == 2, "A must have rank 2"

        return x, a, e

    def get_config(self):
        mp_config = {"aggregate": serialize_scatter(self.agg)}
        keras_config = {}
        for key in self.kwargs_keys:
            keras_config[key] = serialize_kwarg(key, getattr(self, key))
        base_config = super().get_config()

        return {**base_config, **keras_config, **mp_config, **self.config}

    @property
    def config(self):
        return {}

    @staticmethod
    def preprocess(a):
        return a
