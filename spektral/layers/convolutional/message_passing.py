import inspect

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from spektral.utils.keras import is_layer_kwarg, is_keras_kwarg, deserialize_kwarg, serialize_kwarg
from spektral.layers.ops.scatter import deserialize_scatter


class MessagePassing(Layer):
    r"""
    A general class for message passing as presented by
    [Gilmer et al. (2017)](https://arxiv.org/abs/1704.01212).

    **Mode**: single, disjoint.

    **This layer and all of its extensions expect a sparse adjacency matrix.**

    This layer computes:
    $$
        \Z_i = \gamma \left( \X_i, \square_{j \in \mathcal{N}(i)} \,
        \phi \left(\X_i, \X_j, \E_{j,i} \right) \right),
    $$
    
    where \( \gamma \) is a differentiable update function, \( \phi \) is a
    differentiable message function, \( \square \) is a permutation-invariant
    function to aggregate the messages (like the sum or the average), and
    \(\E_{ij}\) is the edge attribute of edge i-j.

    By extending this class, it is possible to create any message-passing layer
    in single/disjoint mode.

    **API:**

    - `propagate(X, A, E=None, **kwargs)`: propagate the messages and computes
    embeddings for each node in the graph. `kwargs` will be propagated as
    keyword arguments to `message()`, `aggregate()` and `update()`.
    - `message(X, **kwargs)`: computes messages, equivalent to \(\phi\) in the
    definition.
    Any extra keyword argument of this function will be  populated by
    `propagate()` if a matching keyword is found.
    Use `self.get_i()` and  `self.get_j()` to gather the elements using the
    indices `i` or `j` of the adjacency matrix (e.g, `self.get_j(X)` will get
    the features of the neighbours).
    - `aggregate(messages, **kwargs)`: aggregates the messages, equivalent to
    \(\square\) in the definition.
    The behaviour of this function can also be controlled using the `aggregate`
    keyword in the constructor of the layer (supported aggregations: sum, mean,
    max, min, prod).
    Any extra keyword argument of this function will be  populated by
    `propagate()` if a matching keyword is found.
    - `update(embeddings, **kwargs)`: updates the aggregated messages to obtain
    the final node embeddings, equivalent to \(\gamma\) in the definition.
    Any extra keyword argument of this function will be  populated by
    `propagate()` if a matching keyword is found.

    **Arguments**:

    - `aggregate`: string or callable, an aggregate function. This flag can be
    used to control the behaviour of `aggregate()` wihtout re-implementing it.
    Supported aggregations: 'sum', 'mean', 'max', 'min', 'prod'.
    If callable, the function must have the signature `foo(updates, indices, N)`
    and return a rank 2 tensor with shape `(N, ...)`.
    """
    def __init__(self, aggregate='sum', **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if is_keras_kwarg(k)})
        self.output_dim = None
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
        X, A, E = self.get_inputs(inputs)
        return self.propagate(X, A, E)

    def build(self, input_shape):
        self.built = True

    def propagate(self, X, A, E=None, **kwargs):
        self.N = tf.shape(X)[0]
        self.index_i = A.indices[:, 0]
        self.index_j = A.indices[:, 1]

        # Message
        msg_kwargs = self.get_kwargs(X, A, E, self.msg_signature, kwargs)
        messages = self.message(X, **msg_kwargs)

        # Aggregate
        agg_kwargs = self.get_kwargs(X, A, E, self.agg_signature, kwargs)
        embeddings = self.aggregate(messages, **agg_kwargs)

        # Update
        upd_kwargs = self.get_kwargs(X, A, E, self.upd_signature, kwargs)
        output = self.update(embeddings, **upd_kwargs)

        return output

    def message(self, X, **kwargs):
        return self.get_j(X)

    def aggregate(self, messages, **kwargs):
        return self.agg(messages, self.index_i, self.N)

    def update(self, embeddings, **kwargs):
        return embeddings

    def get_i(self, x):
        return tf.gather(x, self.index_i)

    def get_j(self, x):
        return tf.gather(x, self.index_j)

    def get_kwargs(self, X, A, E, signature, kwargs):
        output = {}
        for k in signature.keys():
            if signature[k].default is inspect.Parameter.empty or k == 'kwargs':
                pass
            elif k == 'X':
                output[k] = X
            elif k == 'A':
                output[k] = A
            elif k == 'E':
                output[k] = E
            elif k in kwargs:
                output[k] = kwargs[k]
            else:
                raise ValueError('Missing key {} for signature {}'
                                 .format(k, signature))

        return output

    @staticmethod
    def get_inputs(inputs):
        if len(inputs) == 3:
            X, A, E = inputs
            assert K.ndim(E) == 2, 'E must have rank 2'
        elif len(inputs) == 2:
            X, A = inputs
            E = None
        else:
            raise ValueError('Expected 2 or 3 inputs tensors (X, A, E), got {}.'
                             .format(len(inputs)))
        assert K.ndim(X) == 2, 'X must have rank 2'
        assert K.is_sparse(A), 'A must be a SparseTensor'
        assert K.ndim(A) == 2, 'A must have rank 2'

        return X, A, E

    def compute_output_shape(self, input_shape):
        if self.output_dim:
            output_shape = input_shape[0][:-1] + (self.output_dim, )
        else:
            output_shape = input_shape[0]
        return output_shape

    def get_config(self):
        config = {
            'aggregate': self.agg,
        }
        for key in self.kwargs_keys:
            config[key] = serialize_kwarg(key, getattr(self, key))
        base_config = super().get_config()
        return {**base_config, **config}

    @staticmethod
    def preprocess(A):
        return A
