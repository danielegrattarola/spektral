import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from spektral.utils.keras import is_layer_kwarg, is_keras_kwarg, deserialize_kwarg, serialize_kwarg
from spektral.layers.ops.scatter import deserialize_scatter


class MessagePassing(Layer):
    r"""
    A general class for message passing as presented by
    [Gilmer et al. (2017)](https://arxiv.org/abs/1704.01212).

    **Mode**: single/disjoint.

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

    - `propagate(X, A, E=None)`: propagate the messages and computes embeddings
    for each node in the graph.
    - `message(X_j)`: computes messages for each node j. `X_j` is rank 2 tensor
    obtained by gathering the node features of each node's neighbours, i.e.,
    `X_j = X[A.indices[1]]`. This is equivalent to \(\phi\) in the definition
    (support for X_i and E_ij can be obtained by extending this layer and
    re-implementing the `message` and `propagate` functions).
    - `aggregate(messages, indices, N)`: aggregates the messages according to
    `indices` (usually `indices = A.indices[0]`, i.e., the messages from each
    node's neighbours are aggregated). This is equivalent to \(\square\) in
    the definition. The behaviour of this function can also be controlled using
    the `aggregate` keyword in the constructor of the layer (supported
    aggregations: sum, mean, max, min, prod).
    - `update(embeddings)`: updates the aggregated messages. This is equivalent
    to \(\gamma\) in the definition (support for X_i can be obtained by
    extending this layer and re-implementing the `update` and `propagate`
    functions).

    **Arguments**:

    - `aggregate`: string or callable, an aggregate function. This flag can be
    used to control the behaviour of `aggregate()` wihtout re-implementing it.
    Supported aggregations: 'sum', 'mean', 'max', 'min', 'prod'.
    If callable, the function must have the signature `foo(updates, indices, N)`
    and return a rank 2 tensor with shape `(N, ...)`.
    """
    def __init__(self, aggregate='sum', **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if is_keras_kwarg(k)})
        self.aggr = deserialize_scatter(aggregate)
        self.output_dim = None
        self.kwargs_keys = []
        for key in kwargs:
            if is_layer_kwarg(key):
                attr = kwargs[key]
                attr = deserialize_kwarg(key, attr)
                self.kwargs_keys.append(key)
                setattr(self, key, attr)

    def call(self, inputs, **kwargs):
        X, A, E = self.get_inputs(inputs)
        return self.propagate(X, A, E)

    def build(self, input_shape):
        self.built = True

    def propagate(self, X, A, E=None):
        assert K.is_sparse(A), 'A must be a SparseTensor'
        N = tf.shape(X)[0]
        index_i, index_j = A.indices[:, 0], A.indices[:, 1]
        x_j = tf.gather(X, index_j)
        messages = self.message(x_j)
        embeddings = self.aggregate(messages, index_i, N)
        output = self.update(embeddings)

        return output

    def message(self, x_j):
        return x_j

    def aggregate(self, messages, indices, N):
        return self.aggr(messages, indices, N)

    def update(self, embeddings):
        return embeddings

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
            'aggregate': self.aggr,
        }
        for key in self.kwargs_keys:
            config[key] = serialize_kwarg(key, getattr(self, key))
        base_config = super().get_config()
        return {**base_config, **config}

    @staticmethod
    def preprocess(A):
        return A
