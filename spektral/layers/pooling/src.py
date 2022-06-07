import inspect

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from spektral.utils.keras import (
    deserialize_kwarg,
    is_keras_kwarg,
    is_layer_kwarg,
    serialize_kwarg,
)


class SRCPool(Layer):
    r"""
    A general class for graph pooling layers based on the "Select, Reduce,
    Connect" framework presented in:

    > [Understanding Pooling in Graph Neural Networks.](https://arxiv.org/abs/2110.05292)<br>
    > Daniele Grattarola et al.

    This layer computes:
    $$
        \begin{align}
            & \mathcal{S} = \left\{\mathcal{S}_k\right\}_{k=1:K} = \textrm{Sel}(\mathcal{G}) \\
            & \mathcal{X}'=\left\{\textrm{Red}( \mathcal{G}, \mathcal{S}_k )\right\}_{k=1:K} \\
            & \mathcal{E}'=\left\{\textrm{Con}( \mathcal{G}, \mathcal{S}_k, \mathcal{S}_l )\right\}_{k,L=1:K} \\
        \end{align}
    $$
    Where \(\textrm{Sel}\) is a node equivariant selection function that computes
    the supernode assignments \(\mathcal{S}_k\), \(\textrm{Red}\) is a
    permutation-invariant function to reduce the supernodes into the new node
    attributes, and \(\textrm{Con}\) is a permutation-invariant connection
    function that computes the link between the pooled nodes.

    By extending this class, it is possible to create any pooling layer in the
    SRC formalism.

    **Input**

    - `x`: Tensor of shape `([batch], N, F)` representing node features;
    - `a`: Tensor or SparseTensor of shape `([batch], N, N)` representing the
    adjacency matrix;
    - `i`: (optional) Tensor of integers with shape `(N, )` representing the
    batch index;

    **Output**

    - `x_pool`: Tensor of shape `([batch], K, F)`, representing the node
    features of the output. `K` is the number of output nodes and depends on the
    specific pooling strategy;
    - `a_pool`: Tensor or SparseTensor of shape `([batch], K, K)` representing
    the adjacency matrix of the output;
    - `i_pool`: (only if i was given as input) Tensor of integers with shape
    `(K, )` representing the batch index of the output;
    - `s`: (if `return_selection=True`) Tensor or SparseTensor representing the
    supernode assignments;

    **API**

    - `pool(x, a, i, **kwargs)`: pools the graph and returns the reduced node
    features and adjacency matrix. If the batch index `i` is not `None`, a
    reduced version of `i` will be returned as well.
    Any given `kwargs` will be passed as keyword arguments to `select()`,
    `reduce()` and `connect()` if any matching key is found.
    The mandatory arguments of `pool()` **must** be computed in `call()` by
    calling `self.get_inputs(inputs)`.
    - `select(x, a, i, **kwargs)`: computes supernode assignments mapping the
    nodes of the input graph to the nodes of the output.
    - `reduce(x, s, **kwargs)`: reduces the supernodes to form the nodes of the
    pooled graph.
    - `connect(a, s, **kwargs)`: connects the reduced supernodes.
    - `reduce_index(i, s, **kwargs)`: helper function to reduce the batch index
    (only called if `i` is given as input).

    When overriding any function of the API, it is possible to access the
    true number of nodes of the input (`n_nodes`) as a Tensor in the instance variable
    `self.n_nodes` (this is populated by `self.get_inputs()` at the beginning of
    `call()`).

    **Arguments**:

    - `return_selection`: if `True`, the Tensor used to represent supernode assignments
    will be returned with `x_pool`, `a_pool`, and `i_pool`;
    """

    def __init__(self, return_selection=False, **kwargs):
        # kwargs for the Layer class are handled automatically
        super().__init__(**{k: v for k, v in kwargs.items() if is_keras_kwarg(k)})
        self.supports_masking = True
        self.return_selection = return_selection

        # *_regularizer, *_constraint, *_initializer, activation, and use_bias
        # are dealt with automatically if passed to the constructor
        self.kwargs_keys = []
        for key in kwargs:
            if is_layer_kwarg(key):
                attr = kwargs[key]
                attr = deserialize_kwarg(key, attr)
                self.kwargs_keys.append(key)
                setattr(self, key, attr)

        # Signature of the SRC functions
        self.sel_signature = inspect.signature(self.select).parameters
        self.red_signature = inspect.signature(self.reduce).parameters
        self.con_signature = inspect.signature(self.connect).parameters
        self.i_red_signature = inspect.signature(self.reduce_index).parameters

        self._n_nodes = None

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # Always start the call() method with get_inputs(inputs) to set self.n_nodes
        x, a, i = self.get_inputs(inputs)

        return self.pool(x, a, i, **kwargs)

    def pool(self, x, a, i, **kwargs):
        """
        This is the core method of the SRC class, which runs a full pass of
        selection, reduction and connection.
        It is usually not necessary to modify this function. Any previous/shared
        operations should be done in `call()` and their results can be passed to
        the three SRC functions via keyword arguments (any kwargs given to this
        function will be matched to the signature of `select()`, `reduce()` and
        `connect()` and propagated as input to the three functions).
        Any pooling logic should go in the SRC functions themselves.
        :param x: Tensor of shape `([batch], N, F)`;
        :param a: Tensor or SparseTensor of shape `([batch], N, N)`;
        :param i: only in single/disjoint mode, Tensor of integers with shape
        `(N, )`; otherwise, `None`;
        :param kwargs: additional keyword arguments for `select()`, `reduce()`
        and `connect()`. Any matching kwargs will be passed to each of the three
        functions.
        :return:
            - `x_pool`: Tensor of shape `([batch], K, F)`, where `K` is the
            number of output nodes and depends on the pooling strategy;
            - `a_pool`: Tensor or SparseTensor of shape `([batch], K, K)`;
            - `i_pool`: (only if `i` is not `None`) Tensor of integers with shape
            `(K, )`;
        """
        # Select
        sel_kwargs = self._get_kwargs(x, a, i, self.sel_signature, kwargs)
        s = self.select(x, a, i, **sel_kwargs)

        # Reduce
        red_kwargs = self._get_kwargs(x, a, i, self.red_signature, kwargs)
        x_pool = self.reduce(x, s, **red_kwargs)

        # Index reduce
        i_red_kwargs = self._get_kwargs(x, a, i, self.i_red_signature, kwargs)
        i_pool = self.reduce_index(i, s, **i_red_kwargs) if i is not None else None

        # Connect
        con_kwargs = self._get_kwargs(x, a, i, self.con_signature, kwargs)
        a_pool = self.connect(a, s, **con_kwargs)

        return self.get_outputs(x_pool, a_pool, i_pool, s)

    def select(self, x, a, i, **kwargs):
        """
        Selection function. Given the graph, computes the supernode assignments
        that will eventually be mapped to the `K` nodes of the pooled graph.
        Supernode assignments are usually represented as a dense matrix of shape
        `(N, K)` or sparse indices of shape `(K, )`.
        :param x: Tensor of shape `([batch], N, F)`;
        :param a: Tensor or SparseTensor (depending on the implementation of the
        SRC functions) of shape `([batch], N, N)`;
        :param i: Tensor of integers with shape `(N, )` or `None`;
        :param kwargs: additional keyword arguments.
        :return: Tensor representing supernode assignments.
        """
        return tf.range(tf.shape(i))

    def reduce(self, x, s, **kwargs):
        """
        Reduction function. Given a selection, reduces the supernodes to form
        the nodes of the new graph.
        :param x: Tensor of shape `([batch], N, F)`;
        :param s: Tensor representing supernode assignments, as computed by
        `select()`;
        :param kwargs: additional keyword arguments; when overriding this
        function, any keyword argument defined explicitly as `key=None` will be
        automatically filled in when calling `pool(key=value)`.
        :return: Tensor of shape `([batch], K, F)` representing the node attributes of
        the pooled graph.
        """
        return tf.gather(x, s)

    def connect(self, a, s, **kwargs):
        """
        Connection function. Given a selection, connects the nodes of the pooled
        graphs.
        :param a: Tensor or SparseTensor of shape `([batch], N, N)`;
        :param s: Tensor representing supernode assignments, as computed by
        `select()`;
        :param kwargs: additional keyword arguments; when overriding this
        function, any keyword argument defined explicitly as `key=None` will be
        automatically filled in when calling `pool(key=value)`.
        :return: Tensor or SparseTensor of shape `([batch], K, K)` representing
        the adjacency matrix of the pooled graph.
        """
        return sparse_connect(a, s, self.n_nodes)

    def reduce_index(self, i, s, **kwargs):
        """
        Helper function to reduce the batch index `i`. Given a selection,
        returns a new batch index for the pooled graph. This is only called by
        `pool()` when `i` is given as input to the layer.
        :param i: Tensor of integers with shape `(N, )`;
        :param s: Tensor representing supernode assignments, as computed by
        `select()`.
        :param kwargs: additional keyword arguments; when overriding this
        function, any keyword argument defined explicitly as `key=None` will be
        automatically filled in when calling `pool(key=value)`.
        :return: Tensor of integers of shape `(K, )`.
        """
        return tf.gather(i, s)

    @staticmethod
    def _get_kwargs(x, a, i, signature, kwargs):
        output = {}
        for k in signature.keys():
            if signature[k].default is inspect.Parameter.empty or k == "kwargs":
                pass
            elif k == "x":
                output[k] = x
            elif k == "a":
                output[k] = a
            elif k == "i":
                output[k] = i
            elif k in kwargs:
                output[k] = kwargs[k]
            else:
                raise ValueError("Missing key {} for signature {}".format(k, signature))

        return output

    def get_inputs(self, inputs):
        if len(inputs) == 3:
            x, a, i = inputs
            if K.ndim(i) == 2:
                i = i[:, 0]
            assert K.ndim(i) == 1, "i must have rank 1"
        elif len(inputs) == 2:
            x, a = inputs
            i = None
        else:
            raise ValueError(
                "Expected 2 or 3 inputs tensors (x, a, i), got {}.".format(len(inputs))
            )

        self.n_nodes = tf.shape(x)[-2]

        return x, a, i

    def get_outputs(self, x_pool, a_pool, i_pool, s):
        output = [x_pool, a_pool]
        if i_pool is not None:
            output.append(i_pool)
        if self.return_selection:
            output.append(s)

        return output

    def get_config(self):
        config = {
            "return_selection": self.return_selection,
        }
        for key in self.kwargs_keys:
            config[key] = serialize_kwarg(key, getattr(self, key))
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_mask(self, inputs, mask=None):
        # After pooling all nodes are always valid
        return None

    @property
    def n_nodes(self):
        if self._n_nodes is None:
            raise ValueError(
                "self.n_nodes has not been defined. Have you called "
                "self.get_inputs(inputs) at the beginning of call()?"
            )
        return self._n_nodes

    @n_nodes.setter
    def n_nodes(self, value):
        self._n_nodes = value

    @n_nodes.deleter
    def n_nodes(self):
        self._n_nodes = None


def sparse_connect(A, S, N):
    N_sel = tf.cast(tf.shape(S), tf.int64)[0]
    m = tf.scatter_nd(S[:, None], tf.range(N_sel) + 1, (N,)) - 1

    row, col = A.indices[:, 0], A.indices[:, 1]
    r_mask = tf.gather(m, row)
    c_mask = tf.gather(m, col)
    mask_total = (r_mask >= 0) & (c_mask >= 0)
    r_new = tf.boolean_mask(r_mask, mask_total)
    c_new = tf.boolean_mask(c_mask, mask_total)
    v_new = tf.boolean_mask(A.values, mask_total)

    output = tf.SparseTensor(
        values=v_new, indices=tf.stack((r_new, c_new), 1), dense_shape=(N_sel, N_sel)
    )
    return tf.sparse.reorder(output)
