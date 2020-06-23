import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense

from spektral.layers import ops
from spektral.layers.ops import modes
from spektral.layers.convolutional.graph_conv import GraphConv


class EdgeConditionedConv(GraphConv):
    r"""
    An edge-conditioned convolutional layer (ECC) as presented by
    [Simonovsky & Komodakis (2017)](https://arxiv.org/abs/1704.02901).

    **Mode**: single, disjoint, batch.

    **Notes**:
        - This layer expects dense inputs and self-loops when working in batch mode.
        - In single mode, if the adjacency matrix is dense it will be converted
        to a SparseTensor automatically (which is an expensive operation).

    For each node \( i \), this layer computes:
    $$
        \Z_i = \X_{i} \W_{\textrm{root}} + \sum\limits_{j \in \mathcal{N}(i)} \X_{j} \textrm{MLP}(\E_{ji}) + \b
    $$
    where \(\textrm{MLP}\) is a multi-layer perceptron that outputs an
    edge-specific weight as a function of edge attributes.

    **Input**

    - Node features of shape `([batch], N, F)`;
    - Binary adjacency matrices of shape `([batch], N, N)`;
    - Edge features. In single mode, shape `(num_edges, S)`; in batch mode, shape
    `(batch, N, N, S)`.

    **Output**

    - node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `kernel_network`: a list of integers representing the hidden neurons of
    the kernel-generating network;
    - 'root': if False, the layer will not consider the root node for computing
    the message passing (first term in equation above), but only the neighbours.
    - `activation`: activation function to use;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `bias_constraint`: constraint applied to the bias vector.

    """

    def __init__(self,
                 channels,
                 kernel_network=None,
                 root=True,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(channels,
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs)
        self.kernel_network = kernel_network
        self.root = root

    def build(self, input_shape):
        F = input_shape[0][-1]
        F_ = self.channels
        self.kernel_network_layers = []
        if self.kernel_network is not None:
            for i, l in enumerate(self.kernel_network):
                self.kernel_network_layers.append(
                    Dense(l,
                          name='FGN_{}'.format(i),
                          activation='relu',
                          use_bias=self.use_bias,
                          kernel_initializer=self.kernel_initializer,
                          bias_initializer=self.bias_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          bias_regularizer=self.bias_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          bias_constraint=self.bias_constraint)
                )
        self.kernel_network_layers.append(Dense(F_ * F, name='FGN_out'))

        if self.root:
            self.root_kernel = self.add_weight(name='root_kernel',
                                               shape=(F, F_),
                                               initializer=self.kernel_initializer,
                                               regularizer=self.kernel_regularizer,
                                               constraint=self.kernel_constraint)
        else:
            self.root_kernel = None
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        X = inputs[0]  # (batch_size, N, F)
        A = inputs[1]  # (batch_size, N, N)
        E = inputs[2]  # (n_edges, S) or (batch_size, N, N, S)

        mode = ops.autodetect_mode(A, X)
        if mode == modes.SINGLE:
            return self._call_single(inputs)

        # Parameters
        N = K.shape(X)[-2]
        F = K.int_shape(X)[-1]
        F_ = self.channels

        # Filter network
        kernel_network = E
        for l in self.kernel_network_layers:
            kernel_network = l(kernel_network)

        # Convolution
        target_shape = (-1, N, N, F_, F) if mode == modes.BATCH else (N, N, F_, F)
        kernel = K.reshape(kernel_network, target_shape)
        output = kernel * A[..., None, None]
        output = tf.einsum('abicf,aif->abc', output, X)

        if self.root:
            output += ops.dot(X, self.root_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def _call_single(self, inputs):
        X = inputs[0]  # (N, F)
        A = inputs[1]  # (N, N)
        E = inputs[2]  # (n_edges, S)
        assert K.ndim(E) == 2, 'In single mode, E must have shape (n_edges, S).'

        # Enforce sparse representation
        if not K.is_sparse(A):
            A = ops.dense_to_sparse(A)

        # Parameters
        N = tf.shape(X)[-2]
        F = K.int_shape(X)[-1]
        F_ = self.channels

        # Filter network
        kernel_network = E
        for l in self.kernel_network_layers:
            kernel_network = l(kernel_network)  # (n_edges, F * F_)
        target_shape = (-1, F, F_)
        kernel = tf.reshape(kernel_network, target_shape)

        # Propagation
        index_i = A.indices[:, -2]
        index_j = A.indices[:, -1]
        messages = tf.gather(X, index_j)
        messages = ops.dot(messages[:, None, :], kernel)[:, 0, :]
        aggregated = ops.scatter_sum(messages, index_i, N)

        # Update
        output = aggregated
        if self.root:
            output += ops.dot(X, self.root_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = {
            'kernel_network': self.kernel_network,
            'root': self.root,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def preprocess(A):
        return A
