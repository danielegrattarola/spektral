import tensorflow as tf
from keras import activations, initializers, regularizers, constraints
from keras import backend as K
from keras.layers import Layer, LeakyReLU, Dropout

from spektral.layers.ops import filter_dot
from spektral.layers import ops


class GraphConv(Layer):
    """
    A graph convolutional layer as presented by
    [Kipf & Welling (2016)](https://arxiv.org/abs/1609.02907).

    **Mode**: single, mixed, batch.
    
    This layer computes:
    $$  
        Z = \\sigma( \\tilde{A} XW + b)
    $$
    where \(X\) is the node features matrix, \(\\tilde{A}\) is the normalized
    Laplacian, \(W\) is the convolution kernel, \(b\) is a bias vector, and
    \(\\sigma\) is the activation function.
    
    **Input**
    
    - Node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - Normalized Laplacian of shape `(n_nodes, n_nodes)` (with optional `batch`
    dimension); see `spektral.utils.convolution.localpooling_filter`.
    
    **Output**
    
    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.
        
    **Arguments**
    
    - `channels`: integer, number of output channels;
    - `activation`: activation function to use;
    - `use_bias`: whether to add a bias to the linear transformation;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;  
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.
    
    **Usage**  
    
    ```py
    # Load data
    A, X, _, _, _, _, _, _ = citation.load_data('cora')

    # Preprocessing operations
    fltr = utils.localpooling_filter(A)

    # Model definition
    X_in = Input(shape=(F, ))
    fltr_in = Input((N, ), sparse=True)
    output = GraphConv(channels)([X_in, fltr_in])
    ```
    """
    def __init__(self,
                 channels,
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
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.channels = channels
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(shape=(input_dim, self.channels),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        features = inputs[0]
        fltr = inputs[1]

        # Convolution
        output = K.dot(features, self.kernel)
        output = filter_dot(fltr, output)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        features_shape = input_shape[0]
        output_shape = features_shape[:-1] + (self.channels,)
        return output_shape

    def get_config(self):
        config = {
            'channels': self.channels,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ChebConv(GraphConv):
    """
    A Chebyshev convolutional layer as presented by
    [Defferrard et al. (2016)](https://arxiv.org/abs/1606.09375).

    **Mode**: single, mixed, batch.
    
    Given a list of Chebyshev polynomials \(T = [T_{1}, ..., T_{K}]\), 
    this layer computes:
    $$
        Z = \\sigma( \\sum \\limits_{k=1}^{K} T_{k} X W  + b)
    $$
    where \(X\) is the node features matrix, \(W\) is the convolution kernel, 
    \(b\) is the bias vector, and \(\sigma\) is the activation function.

    **Input**

    - Node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - A list of Chebyshev polynomials of shape `(num_nodes, num_nodes)` (with
    optional `batch` dimension); see `spektral.utils.convolution.chebyshev_filter`.

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.
    
    **Arguments**
    
    - `channels`: integer, number of output channels;
    - `activation`: activation function to use;
    - `use_bias`: boolean, whether to add a bias to the linear transformation;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;  
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.
    
    **Usage**
    ```py
    # Load data
    A, X, _, _, _, _, _, _ = citation.load_data('cora')

    # Preprocessing operations
    fltr = utils.chebyshev_filter(A, K)

    # Model definition
    X_in = Input(shape=(F, ))
    fltr_in = [Input((N, ), sparse=True) for _ in range(K + 1)]
    output = ChebConv(channels)([X_in] + fltr_in)
    ```
    """
    def __init__(self,
                 channels,
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
        super().__init__(channels, **kwargs)
        self.channels = channels
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        support_len = len(input_shape) - 1
        self.kernel = self.add_weight(shape=(input_dim * support_len, self.channels),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        features = inputs[0]
        fltr_list = inputs[1:]

        # Convolution
        supports = list()
        for fltr in fltr_list:
            s = filter_dot(fltr, features)
            supports.append(s)
        supports = K.concatenate(supports, axis=-1)
        output = K.dot(supports, self.kernel)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


class GraphSageConv(GraphConv):
    """
    A GraphSage layer as presented by [Hamilton et al. (2017)](https://arxiv.org/abs/1706.02216).

    **Mode**: single.

    This layer computes:
    $$
        Z = \\sigma \\big( \\big[ \\textrm{AGGREGATE}(X) \\| X \\big] W + b \\big)
    $$
    where \(X\) is the node features matrix, \(W\) is a trainable kernel,
    \(b\) is a bias vector, and \(\\sigma\) is the activation function.
    \(\\textrm{AGGREGATE}\) is an aggregation function as described in the
    original paper, that works by aggregating each node's neighbourhood
    according to some rule. The supported aggregation methods are: sum, mean,
    max, min, and product.

    **Input**

    - Node features of shape `(n_nodes, n_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`.

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `aggregate_method`: str, aggregation method to use (`'sum'`, `'mean'`,
    `'max'`, `'min'`, `'prod'`);
    - `activation`: activation function to use;
    - `use_bias`: whether to add a bias to the linear transformation;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.

    **Usage**

    ```py
    X_in = Input(shape=(F, ))
    A_in = Input((N, ), sparse=True)
    output = GraphSageConv(channels)([X_in, A_in])
    ```
    """

    def __init__(self,
                 channels,
                 aggregate_method='mean',
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
        super().__init__(channels, **kwargs)
        self.channels = channels
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False
        if aggregate_method == 'sum':
            self.aggregate_op = tf.segment_sum
        elif aggregate_method == 'mean':
            self.aggregate_op = tf.segment_mean
        elif aggregate_method == 'max':
            self.aggregate_op = tf.segment_max
        elif aggregate_method == 'min':
            self.aggregate_op = tf.segment_sum
        elif aggregate_method == 'prod':
            self.aggregate_op = tf.segment_prod
        else:
            raise ValueError('Possbile aggragation methods: sum, mean, max, min, '
                             'prod')

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(shape=(2 * input_dim, self.channels),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        features = inputs[0]
        fltr = inputs[1]

        if not K.is_sparse(fltr):
            fltr = tf.contrib.layers.dense_to_sparse(fltr)

        features_neigh = self.aggregate_op(
            tf.gather(features, fltr.indices[:, -1]), fltr.indices[:, -2]
        )
        output = K.concatenate([features, features_neigh])
        output = K.dot(output, self.kernel)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        output = K.l2_normalize(output, axis=-1)
        return output


class EdgeConditionedConv(GraphConv):
    """
    An edge-conditioned convolutional layer as presented by [Simonovsky and
    Komodakis (2017)](https://arxiv.org/abs/1704.02901).

    **Mode**: single, batch.

    **This layer expects dense inputs.**
    
    For each node \(i\), this layer computes:
    $$
        Z_i =  \\frac{1}{\\mathcal{N}(i)} \\sum\\limits_{j \\in \\mathcal{N}(i)} F(E_{ji}) X_{j} + b
    $$
    where \(\\mathcal{N}(i)\) represents the one-step neighbourhood of node \(i\),
     \(F\) is a neural network that outputs the convolution kernel as a
    function of edge attributes, \(E\) is the edge attributes matrix, and \(b\)
    is a bias vector.

    **Input**

    - node features of shape `(n_nodes, n_node_features)` (with optional `batch`
    dimension);
    - binary adjacency matrices with self-loops, of shape `(n_nodes, num_nodes)`
    (with optional `batch` dimension);
    - edge features of shape `(n_nodes, n_nodes, n_edge_features)` (with
    optional `batch` dimension);

    **Output**

    - node features with the same shape of the input, but the last dimension
    changed to `channels`.
    
    **Arguments**
    
    - `channels`: integer, number of output channels;
    - `kernel_network`: a list of integers describing the hidden structure of
    the kernel-generating network (i.e., the ReLU layers before the linear
    output);
    - `activation`: activation function to use;
    - `use_bias`: boolean, whether to add a bias to the linear transformation;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;  
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.
    
    **Usage**
    ```py
    X_in = Input(shape=(N, F))
    A_in = Input(shape=(N, N))
    E_in = Input(shape=(N, N, S))
    output = EdgeConditionedConv(channels)([X_in, A_in, E_in])
    ```
    """
    # TODO: single, mixed
    def __init__(self,
                 channels,
                 kernel_network=None,
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
        super().__init__(channels, **kwargs)
        self.channels = channels
        self.kernel_network = kernel_network
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        X = inputs[0]  # (batch_size, N, F)
        A = inputs[1]  # (batch_size, N, N)
        E = inputs[2]  # (batch_size, N, N, S)

        mode = ops.autodetect_mode(A, X)

        # Parameters
        N = K.shape(X)[-2]
        F = K.int_shape(X)[-1]
        F_ = self.channels

        # Normalize adjacency matrix
        A = ops.normalize_A(A)

        # Filter network
        kernel_network = E
        if self.kernel_network is not None:
            for i, l in enumerate(self.kernel_network):
                kernel_network = self.dense_layer(kernel_network, l,
                                                  'FGN_{}'.format(i),
                                                  activation='relu',
                                                  use_bias=self.use_bias,
                                                  kernel_initializer=self.kernel_initializer,
                                                  bias_initializer=self.bias_initializer,
                                                  kernel_regularizer=self.kernel_regularizer,
                                                  bias_regularizer=self.bias_regularizer,
                                                  kernel_constraint=self.kernel_constraint,
                                                  bias_constraint=self.bias_constraint)
        kernel_network = self.dense_layer(kernel_network, F_ * F, 'FGN_out')

        # Convolution
        target_shape = (-1, N, N, F_, F) if mode == ops.modes['B'] else (N, N, F_, F)
        kernel = K.reshape(kernel_network, target_shape)
        output = kernel * A[..., None, None]

        if mode == ops.modes['B']:
            output = tf.einsum('abicf,aif->abc', output, X)
        else:
            output = tf.einsum('bicf,if->bc', output, X)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = {
            'channels': self.channels,
            'kernel_network': self.kernel_network,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def dense_layer(self,
                    x,
                    units,
                    name,
                    activation=None,
                    use_bias=True,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None):
        input_dim = K.int_shape(x)[-1]
        kernel = self.add_weight(shape=(input_dim, units),
                                 name=name + '_kernel',
                                 initializer=kernel_initializer,
                                 regularizer=kernel_regularizer,
                                 constraint=kernel_constraint)
        bias = self.add_weight(shape=(units,),
                               name=name + '_bias',
                               initializer=bias_initializer,
                               regularizer=bias_regularizer,
                               constraint=bias_constraint)
        act = activations.get(activation)
        output = K.dot(x, kernel)
        if use_bias:
            output = K.bias_add(output, bias)
        output = act(output)
        return output


class GraphAttention(GraphConv):
    """
    A graph attention layer as presented by
    [Velickovic et al. (2017)](https://arxiv.org/abs/1710.10903).

    **Mode**: single, mixed, batch.

    **This layer expects dense inputs.**
    
    This layer computes a convolution similar to `layers.GraphConv`, but
    uses the attention mechanism to weight the adjacency matrix instead of
    using the normalized Laplacian.

    **Input**

    - node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - adjacency matrices of shape `(n_nodes, n_nodes)` (with optional `batch`
    dimension);

    **Output**

    - node features with the same shape of the input, but the last dimension
    changed to `channels`.
    - if `return_attn_coef=True`, a list with the attention coefficients for each
    attention head is returned as well. Each attention coefficient matrix has
    shape `(n_nodes, n_nodes)` (with optional `batch` dimension);
    
    **Arguments**
    
    - `channels`: integer, number of output channels;
    - `attn_heads`: number of attention heads to use;
    - `concat_heads`: bool, whether to concatenate the output of the attention
     heads instead of averaging;
    - `dropout_rate`: internal dropout rate for attention coefficients;
    - `return_attn_coef`: bool, if True, return the attention coefficients for
    the given input (one N x N matrix for each head).
    - `activation`: activation function to use;
    - `use_bias`: boolean, whether to add a bias to the linear transformation;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `attn_kernel_initializer`: initializer for the attention kernel matrices;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;  
    - `attn_kernel_regularizer`: regularization applied to the attention kernel 
    matrices;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `attn_kernel_constraint`: constraint applied to the attention kernel
    matrices;
    - `bias_constraint`: constraint applied to the bias vector.
    
    **Usage**

    ```py
    # Load data
    A, X, _, _, _, _, _, _ = citation.load_data('cora')

    # Preprocessing operations
    A = utils.add_eye(A).toarray()  # Add self-loops

    # Model definition
    X_in = Input(shape=(F, ))
    A_in = Input((N, ))
    output = GraphAttention(channels)([X_in, A_in])
    # Alternative
    # output, attn_coef = GraphAttention(channels, return_attn_coef=True)([X_in, A_in])
    ```
    """
    def __init__(self,
                 channels,
                 attn_heads=1,
                 concat_heads=True,
                 dropout_rate=0.5,
                 return_attn_coef=False,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        super().__init__(channels, **kwargs)

        self.channels = channels
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.dropout_rate = dropout_rate
        self.return_attn_coef = return_attn_coef
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if concat_heads:
            # Output will have shape (..., attention_heads * channels)
            self.output_dim = self.channels * self.attn_heads
        else:
            # Output will have shape (..., channels)
            self.output_dim = self.channels

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(input_dim, self.channels),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.channels,),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.channels, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head))
            attn_kernel_neighs = self.add_weight(shape=(self.channels, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs):
        X = inputs[0]
        A = inputs[1]

        outputs = []
        output_attn = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = K.dot(X, kernel)

            # Compue attention coefficients
            # [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features, attention_kernel[0])    # [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(features, attention_kernel[1])  # [a_2]^T [Wh_j]
            if len(K.int_shape(features)) == 2:
                # Single / mixed mode
                attn_for_neighs_T = K.transpose(attn_for_neighs)
            else:
                # Batch mode
                attn_for_neighs_T = K.permute_dimensions(attn_for_neighs, (0, 2, 1))
            attn_coef = attn_for_self + attn_for_neighs_T
            attn_coef = LeakyReLU(alpha=0.2)(attn_coef)

            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e9 * (1.0 - A)
            attn_coef += mask

            # Apply softmax to get attention coefficients
            attn_coef = K.softmax(attn_coef)
            output_attn.append(attn_coef)

            # Apply dropout to attention coefficients
            attn_coef_drop = Dropout(self.dropout_rate)(attn_coef)

            # Convolution
            features = filter_dot(attn_coef_drop, features)
            if self.use_bias:
                features = K.bias_add(features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(features)

        # Aggregate the heads' output according to the reduction method
        if self.concat_heads:
            output = K.concatenate(outputs)
        else:
            output = K.mean(K.stack(outputs), axis=0)

        output = self.activation(output)

        if self.return_attn_coef:
            return output, output_attn
        else:
            return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][:-1] + (self.output_dim,)
        return output_shape

    def get_config(self):
        config = {
            'channels': self.channels,
            'attn_heads': self.attn_heads,
            'concat_heads': self.concat_heads,
            'dropout_rate': self.dropout_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'attn_kernel_initializer': initializers.serialize(self.attn_kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'attn_kernel_regularizer': regularizers.serialize(self.attn_kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'attn_kernel_constraint': constraints.serialize(self.attn_kernel_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GraphConvSkip(GraphConv):
    """
    A graph convolutional layer as presented by
    [Kipf & Welling (2016)](https://arxiv.org/abs/1609.02907), with the addition
    of a skip connection.

    **Mode**: single, mixed, batch.

    This layer computes:
    $$
        Z = \\sigma(A X W_1 + X W_2 + b)
    $$
    where \(X\) is the node features matrix, \(A\) is the normalized laplacian,
    \(W_1\) and \(W_2\) are the convolution kernels, \(b\) is a bias vector,
    and \(\\sigma\) is the activation function.

    **Input**

    - node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - Normalized adjacency matrix of shape `(n_nodes, n_nodes)` (with optional
    `batch` dimension); see `spektral.utils.convolution.normalized_adjacency`.

    **Output**

    - node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `activation`: activation function to use;
    - `use_bias`: whether to add a bias to the linear transformation;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.

    **Usage**
    ```py
    # Load data
    A, X, _, _, _, _, _, _ = citation.load_data('cora')

    # Preprocessing operations
    fltr = utils.normalized_adjacency(A)

    # Model definition
    X_in = Input(shape=(F, ))
    fltr_in = Input((N, ), sparse=True)
    output = GraphConvSkip(channels)([X_in, fltr_in])
    ```
    """

    def __init__(self,
                 channels,
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
        super().__init__(channels, **kwargs)
        self.channels = channels
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]

        self.kernel_1 = self.add_weight(shape=(input_dim, self.channels),
                                        initializer=self.kernel_initializer,
                                        name='kernel_1',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        self.kernel_2 = self.add_weight(shape=(input_dim, self.channels),
                                        initializer=self.kernel_initializer,
                                        name='kernel_2',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        features = inputs[0]
        fltr = inputs[1]

        # Convolution
        output = K.dot(features, self.kernel_1)
        output = filter_dot(fltr, output)

        # Skip connection
        skip = K.dot(features, self.kernel_2)
        output += skip

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


class ARMAConv(GraphConv):
    """
    A graph convolutional layer with ARMA(K) filters, as presented by
    [Bianchi et al. (2019)](https://arxiv.org/abs/1901.01343).

    **Mode**: single, mixed, batch.

    This layer computes:
    $$
        Z = \\frac{1}{K}\\sum \\limits_{k=1}^K \\bar{X}_k^{(T)},
    $$
    where \(K\) is the order of the ARMA(K) filter, and where:
    $$
        \\bar{X}_k^{(t + 1)} =  \\sigma\\left(\\tilde{L}\\bar{X}^{(t)}W^{(t)} + XV^{(t)}\\right)
    $$
    is a graph convolutional skip layer implementing a recursive approximation
    of an ARMA(1) filter, \(\\tilde{L}\) is  normalized graph Laplacian with
    a rescaled spectrum, \(\\bar{X}^{(0)} = X\), and \(W, V\) are trainable
    kernels.

    **Input**

    - node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - Normalized Laplacian of shape `(n_nodes, n_nodes)` (with optional `batch`
    dimension); see the [ARMA node classification example](https://github.com/danielegrattarola/spektral/blob/master/examples/node_classification_arma.py)

    **Output**

    - node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `order`: order of the full ARMA(K) filter, i.e., the number of parallel
    stacks in the layer;
    - `iterations`: number of iterations to compute each ARMA(1) approximation;
    - `share_weights`: share the weights in each ARMA(1) stack.
    - `gcn_activation`: activation function to use to compute each ARMA(1) stack;
    - `dropout_rate`: dropout rate for skip connection;
    - `activation`: activation function to use;
    - `use_bias`: whether to add a bias to the linear transformation;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.

    **Usage**
    ```py
    # Load data
    A, X, _, _, _, _, _, _ = citation.load_data('cora')

    # Preprocessing operations
    fltr = utils.normalized_adjacency(A)

    # Model definition
    X_in= Input(shape=(F, ), sparse=True)
    fltr_in = Input((N, ))
    output = ARMAConv(channels)([X_in, fltr_in])
    ```
    """

    def __init__(self,
                 channels,
                 order=1,
                 iterations=1,
                 share_weights=False,
                 gcn_activation='relu',
                 dropout_rate=0.0,
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
        super().__init__(channels, **kwargs)
        self.channels = channels
        self.iterations = iterations
        self.order = order
        self.share_weights = share_weights
        self.activation = activations.get(activation)
        self.gcn_activation = activations.get(gcn_activation)
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 2
        # When using shared weights, pre-compute them here
        if self.share_weights:
            self.kernels_in = []  # Weights from input space to output space
            self.kernels_hid = []  # Weights from output space to output space
            for k in range(self.order):
                self.kernels_in.append(self.get_gcn_weights(input_shape[0][-1],
                                                            input_shape[0][-1],
                                                            self.channels,
                                                            name='ARMA_skip_{}r_in'.format(k),
                                                            use_bias=self.use_bias,
                                                            kernel_initializer=self.kernel_initializer,
                                                            bias_initializer=self.bias_initializer,
                                                            kernel_regularizer=self.kernel_regularizer,
                                                            bias_regularizer=self.bias_regularizer,
                                                            kernel_constraint=self.kernel_constraint,
                                                            bias_constraint=self.bias_constraint))
                if self.iterations > 1:
                    self.kernels_hid.append(self.get_gcn_weights(self.channels,
                                                                 input_shape[0][-1],
                                                                 self.channels,
                                                                 name='ARMA_skip_{}r_hid'.format(k),
                                                                 use_bias=self.use_bias,
                                                                 kernel_initializer=self.kernel_initializer,
                                                                 bias_initializer=self.bias_initializer,
                                                                 kernel_regularizer=self.kernel_regularizer,
                                                                 bias_regularizer=self.bias_regularizer,
                                                                 kernel_constraint=self.kernel_constraint,
                                                                 bias_constraint=self.bias_constraint))
        self.built = True

    def call(self, inputs):
        features = inputs[0]
        fltr = inputs[1]

        # Convolution
        output = []  # Stores the parallel filters
        for k in range(self.order):
            output_k = features
            for t in range(self.iterations):
                output_k = self.graph_conv_skip([output_k, features, fltr],
                                                self.channels,
                                                'ARMA_skip_{}{}'.format(k, t),
                                                recurrent_k=k if self.share_weights else None,
                                                recurrent_t=t if self.share_weights else None,
                                                activation=self.gcn_activation,
                                                use_bias=self.use_bias,
                                                kernel_initializer=self.kernel_initializer,
                                                bias_initializer=self.bias_initializer,
                                                kernel_regularizer=self.kernel_regularizer,
                                                bias_regularizer=self.bias_regularizer,
                                                kernel_constraint=self.kernel_constraint,
                                                bias_constraint=self.bias_constraint)
            output.append(output_k)

        # Average pooling
        output = K.stack(output, axis=-1)
        output = K.mean(output, axis=-1)
        output = self.activation(output)
        
        return output

    def get_config(self):
        config = {
            'channels': self.channels,
            'iterations': self.iterations,
            'order': self.order,
            'share_weights': self.share_weights,
            'activation': activations.serialize(self.activation),
            'gcn_activation': activations.serialize(self.gcn_activation),
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_gcn_weights(self, input_dim, input_dim_skip, channels, name,
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None):
        """
        Creates a set of weights for a GCN with skip connections
        :param input_dim: dimension of the input space of the layer
        :param input_dim_skip: dimension of the input space for the skip connection
        :param channels: dimension of the output space
        :param name: name of the layer
        :param use_bias: whether to create a bias vector (if False, returns None as bias)
        :param kernel_initializer: initializer for the kernels
        :param bias_initializer: initializer for the bias
        :param kernel_regularizer: regularizer for the kernels
        :param bias_regularizer: regularizer for the bias
        :param kernel_constraint: constraint for the kernel
        :param bias_constraint: constraint for the bias
        :return:
            - kernel_1, from input space of the layer to output space
            - kernel_2, from input space of the skip connection to output space
            - bias, bias vector on the output space if use_bias=True, None otherwise.
        """
        kernel_initializer = initializers.get(kernel_initializer)
        kernel_regularizer = regularizers.get(kernel_regularizer)
        kernel_constraint = constraints.get(kernel_constraint)
        bias_initializer = initializers.get(bias_initializer)
        bias_regularizer = regularizers.get(bias_regularizer)
        bias_constraint = constraints.get(bias_constraint)
        kernel_1 = self.add_weight(shape=(input_dim, channels),
                                   name=name + '_kernel_1',
                                   initializer=kernel_initializer,
                                   regularizer=kernel_regularizer,
                                   constraint=kernel_constraint)
        kernel_2 = self.add_weight(shape=(input_dim_skip, channels),
                                   name=name + '_kernel_2',
                                   initializer=kernel_initializer,
                                   regularizer=kernel_regularizer,
                                   constraint=kernel_constraint)
        if use_bias:
            bias = self.add_weight(shape=(channels,),
                                   name=name + '_bias',
                                   initializer=bias_initializer,
                                   regularizer=bias_regularizer,
                                   constraint=bias_constraint)
        else:
            bias = None
        return kernel_1, kernel_2, bias

    def graph_conv_skip(self, x, channels, name,
                        recurrent_k=None,
                        recurrent_t=None,
                        activation=None,
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None):
        """
        Creates a graph convolutional layer with a skip connection
        :param x: list of input tensors, namely
            - input of the layer
            - input for the skip connection
            - laplacian, or normalized adjacency matrix
        :param channels: dimension of the output space
        :param name: name of the layer
        :param recurrent_k: if the recurrent flag was set, then use the shared
        weights of the k-th filter when creating the layer
        :param recurrent_t: if the recurrent flag was set, then use the shared
        weights when computing the t-th recursive step of the k-th filter.
        Note that this parameter cannot be None if recurent_k is not None.
        :param activation: activation function for the layer
        :param use_bias: whether to add a bias vector
        :param kernel_initializer: initializer for the kernels
        :param bias_initializer: initializer for the bias
        :param kernel_regularizer: regularizer for the kernels
        :param bias_regularizer: regularizer for the bias
        :param kernel_constraint: constraint for the kernel
        :param bias_constraint: constraint for the bias
        :return: output of the layer
        """
        input_dim = K.int_shape(x[0])[-1]
        input_dim_skip = K.int_shape(x[1])[-1]

        if recurrent_k is None:
            kernel_1, kernel_2, bias = self.get_gcn_weights(input_dim,
                                                            input_dim_skip,
                                                            channels,
                                                            name,
                                                            kernel_initializer=kernel_initializer,
                                                            bias_initializer=bias_initializer,
                                                            kernel_regularizer=kernel_regularizer,
                                                            bias_regularizer=bias_regularizer,
                                                            kernel_constraint=kernel_constraint,
                                                            bias_constraint=bias_constraint)
        else:
            # When using shared weights, use the pre-computed ones.
            if recurrent_t is None:
                raise ValueError('recurrent_k and recurrent_t must be set together.')
            if recurrent_t == 0:
                kernel_1, kernel_2, bias = self.kernels_in[recurrent_k]
            else:
                kernel_1, kernel_2, bias = self.kernels_hid[recurrent_k]
        features = x[0]
        features_skip = x[1]
        fltr = x[2]

        # Convolution
        output = K.dot(features, kernel_1)
        output = filter_dot(fltr, output)

        # Skip connection
        skip = K.dot(features_skip, kernel_2)
        skip = Dropout(self.dropout_rate)(skip)
        output += skip

        if use_bias:
            output = K.bias_add(output, bias)
        if activation is not None:
            output = activations.get(activation)(output)
        return output


class APPNP(GraphConv):
    """
    A graph convolutional layer implementing the APPNP operator, as presented by
    [Klicpera et al. (2019)](https://arxiv.org/abs/1810.05997).

    **Mode**: single, mixed, batch.

    **Input**

    - node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - Normalized Laplacian of shape `(n_nodes, n_nodes)` (with optional
    `batch` dimension); see `spektral.utils.convolution.localpooling_filter`.

    **Output**

    - node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `alpha`: teleport probability during propagation;
    - `propagations`: number of propagation steps;
    - `mlp_hidden`: list of integers, number of hidden units for each hidden
    layer in the MLP (if None, the MLP has only one layer);
    - `mlp_activation`: activation for the MLP layers;
    - `dropout_rate`: dropout rate for Laplacian and MLP layers;
    - `activation`: activation function to use;
    - `use_bias`: whether to add a bias to the linear transformation;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.

    **Usage**
    ```py
    # Load data
    A, X, _, _, _, _, _ = citation.load_data('cora')

    # Preprocessing operations
    fltr = utils.localpooling_filter(A)

    # Model definition
    X_in = Input(shape=(F, ))
    fltr_in = Input((N, ))
    output = APPNP(channels)([X_in, fltr_in])
    ```
    """

    def __init__(self,
                 channels,
                 alpha=0.2,
                 propagations=1,
                 mlp_hidden=None,
                 mlp_activation='relu',
                 dropout_rate=0.0,
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
        super().__init__(channels, **kwargs)
        self.channels = channels
        self.mlp_hidden = mlp_hidden if mlp_hidden else []
        self.alpha = alpha
        self.propagations = propagations
        self.mlp_activation = activations.get(mlp_activation)
        self.activation = activations.get(activation)
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.kernels_mlp = []
        self.biases_mlp = []

        # Hidden layers
        input_dim = input_shape[0][-1]
        for i, channels in enumerate(self.mlp_hidden):
            self.kernels_mlp.append(
                self.add_weight(shape=(input_dim, channels),
                                initializer=self.kernel_initializer,
                                name='kernel_mlp_{}'.format(i),
                                regularizer=self.kernel_regularizer,
                                constraint=self.kernel_constraint)
            )
            if self.use_bias:
                self.biases_mlp.append(
                    self.add_weight(shape=(channels,),
                                    initializer=self.bias_initializer,
                                    name='bias_mlp_{}'.format(i),
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
                )
            input_dim = channels

        # Output layer
        self.kernel_out = self.add_weight(shape=(input_dim, self.channels),
                                          initializer=self.kernel_initializer,
                                          name='kernel_mlp_out',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias_out = self.add_weight(shape=(self.channels, ),
                                            initializer=self.bias_initializer,
                                            name='bias_mlp_out',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)

        self.built = True

    def call(self, inputs):
        features = inputs[0]
        fltr = inputs[1]

        # Compute MLP hidden features
        for i in range(len(self.kernels_mlp)):
            features = Dropout(self.dropout_rate)(features)
            features = K.dot(features, self.kernels_mlp[i])
            if self.use_bias:
                features += self.biases_mlp[i]
            if self.mlp_activation is not None:
                features = self.mlp_activation(features)

        # Compute MLP output
        mlp_out = K.dot(features, self.kernel_out)
        if self.use_bias:
            mlp_out += self.bias_out

        # Propagation
        Z = mlp_out
        for k in range(self.propagations):
            Z = (1 - self.alpha) * filter_dot(fltr, Z) + self.alpha * mlp_out

        if self.activation is not None:
            output = self.activation(Z)
        else:
            output = Z
        return output

    def get_config(self):
        config = {
            'channels': self.channels,
            'alpha': self.alpha,
            'propagations': self.propagations,
            'mlp_hidden': self.mlp_hidden,
            'mlp_activation': activations.serialize(self.mlp_activation),
            'activation': activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GINConv(GraphConv):
    """
    A Graph Isomorphism Network (GIN) as presented by
    [Xu et al. (2018)](https://arxiv.org/abs/1810.00826).

    **Mode**: single.

    This layer computes for each node \(i\):
    $$
        Z_i = \\textrm{MLP} ( (1 + \\epsilon) \\cdot X_i + \\sum\\limits_{j \\in \\mathcal{N}(i)} X_j)
    $$
    where \(X\) is the node features matrix and \(\\textrm{MLP}\) is a
    multi-layer perceptron.

    **Input**

    - Node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - Normalized and rescaled Laplacian of shape `(n_nodes, n_nodes)` (with
    optional `batch` dimension);

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `mlp_channels`: integer, number of channels in the inner MLP;
    - `n_hidden_layers`: integer, number of hidden layers in the MLP (default 0)
    - `epsilon`: unnamed parameter, see
    [Xu et al. (2018)](https://arxiv.org/abs/1810.00826), and the equation above.
    This parameter can be learned by setting `epsilon=None`, or it can be set
    to a constant value, which is what happens by default (0). In practice, it
    is safe to leave it to 0.
    - `mlp_activation`: activation function for the MLP,
    - `activation`: activation function to use;
    - `use_bias`: whether to add a bias to the linear transformation;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.

    **Usage**

    ```py
    # Load data
    A, X, _, _, _, _, _, _ = citation.load_data('cora')

    # Preprocessing operations
    fltr = utils.normalized_laplacian(A)
    fltr = utils.rescale_laplacian(X, lmax=2)

    # Model definition
    X_in = Input(shape=(F, ))
    fltr_in = Input((N, ), sparse=True)
    output = GINConv(channels)([X_in, fltr_in])
    ```
    """

    def __init__(self,
                 channels,
                 mlp_channels=16,
                 n_hidden_layers=0,
                 epsilon=None,
                 mlp_activation='relu',
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
        super().__init__(channels, **kwargs)
        self.channels = channels
        self.channels_hid = mlp_channels
        self.extra_hidden_layers = n_hidden_layers
        self.epsilon = epsilon
        self.hidden_activation = activations.get(mlp_activation)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]

        self.kernel_in = self.add_weight(shape=(input_dim, self.channels_hid),
                                         initializer=self.kernel_initializer,
                                         name='kernel_in',
                                         regularizer=self.kernel_regularizer,
                                         constraint=self.kernel_constraint)

        self.kernel_out = self.add_weight(shape=(self.channels_hid, self.channels),
                                          initializer=self.kernel_initializer,
                                          name='kernel_out',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias_in = self.add_weight(shape=(self.channels_hid,),
                                           initializer=self.bias_initializer,
                                           name='bias_in',
                                           regularizer=self.bias_regularizer,
                                           constraint=self.bias_constraint)

            self.bias_out = self.add_weight(shape=(self.channels,),
                                            initializer=self.bias_initializer,
                                            name='bias_out',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)

        if self.epsilon == None:
            self.eps = self.add_weight(shape=(1,),
                                       initializer=self.bias_initializer,
                                       name='eps')
        else:
            self.eps = K.constant(self.epsilon)

        # Additional hidden layers
        if self.extra_hidden_layers > 0:
            self.kernels_hid = []
            self.biases_hid = []
            for k in range(self.extra_hidden_layers):
                self.kernels_hid.append(self.add_weight(shape=(self.channels_hid, self.channels_hid),
                                                        initializer=self.kernel_initializer,
                                                        name='kernel_hid_{}'.format(k),
                                                        regularizer=self.kernel_regularizer,
                                                        constraint=self.kernel_constraint))
                if self.use_bias:
                    self.biases_hid.append(self.add_weight(shape=(self.channels_hid,),
                                                           initializer=self.bias_initializer,
                                                           name='bias_hid_{}'.format(k),
                                                           regularizer=self.bias_regularizer,
                                                           constraint=self.bias_constraint))

        self.built = True

    def call(self, inputs):
        features = inputs[0]
        fltr = inputs[1]

        if not K.is_sparse(fltr):
            fltr = tf.contrib.layers.dense_to_sparse(fltr)

        # Input layer
        features_neigh = tf.segment_sum(tf.gather(features, fltr.indices[:, -1]), fltr.indices[:, -2])
        hidden = (1.0 + self.eps) * features + features_neigh
        hidden = K.dot(hidden, self.kernel_in)
        if self.use_bias:
            hidden = K.bias_add(hidden, self.bias_in)
        if self.hidden_activation is not None:
            hidden = self.hidden_activation(hidden)

        # More hidden layers (optional)
        for k in range(self.extra_hidden_layers):
            hidden = K.dot(hidden, self.kernels_hid[k])
            if self.use_bias:
                hidden = K.bias_add(hidden, self.biases_hid[k])
            if self.hidden_activation is not None:
                hidden = self.hidden_activation(hidden)

        # Output layer
        output = K.dot(hidden, self.kernel_out)
        if self.use_bias:
            output = K.bias_add(output, self.bias_out)
        if self.activation is not None:
            output = self.activation(output)

        return output
