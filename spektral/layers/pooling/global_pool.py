import tensorflow as tf
from tensorflow.keras import backend as K, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, Dense

from spektral.layers import ops


class GlobalPooling(Layer):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.supports_masking = True
        self.pooling_op = None
        self.batch_pooling_op = None

    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = 'disjoint'
        else:
            if len(input_shape) == 2:
                self.data_mode = 'single'
            else:
                self.data_mode = 'batch'
        super().build(input_shape)

    def call(self, inputs):
        if self.data_mode == 'disjoint':
            X = inputs[0]
            I = inputs[1]
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X = inputs

        if self.data_mode == 'disjoint':
            return self.pooling_op(X, I)
        else:
            return self.batch_pooling_op(X, axis=-2, keepdims=(self.data_mode == 'single'))

    def compute_output_shape(self, input_shape):
        if self.data_mode == 'single':
            return (1,) + input_shape[-1:]
        elif self.data_mode == 'batch':
            return input_shape[:-2] + input_shape[-1:]
        else:
            # Input shape is a list of shapes for X and I
            return input_shape[0]

    def get_config(self):
        return super().get_config()


class GlobalSumPool(GlobalPooling):
    """
    A global sum pooling layer. Pools a graph by computing the sum of its node
    features.

    **Mode**: single, disjoint, mixed, batch.

    **Input**

    - Node features of shape `([batch], N, F)`;
    - Graph IDs of shape `(N, )` (only in disjoint mode);

    **Output**

    - Pooled node features of shape `(batch, F)` (if single mode, shape will
    be `(1, F)`).

    **Arguments**

    None.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pooling_op = tf.math.segment_sum
        self.batch_pooling_op = tf.reduce_sum


class GlobalAvgPool(GlobalPooling):
    """
    An average pooling layer. Pools a graph by computing the average of its node
    features.

    **Mode**: single, disjoint, mixed, batch.

    **Input**

    - Node features of shape `([batch], N, F)`;
    - Graph IDs of shape `(N, )` (only in disjoint mode);

    **Output**

    - Pooled node features of shape `(batch, F)` (if single mode, shape will
    be `(1, F)`).

    **Arguments**

    None.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pooling_op = tf.math.segment_mean
        self.batch_pooling_op = tf.reduce_mean


class GlobalMaxPool(GlobalPooling):
    """
    A max pooling layer. Pools a graph by computing the maximum of its node
    features.

    **Mode**: single, disjoint, mixed, batch.

    **Input**

    - Node features of shape `([batch], N, F)`;
    - Graph IDs of shape `(N, )` (only in disjoint mode);

    **Output**

    - Pooled node features of shape `(batch, F)` (if single mode, shape will
    be `(1, F)`).

    **Arguments**

    None.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pooling_op = tf.math.segment_max
        self.batch_pooling_op = tf.reduce_max


class GlobalAttentionPool(GlobalPooling):
    r"""
    A gated attention global pooling layer as presented by
    [Li et al. (2017)](https://arxiv.org/abs/1511.05493).

    This layer computes:
    $$
        \X' = \sum\limits_{i=1}^{N} (\sigma(\X \W_1 + \b_1) \odot (\X \W_2 + \b_2))_i
    $$
    where \(\sigma\) is the sigmoid activation function.

    **Mode**: single, disjoint, mixed, batch.

    **Input**

    - Node features of shape `([batch], N, F)`;
    - Graph IDs of shape `(N, )` (only in disjoint mode);

    **Output**

    - Pooled node features of shape `(batch, channels)` (if single mode,
    shape will be `(1, channels)`).

    **Arguments**

    - `channels`: integer, number of output channels;
    - `bias_initializer`: initializer for the bias vectors;
    - `kernel_regularizer`: regularization applied to the kernel matrices;
    - `bias_regularizer`: regularization applied to the bias vectors;
    - `kernel_constraint`: constraint applied to the kernel matrices;
    - `bias_constraint`: constraint applied to the bias vectors.
    """

    def __init__(self,
                 channels,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        super().build(input_shape)
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint
        )
        self.features_layer = Dense(self.channels,
                                    name='features_layer',
                                    **layer_kwargs)
        self.attention_layer = Dense(self.channels,
                                     activation='sigmoid',
                                     name='attn_layer',
                                     **layer_kwargs)
        self.built = True

    def call(self, inputs):
        if self.data_mode == 'disjoint':
            X, I = inputs
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X = inputs
        inputs_linear = self.features_layer(X)
        attn = self.attention_layer(X)
        masked_inputs = inputs_linear * attn
        if self.data_mode in {'single', 'batch'}:
            output = K.sum(masked_inputs, axis=-2,
                           keepdims=self.data_mode == 'single')
        else:
            output = tf.math.segment_sum(masked_inputs, I)

        return output

    def compute_output_shape(self, input_shape):
        if self.data_mode == 'single':
            return (1,) + (self.channels,)
        elif self.data_mode == 'batch':
            return input_shape[:-2] + (self.channels,)
        else:
            output_shape = input_shape[0]
            output_shape = output_shape[:-1] + (self.channels,)
            return output_shape

    def get_config(self):
        config = {
            'channels': self.channels,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalAttnSumPool(GlobalPooling):
    r"""
    A node-attention global pooling layer. Pools a graph by learning attention
    coefficients to sum node features.

    This layer computes:
    $$
        \alpha = \textrm{softmax}( \X \a); \\
        \X' = \sum\limits_{i=1}^{N} \alpha_i \cdot \X_i
    $$
    where \(\a \in \mathbb{R}^F\) is a trainable vector. Note that the softmax
    is applied across nodes, and not across features.

    **Mode**: single, disjoint, mixed, batch.

    **Input**

    - Node features of shape `([batch], N, F)`;
    - Graph IDs of shape `(N, )` (only in disjoint mode);

    **Output**

    - Pooled node features of shape `(batch, F)` (if single mode, shape will
    be `(1, F)`).

    **Arguments**

    - `attn_kernel_initializer`: initializer for the attention weights;
    - `attn_kernel_regularizer`: regularization applied to the attention kernel
    matrix;
    - `attn_kernel_constraint`: constraint applied to the attention kernel
    matrix;
    """

    def __init__(self,
                 attn_kernel_initializer='glorot_uniform',
                 attn_kernel_regularizer=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.attn_kernel_initializer = initializers.get(
            attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(
            attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = 'disjoint'
            F = input_shape[0][-1]
        else:
            if len(input_shape) == 2:
                self.data_mode = 'single'
            else:
                self.data_mode = 'batch'
            F = input_shape[-1]
        # Attention kernels
        self.attn_kernel = self.add_weight(shape=(F, 1),
                                           initializer=self.attn_kernel_initializer,
                                           regularizer=self.attn_kernel_regularizer,
                                           constraint=self.attn_kernel_constraint,
                                           name='attn_kernel')
        self.built = True

    def call(self, inputs):
        if self.data_mode == 'disjoint':
            X, I = inputs
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X = inputs
        attn_coeff = K.dot(X, self.attn_kernel)
        attn_coeff = K.squeeze(attn_coeff, -1)
        attn_coeff = K.softmax(attn_coeff)
        if self.data_mode == 'single':
            output = K.dot(attn_coeff[None, ...], X)
        elif self.data_mode == 'batch':
            output = K.batch_dot(attn_coeff, X)
        else:
            output = attn_coeff[:, None] * X
            output = tf.math.segment_sum(output, I)

        return output

    def get_config(self):
        config = {
            'attn_kernel_initializer': self.attn_kernel_initializer,
            'attn_kernel_regularizer': self.attn_kernel_regularizer,
            'attn_kernel_constraint': self.attn_kernel_constraint,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SortPool(Layer):
    r"""
    A SortPool layer as described by
    [Zhang et al](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf).
    This layers takes a graph signal \(\mathbf{X}\) and returns the topmost k
    rows according to the last column.
    If \(\mathbf{X}\) has less than k rows, the result is zero-padded to k.
    
    **Mode**: single, disjoint, batch.
    
    **Input**

    - Node features of shape `([batch], N, F)`;
    - Graph IDs of shape `(N, )` (only in disjoint mode);

    **Output**

    - Pooled node features of shape `(batch, k, F)` (if single mode, shape will
    be `(1, k, F)`).

    **Arguments**

    - `k`: integer, number of nodes to keep;
    """

    def __init__(self, k):
        super(SortPool, self).__init__()
        k = int(k)
        if k <= 0:
            raise ValueError("K must be a positive integer")
        self.k = k

    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = 'disjoint'
            self.F = input_shape[0][-1]
        else:
            if len(input_shape) == 2:
                self.data_mode = 'single'
            else:
                self.data_mode = 'batch'
            self.F = input_shape[-1]

    def call(self, inputs):
        if self.data_mode == 'disjoint':
            X, I = inputs
            X = ops.disjoint_signal_to_batch(X, I)
        else:
            X = inputs
            if self.data_mode == 'single':
                X = tf.expand_dims(X, 0)

        N = tf.shape(X)[-2]
        sort_perm = tf.argsort(X[..., -1], direction='DESCENDING')
        X_sorted = tf.gather(X, sort_perm, axis=-2, batch_dims=1)

        def truncate():
            _X_out = X_sorted[..., : self.k, :]
            return _X_out

        def pad():
            padding = [[0, 0], [0, self.k - N], [0, 0]]
            _X_out = tf.pad(X_sorted, padding)
            return _X_out

        X_out = tf.cond(tf.less_equal(self.k, N), truncate, pad)

        if self.data_mode == 'single':
            X_out = tf.squeeze(X_out, [0])
            X_out.set_shape((self.k, self.F))
        elif self.data_mode == 'batch' or self.data_mode == 'disjoint':
            X_out.set_shape((None, self.k, self.F))

        return X_out

    def get_config(self):
        config = {
            'k': self.k,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if self.data_mode == 'single':
            return self.k, self.F
        elif self.data_mode == 'batch' or self.data_mode == 'disjoint':
            return input_shape[0], self.k, self.F
