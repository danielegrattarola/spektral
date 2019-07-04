from keras import backend as K
from keras import regularizers, constraints, initializers
from keras.backend import tf
from keras.engine import Layer


################################################################################
# Pooling layers
################################################################################
from spektral.layers.ops import top_k


class TopKPool(Layer):
    """
    A gPool/Top-K layer as presented by
    [Gao & Ji (2017)](http://proceedings.mlr.press/v97/gao19a/gao19a.pdf) and
    [Cangea et al.](https://arxiv.org/abs/1811.01287).

    This layer computes the following operations:

    $$
    y = \\cfrac{Xp}{\\| p \\|}; \\;\\;\\;\\;
    \\textrm{idx} = \\textrm{rank}(y, k); \\;\\;\\;\\;
    \\bar X = (X \\odot \\textrm{tanh}(y))_{\\textrm{idx}}; \\;\\;\\;\\;
    \\bar A = A^2_{\\textrm{idx}, \\textrm{idx}}
    $$

    where \( \\textrm{rank}(y, k) \) returns the indices of the top k values of
    \( y \), and \( p \) is a learnable parameter vector of size \(F\).
    Note that the the gating operation \( \\textrm{tanh}(y) \) (Cangea et al.)
    can be replaced with a sigmoid (Gao & Ji). The original paper by Gao & Ji
    used a tanh as well, but was later updated to use a sigmoid activation.

    Due to the lack of sparse-sparse matrix multiplication support, this layer
    temporarily makes the adjacency matrix dense in order to compute \(A^2\)
    (needed to preserve connectivity after pooling).
    **If memory is not an issue, considerable speedups can be achieved by using
    dense graphs directly.
    Converting a graph from dense to sparse and viceversa is a costly operation.**

    **Mode**: single, graph batch.

    **Input**

    - node features of shape `(n_nodes, n_features)`;
    - adjacency matrix of shape `(n_nodes, n_nodes)`;
    - (optional) graph IDs of shape `(n_nodes, )` (graph batch mode);

    **Output**

    - reduced node features of shape `(n_graphs * k, n_features)`;
    - reduced adjacency matrix of shape `(n_graphs * k, n_graphs * k)`;
    - reduced graph IDs with shape `(n_graphs * k, )` (graph batch mode);

    **Arguments**

    - `ratio`: float between 0 and 1, ratio of nodes to keep in each graph;
    - `return_mask`: boolean, whether to return the binary mask used for pooling;
    - `sigmoid_gating`: boolean, use a sigmoid gating activation instead of a
        tanh;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    """

    def __init__(self, ratio,
                 return_mask=False,
                 sigmoid_gating=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(TopKPool, self).__init__(**kwargs)
        self.ratio = ratio  # Ratio of nodes to keep in each graph
        self.return_mask = return_mask
        self.sigmoid_gating = sigmoid_gating
        self.gating_op = K.sigmoid if self.sigmoid_gating else K.tanh
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        self.F = input_shape[0][-1]
        self.N = input_shape[0][0]
        self.kernel = self.add_weight(shape=(self.F, 1),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.top_k_var = tf.Variable(0.0, validate_shape=False)
        super(TopKPool, self).build(input_shape)

    def call(self, inputs):
        if len(inputs) == 3:
            X, A, I = inputs
            self.data_mode = 'graph'
        else:
            X, A = inputs
            I = tf.zeros(tf.shape(X)[:1], dtype=tf.int32)
            self.data_mode = 'single'

        A_is_sparse = K.is_sparse(A)

        # Get mask
        y = K.dot(X, K.l2_normalize(self.kernel))
        N = K.shape(X)[-2]
        indices = top_k(y[:, 0], I, self.ratio, self.top_k_var)
        mask = tf.scatter_nd(tf.expand_dims(indices, 1), tf.ones_like(indices), (N,))

        # Multiply X and y to make layer differentiable
        features = X * self.gating_op(y)

        axis = 0 if len(K.int_shape(A)) == 2 else 1  # Cannot use negative axis in tf.boolean_mask
        # Reduce X
        X_pooled = tf.boolean_mask(features, mask, axis=axis)

        # Compute A^2
        if A_is_sparse:
            A_dense = tf.sparse.to_dense(A)
        else:
            A_dense = A
        A_squared = K.dot(A, A_dense)

        # Reduce A
        A_pooled = tf.boolean_mask(A_squared, mask, axis=axis)
        A_pooled = tf.boolean_mask(A_pooled, mask, axis=axis + 1)
        if A_is_sparse:
            A_pooled = tf.contrib.layers.dense_to_sparse(A_pooled)

        output = [X_pooled, A_pooled]

        # Reduce I
        if self.data_mode == 'graph':
            I_pooled = tf.boolean_mask(I[:, None], mask)[:, 0]
            output.append(I_pooled)

        if self.return_mask:
            output.append(mask)

        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_mask:
            output_shape += [(input_shape[0][:-1])]
        return output_shape

    def get_config(self):
        config = {
            'ratio': self.ratio,
            'return_mask': self.return_mask,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(TopKPool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


################################################################################
# Global pooling layers
################################################################################
class GlobalSumPool(Layer):
    """
    A global sum pooling layer. Pools a graph by computing the sum of its node
    features.

    **Mode**: single, mixed, batch, graph batch.

    **Input**

    - node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - (optional) graph IDs of shape `(n_nodes, )` (graph batch mode);

    **Output**

    - tensor like node features, but without node dimension (except for single
    mode, where the node dimension is preserved and set to 1).

    **Arguments**

    None.

    """
    def __init__(self, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GlobalSumPool, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = 'graph'
        else:
            if len(input_shape) == 2:
                self.data_mode = 'single'
            else:
                self.data_mode = 'batch'
        super(GlobalSumPool, self).build(input_shape)

    def call(self, inputs):
        if self.data_mode == 'graph':
            X = inputs[0]
            I = inputs[1]
        else:
            X = inputs

        if self.data_mode == 'graph':
            return tf.segment_sum(X, I)
        else:
            return K.sum(X, axis=-2, keepdims=(self.data_mode == 'single'))

    def compute_output_shape(self, input_shape):
        if self.data_mode == 'single':
            return (1, ) + input_shape[-1:]
        elif self.data_mode == 'batch':
            return input_shape[:-2] + input_shape[-1:]
        else:
            return input_shape[0]  # Input shape is a list of shapes for X and I

    def get_config(self):
        return super(GlobalSumPool, self).get_config()


class GlobalAvgPool(Layer):
    """
    An average pooling layer. Pools a graph by computing the average of its node
    features.

    **Mode**: single, mixed, batch, graph batch.

    **Input**

    - node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - (optional) graph IDs of shape `(n_nodes, )` (graph batch mode);

    **Output**

    - tensor like node features, but without node dimension (except for single
    mode, where the node dimension is preserved and set to 1).

    **Arguments**

    None.
    """
    def __init__(self, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GlobalAvgPool, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = 'graph'
        else:
            if len(input_shape) == 2:
                self.data_mode = 'single'
            else:
                self.data_mode = 'batch'
        super(GlobalAvgPool, self).build(input_shape)

    def call(self, inputs):
        if self.data_mode == 'graph':
            X = inputs[0]
            I = inputs[1]
        else:
            X = inputs

        if self.data_mode == 'graph':
            return tf.segment_mean(X, I)
        else:
            return K.mean(X, axis=-2, keepdims=(self.data_mode == 'single'))

    def compute_output_shape(self, input_shape):
        if self.data_mode == 'single':
            return (1, ) + input_shape[-1:]
        elif self.data_mode == 'batch':
            return input_shape[:-2] + input_shape[-1:]
        else:
            return input_shape[0]

    def get_config(self):
        return super(GlobalAvgPool, self).get_config()


class GlobalMaxPool(Layer):
    """
    A max pooling layer. Pools a graph by computing the maximum of its node
    features.

    **Mode**: single, mixed, batch, graph batch.

    **Input**

    - node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - (optional) graph IDs of shape `(n_nodes, )` (graph batch mode);

    **Output**

    - tensor like node features, but without node dimension (except for single
    mode, where the node dimension is preserved and set to 1).

    **Arguments**

    None.
    """
    def __init__(self, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GlobalMaxPool, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = 'graph'
        else:
            if len(input_shape) == 2:
                self.data_mode = 'single'
            else:
                self.data_mode = 'batch'
        super(GlobalMaxPool, self).build(input_shape)

    def call(self, inputs):
        if self.data_mode == 'graph':
            X = inputs[0]
            I = inputs[1]
        else:
            X = inputs

        if self.data_mode == 'graph':
            return tf.segment_max(X, I)
        else:
            return K.max(X, axis=-2, keepdims=(self.data_mode == 'single'))

    def compute_output_shape(self, input_shape):
        if self.data_mode == 'single':
            return (1, ) + input_shape[-1:]
        elif self.data_mode == 'batch':
            return input_shape[:-2] + input_shape[-1:]
        else:
            return input_shape[0]

    def get_config(self):
        return super(GlobalMaxPool, self).get_config()


class GlobalAttentionPool(Layer):
    """
    A gated attention global pooling layer as presented by
    [Li et al. (2017)](https://arxiv.org/abs/1511.05493).

    **Mode**: single, mixed, batch, graph batch.

    **Input**

    - node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - (optional) graph IDs of shape `(n_nodes, )` (graph batch mode);

    **Output**

    - tensor like node features, but without node dimension (except for single
    mode, where the node dimension is preserved and set to 1), and last
    dimension changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.
    """
    def __init__(self, channels=32,
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
        super(GlobalAttentionPool, self).__init__(**kwargs)
        self.channels = channels
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = 'graph'
            F = input_shape[0][-1]
        else:
            if len(input_shape) == 2:
                self.data_mode = 'single'
            else:
                self.data_mode = 'batch'
            F = input_shape[-1]
        self.lg_kernel = self.add_weight(shape=(F, self.channels),
                                         name='LG_kernel',
                                         initializer=self.kernel_initializer,
                                         regularizer=self.kernel_regularizer,
                                         constraint=self.kernel_constraint)
        self.lg_bias = self.add_weight(shape=(self.channels, ),
                                       name='LG_bias',
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint)
        self.attn_kernel = self.add_weight(shape=(F, self.channels),
                                           name='attn_kernel',
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        self.attn_bias = self.add_weight(shape=(self.channels,),
                                         name='attn_bias',
                                         initializer=self.bias_initializer,
                                         regularizer=self.bias_regularizer,
                                         constraint=self.bias_constraint)
        self.built = True

    def call(self, inputs):
        if self.data_mode == 'graph':
            X, I = inputs
        else:
            X = inputs
        inputs_linear = K.dot(X, self.lg_kernel) + self.lg_bias
        attn_map = K.dot(X, self.attn_kernel) + self.attn_bias
        attn_map = K.sigmoid(attn_map)
        masked_inputs = inputs_linear * attn_map
        if self.data_mode in {'single', 'batch'}:
            output = K.sum(masked_inputs, axis=-2, keepdims=self.data_mode=='single')
        else:
            output = tf.segment_sum(masked_inputs, I)

        return output

    def compute_output_shape(self, input_shape):
        if self.data_mode == 'single':
            return (1,) + (self.channels, )
        elif self.data_mode == 'batch':
            return input_shape[:-2] + (self.channels, )
        else:
            # Input shape is a list of shapes for X and I
            output_shape = input_shape[0]
            output_shape = output_shape[:-1] + (self.channels, )
            return output_shape

    def get_config(self):
        config = {}
        base_config = super(GlobalAttentionPool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalAttnSumPool(Layer):
    """
    A node-attention global pooling layer. Pools a graph by learning attention
    coefficients to sum node features.

    **Mode**: single, mixed, batch, graph batch.

    **Input**

    - node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - (optional) graph IDs of shape `(n_nodes, )` (graph batch mode);

    **Output**

    - tensor like node features, but without node dimension (except for single
    mode, where the node dimension is preserved and set to 1).

    **Arguments**

    - `attn_kernel_initializer`: initializer for the attention kernel matrix;
    - `kernel_regularizer`: regularization applied to the kernel matrix;  
    - `attn_kernel_regularizer`: regularization applied to the attention kernel 
    matrix;
    - `attn_kernel_constraint`: constraint applied to the attention kernel
    matrix;
    """
    def __init__(self,
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 attn_kernel_regularizer=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GlobalAttnSumPool, self).__init__(**kwargs)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = 'graph'
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
        if self.data_mode == 'graph':
            X, I = inputs
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
            output = tf.segment_sum(output, I)

        return output

    def compute_output_shape(self, input_shape):
        if self.data_mode == 'single':
            return (1,) + input_shape[-1:]
        elif self.data_mode == 'batch':
            return input_shape[:-2] + input_shape[-1:]
        else:
            return input_shape[0]

    def get_config(self):
        config = {}
        base_config = super(GlobalAttnSumPool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

