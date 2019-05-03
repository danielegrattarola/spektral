from keras import backend as K
from keras import regularizers, constraints, initializers
from keras.backend import tf
from keras.engine import Layer

from spektral.layers.ops import sparse_bool_mask, matrix_power


class TopKPooling(Layer):
    """
    A top-k / gPool layer as presented by
    [Gao & Ji (2017)](https://openreview.net/forum?id=HJePRoAct7).
    Note that due to the lack of sparse-sparse matrix multiplication, this layer
    temporarily makes the adjacency matrix dense in order to compute \(A^2\)
    (needed to preserve connectivity after pooling).

    **Mode**: single.

    **Input**

    - node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - adjacency matrix of shape `(n_nodes, n_nodes)` (with optional `batch`
    dimension);
    - graph IDs of shape `(n_graphs, )` (graph batch mode);
    - (optional) edge features of shape `(n_nodes, n_nodes, n_edge_features)`
    (with optional `batch` dimension);

    **Output**

    - reduced node features of shape `(k, n_features)` (with optional batch
    dimension);
    - reduced adjacency matrix of shape `(k, k)` (with optional batch
    dimension);
    - reduced graph IDs with shape `(k, )` (graph batch mode);
    - (optional) edge features of shape `(k, k, n_edge_features)`  (with
    optional `batch` dimension);

    **Arguments**

    - `k`: integer, number of nodes to keep;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;

    """
    def __init__(self, k,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(TopKPooling, self).__init__(**kwargs)
        self.k = k  # Number of nodes to keep
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.supports_masking = True

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        F = input_shape[0][-1]
        self.kernel = self.add_weight(shape=(F, 1),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        super(TopKPooling, self).build(input_shape)

    def call(self, inputs):
        X = inputs[0]
        A = inputs[1]
        S = inputs[2]
        if len(inputs) == 4:
            E = inputs[3]
        else:
            E = None
        y = K.dot(X, K.l2_normalize(self.kernel))

        # Get mask
        N = K.shape(X)[-2]
        _, indices = K.tf.nn.top_k(K.squeeze(y, axis=-1), k=self.k, sorted=False)
        mask = tf.scatter_nd(tf.expand_dims(indices, 1), tf.ones_like(indices), (N, ))

        # Multiply X and y to make layer differentiable
        features = X * K.tanh(y)

        axis = 0 if len(K.int_shape(A)) == 2 else 1  # Cannot use negative axis in tf.boolean_mask
        # Reduce X
        X_pooled = tf.boolean_mask(features, mask, axis=axis)
        # Reduce A
        A = matrix_power(A, 2)
        if K.is_sparse(A):
            bool_mask_op = sparse_bool_mask
        else:
            bool_mask_op = tf.boolean_mask
        A_pooled = bool_mask_op(A, mask, axis=axis)
        A_pooled = bool_mask_op(A_pooled, mask, axis=axis + 1)
        # Reduce S
        S_pooled = tf.boolean_mask(S[:, None], mask)[:, 0]

        output = [X_pooled, A_pooled, S_pooled]

        # Reduce E
        if E is not None:
            E_pooled = bool_mask_op(E, mask, axis=axis)
            E_pooled = bool_mask_op(E_pooled, mask, axis=axis + 1)
            output.append(E_pooled)

        return output

    def compute_output_shape(self, input_shape):
        X_shape = input_shape[0]
        A_shape = input_shape[1]
        S_shape = input_shape[2]
        X_shape_out = X_shape[:-2] + (self.k, ) + X_shape[-1:]
        A_shape_out = A_shape[:-2] + (self.k, self.k)
        S_shape_out = S_shape[:-2] + (self.k,)
        output_shape = [X_shape_out, A_shape_out, S_shape_out]
        if len(input_shape) == 4:
            E_shape = input_shape[2]
            E_shape_out = E_shape[:-3] + (self.k, self.k) + E_shape[3:]
            output_shape.append(E_shape_out)
        return output_shape

    def get_config(self):
        config = {
            'k': self.k,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(TopKPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


################################################################################
# Global pooling layers
################################################################################
class GlobalAttentionPool(Layer):
    """
    A gated attention global pooling layer as presented by
    [Li et al. (2017)](https://arxiv.org/abs/1511.05493).
    Note that this layer assumes the `'channels_last'` data format, and cannot
    be used otherwise.

    **Mode**: single, mixed, batch.

    **Input**

    - node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);

    **Output**

    - globally pooled vector of shape `(channels, )` (with optional `batch`
    dimension);

    **Arguments**

    - `channels`: integer, number of output channels;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.

    **Usage**

    ```py
    X = Input(shape=(num_nodes, num_features))
    Z = GlobalAttentionPool(channels)(X)
    ```
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
        self.lg_kernel = self.add_weight(shape=(input_shape[-1], self.channels),
                                         name='LG_kernel',
                                         initializer=self.kernel_initializer,
                                         regularizer=self.kernel_regularizer,
                                         constraint=self.kernel_constraint)
        self.lg_bias = self.add_weight(shape=(self.channels, ),
                                       name='LG_bias',
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint)
        self.attn_kernel = self.add_weight(shape=(input_shape[-1], self.channels),
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
        # Note that the layer assumes the 'channels_last' data format.
        inputs_linear = K.dot(inputs, self.lg_kernel) + self.lg_bias
        attn_map = K.dot(inputs, self.attn_kernel) + self.attn_bias
        attn_map = K.sigmoid(attn_map)
        masked_inputs = inputs_linear * attn_map
        output = K.sum(masked_inputs, 1)
        return output

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            return input_shape[:-1] + (self.channels,)
        else:
            return (input_shape[0], self.channels)

    def get_config(self):
        config = {}
        base_config = super(GlobalAttentionPool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NodeAttentionPool(Layer):
    """
    A node-attention global pooling layer. Pools a graph by learning attention
    coefficients to sum node features.
    Note that this layer assumes the `'channels_last'` data format, and cannot
    be used otherwise.

    **Mode**: single, mixed, batch.

    **Input**

    - node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);

    **Output**

    - globally pooled vector of shape `(channels, )` (with optional `batch`
    dimension);

    **Arguments**

    - `attn_kernel_initializer`: initializer for the attention kernel matrix;
    - `kernel_regularizer`: regularization applied to the kernel matrix;  
    - `attn_kernel_regularizer`: regularization applied to the attention kernel 
    matrix;
    - `attn_kernel_constraint`: constraint applied to the attention kernel
    matrix;

    **Usage**
    ```py
    X = Input(shape=(num_nodes, num_features))
    Z = NodeAttentionPool()(X)
    ```
    """
    def __init__(self,
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 attn_kernel_regularizer=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NodeAttentionPool, self).__init__(**kwargs)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        # Attention kernels
        self.attn_kernel = self.add_weight(shape=(input_shape[-1], 1),
                                           initializer=self.attn_kernel_initializer,
                                           regularizer=self.attn_kernel_regularizer,
                                           constraint=self.attn_kernel_constraint,
                                           name='attn_kernel')

        self.built = True

    def call(self, inputs):
        input_shape = K.int_shape(inputs)
        # Note that the layer assumes the 'channels_last' data format.
        features = K.dot(inputs, self.attn_kernel)
        features = K.squeeze(features, -1)
        attn_coeff = K.softmax(features)  # TODO: maybe sigmoid?
        if len(input_shape) == 2:
            output = K.dot(attn_coeff, inputs)
        else:
            output = K.batch_dot(attn_coeff, inputs)

        return output

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            return input_shape[-1:]
        else:
            return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = {}
        base_config = super(NodeAttentionPool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

