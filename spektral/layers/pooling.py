from keras import backend as K
from keras import regularizers, constraints, initializers
from keras.backend import tf
from keras.engine import Layer


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
            return input_shape[0]

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
            return tf.segment_sum(X, I)
        else:
            return K.sum(X, axis=-2, keepdims=(self.data_mode == 'single'))

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
            return (1,) + input_shape[-1:]
        elif self.data_mode == 'batch':
            return input_shape[:-2] + input_shape[-1:]
        else:
            return input_shape[0]

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

