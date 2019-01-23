from keras import backend as K
from keras.engine import Layer
from keras import regularizers, constraints, initializers


class GlobalAttentionPool(Layer):
    """
    A gated attention global pooling layer as presented by
    [Li et al. (2017)](https://arxiv.org/abs/1511.05493).
    Note that this layer assumes the `'channels_last'` data format, and cannot
    be used otherwise.

    **Mode**: single, batch.

    **Input**

    - node features of shape `(batch, num_nodes, num_features)`, depending on
    the mode;

    **Output**

    - a pooled feature matrix of shape `(batch, channels)`;

    **Arguments**

    - `channels`: integer, number of output channels;
    - `kernel_regularizer`: regularization applied to the gating networks;  

    **Usage**

    ```py
    X = Input(shape=(num_nodes, num_features))
    Z = GlobalAttentionPool(channels)(X)
    ```
        """
    def __init__(self, channels=32, kernel_regularizer=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GlobalAttentionPool, self).__init__(**kwargs)
        self.channels = channels
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.lg_kernel = self.add_weight('LG_kernel',
                                         (input_shape[-1], self.channels),
                                         initializer='glorot_uniform',
                                         regularizer=self.kernel_regularizer)
        self.lg_bias = self.add_weight('LG_bias',
                                       (self.channels,),
                                       initializer='zeros')
        self.attn_kernel = self.add_weight('attn_kernel',
                                           (input_shape[-1], self.channels),
                                           initializer='glorot_uniform',
                                           regularizer=self.kernel_regularizer)
        self.attn_bias = self.add_weight('attn_bias',
                                         (self.channels,),
                                         initializer='zeros')
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

    **Mode**: single, batch.

    **Input**

    - node features of shape `(batch, num_nodes, num_features)`;

    **Output**

    - a pooled feature matrix of shape `(batch, num_features)`;

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

