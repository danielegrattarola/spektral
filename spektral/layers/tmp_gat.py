import tensorflow as tf


class DenseMultiHead(tf.keras.layers.Layer):
    """A Dense Layer using multihead kernel with tf.einsum implementation.
  Attributes:
    num_attention_heads: An integer, number of attention heads for each
      multihead attention layer.
    size_per_head: An integer, hidden size per attention head.
    hidden_size: An integer, dimension of the hidden layer.
    kernel_initializer: An initializer for the kernel weight.
    bias_initializer: An initializer for the bias.
    activation: An activation function to use. If nothing is specified, no
      activation is applied.
    use_bias: A bool, whether the layer uses a bias.
  """

    def __init__(
        self,
        num_attention_heads=12,
        size_per_head=72,
        kernel_initializer=None,
        bias_initializer="zeros",
        activation=None,
        use_bias=True,
        **kwargs
    ):
        """Inits DenseMultiHead."""
        super(DenseMultiHead, self).__init__(**kwargs)

        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.hidden_size = num_attention_heads * size_per_head
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        """Implements build() for the layer."""
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "Unable to build `DenseMultiHead` layer with non-floating "
                "point (and non-complex) dtype %s" % (dtype,)
            )
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The last dimension of the inputs to `DenseMultiHead` "
                "should be defined. Found `None`."
            )
        self.last_dim = tf.compat.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(
            min_ndim=3, axes={-1: self.last_dim}
        )


        self.kernel = self.add_weight(
            "kernel",
            shape=[self.last_dim, self.num_attention_heads, self.size_per_head],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[self.num_attention_heads, self.size_per_head],
                initializer=self.bias_initializer,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None

        super().build(input_shape)

    def call(self, inputs):
        """Implements ``call()`` for DenseMultiHead.
    Args:
      inputs: A float tensor of shape [batch_size, sequence_length, hidden_size]
        when output_projection is False, otherwise a float tensor of shape
        [batch_size, sequence_length, num_heads, dim_per_head].
    Returns:
      The projected tensor with shape [batch_size, sequence_length, num_heads,
        dim_per_head] when output_projection is False, otherwise [batch_size,
        sequence_length, hidden_size].
    """

        kernel = self.kernel
        bias = self.bias

        ret = tf.einsum("...ND,DHF->...NHF", inputs, kernel)

        if self.use_bias:
            ret += bias

        if self.activation is not None:
            return self.activation(ret)

        return ret


class GraphAttention(GraphConv):
    r"""
    A graph attention layer (GAT) as presented by
    [Velickovic et al. (2017)](https://arxiv.org/abs/1710.10903).

    **Mode**: single, mixed, batch.

    **This layer expects dense inputs.**
    
    This layer computes a convolution similar to `layers.GraphConv`, but
    uses the attention mechanism to weight the adjacency matrix instead of
    using the normalized Laplacian:
    $$
        \Z = \mathbf{\alpha}\X\W + \b
    $$
    where
    $$
        \mathbf{\alpha}_{ij} =
            \frac{
                \exp\left(
                    \mathrm{LeakyReLU}\left(
                        \a^{\top} [(\X\W)_i \, \| \, (\X\W)_j]
                    \right)
                \right)
            }
            {\sum\limits_{k \in \mathcal{N}(i) \cup \{ i \}}
                \exp\left(
                    \mathrm{LeakyReLU}\left(
                        \a^{\top} [(\X\W)_i \, \| \, (\X\W)_k]
                    \right)
                \right)
            }
    $$
    where \(\a \in \mathbb{R}^{2F'}\) is a trainable attention kernel.
    Dropout is also applied to \(\alpha\) before computing \(\Z\).
    Parallel attention heads are computed in parallel and their results are
    aggregated by concatenation or average.

    **Input**

    - Node features of shape `([batch], N, F)`;
    - Binary adjacency matrix of shape `([batch], N, N)`;

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`;
    - if `return_attn_coef=True`, a list with the attention coefficients for
    each attention head. Each attention coefficient matrix has shape
    `([batch], N, N)`.
    
    **Arguments**
    
    - `channels`: number of output channels;
    - `attn_heads`: number of attention heads to use;
    - `concat_heads`: bool, whether to concatenate the output of the attention
     heads instead of averaging;
    - `dropout_rate`: internal dropout rate for attention coefficients;
    - `return_attn_coef`: if True, return the attention coefficients for
    the given input (one N x N matrix for each head).
    - `activation`: activation function to use;
    - `use_bias`: whether to add a bias to the linear transformation;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `attn_kernel_initializer`: initializer for the attention kernels;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `attn_kernel_regularizer`: regularization applied to the attention kernels;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `attn_kernel_constraint`: constraint applied to the attention kernels;
    - `bias_constraint`: constraint applied to the bias vector.

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

        if concat_heads:
            # Output will have shape (..., attention_heads * channels)
            self.output_dim = self.channels * self.attn_heads
        else:
            # Output will have shape (..., channels)
            self.output_dim = self.channels


    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]


        self.Wh_dense = DenseMultiHead(
            self.attn_heads,
            self.channels,
            kernel_initializer="glorot_uniform",
            use_bias=False,
            name="query",
        )

        # self.key_dense = DenseMultiHead(
        #     self.num_heads,
        #     self.size_per_head,
        #     kernel_initializer="glorot_uniform",
        #     use_bias=False,
        #     name="key",
        # )
        # self.value_dense = DenseMultiHead(
        #     self.num_heads,
        #     self.size_per_head,
        #     kernel_initializer="glorot_uniform",
        #     use_bias=False,
        #     name="value",
        # )
        # self.projection_kernel = self.add_weight(
        #     "projection_kernel",
        #     shape=[self.num_heads, self.size_per_head, output_size],
        #     initializer="glorot_uniform",
        #     dtype=self.dtype,
        #     trainable=True,
        # )

        # self.dropout = Dropout(self.dropout_rate)

        self.built = True

        # Initialize weights for each attention head
        # for head in range(self.attn_heads):
        #     # Layer kernel
        #     kernel = self.add_weight(shape=(input_dim, self.channels),
        #                              initializer=self.kernel_initializer,
        #                              regularizer=self.kernel_regularizer,
        #                              constraint=self.kernel_constraint,
        #                              name='kernel_{}'.format(head))
        #     self.kernels.append(kernel)

        #     # Layer bias
        #     if self.use_bias:
        #         bias = self.add_weight(shape=(self.channels,),
        #                                initializer=self.bias_initializer,
        #                                regularizer=self.bias_regularizer,
        #                                constraint=self.bias_constraint,
        #                                name='bias_{}'.format(head))
        #         self.biases.append(bias)

        #     # Attention kernels
        #     attn_kernel_self = self.add_weight(shape=(self.channels, 1),
        #                                        initializer=self.attn_kernel_initializer,
        #                                        regularizer=self.attn_kernel_regularizer,
        #                                        constraint=self.attn_kernel_constraint,
        #                                        name='attn_kernel_self_{}'.format(head))
        #     attn_kernel_neighs = self.add_weight(shape=(self.channels, 1),
        #                                          initializer=self.attn_kernel_initializer,
        #                                          regularizer=self.attn_kernel_regularizer,
        #                                          constraint=self.attn_kernel_constraint,
        #                                          name='attn_kernel_neigh_{}'.format(head))
        #     self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])

    

    def call(self, inputs):
        X = inputs[0]
        A = inputs[1]

        features = self.Wh_dense(query)

        print(features.shape)
        exit()

        ######################
        # ^^^ code so far ^^^ 
        ######################

        key = self.key_dense(key)
        value = self.value_dense(value)
        projection = self.projection_kernel

        depth = tf.cast(tf.shape(query)[-1], tf.float32)

        query /= tf.sqrt(depth)

        # Calculate dot product attention
        logits = tf.einsum("...TNH,...FNH->...NFT", key, query)

        # apply mask
        if mask is not None:
            logits += mask

        # Note that softmax internally performs math operations using float32
        # for numeric stability. When training with float16, we keep the input
        # and output in float16 for better performance.
        attention = tf.nn.softmax(logits, name="attention_weights")

        concated_output = tf.einsum("...NFT,...TNH->...FNH", attention, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done --> [batch_size, length, hidden_size]
        attention_output = tf.einsum("...FNH,NHD->...FD", concated_output, projection)

        return attention_output

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



