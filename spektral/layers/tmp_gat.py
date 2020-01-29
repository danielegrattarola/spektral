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

    @property
    def kernel_shape(self):
        return [self.last_dim, self.num_attention_heads, self.size_per_head]

    @property
    def bias_shape(self):
        return [self.num_attention_heads, self.size_per_head]

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

        kernel_shape = self.kernel_shape
        bias_shape = self.bias_shape

        self.kernel = self.add_weight(
            "kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=bias_shape,
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

        ret = tf.einsum("abc,cde->abde", inputs, kernel)

        if self.use_bias:
            ret += bias

        if self.activation is not None:
            return self.activation(ret)

        return ret


class GraphAttention(tf.keras.layers.Layer):
    """Multi-headed graph attention layer."""

    def __init__(self, size_per_head, num_heads, output_size=None):
        """Initialize Attention.

    Args:
      size_per_head: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      dropout_rate: float, dropout rate inside attention for training.
    """

        super().__init__()
        self.size_per_head = size_per_head
        self.num_heads = num_heads
        self.output_size = output_size

    def build(self, input_shape):
        """Builds the layer."""
        # Layers for linearly projecting the queries, keys, and values.

        output_size = self.output_size if self.output_size is None else input_shape[-1]

        self.query_dense = DenseMultiHead(
            self.num_heads,
            self.size_per_head,
            kernel_initializer="glorot_uniform",
            use_bias=False,
            name="query",
        )
        self.key_dense = DenseMultiHead(
            self.num_heads,
            self.size_per_head,
            kernel_initializer="glorot_uniform",
            use_bias=False,
            name="key",
        )
        self.value_dense = DenseMultiHead(
            self.num_heads,
            self.size_per_head,
            kernel_initializer="glorot_uniform",
            use_bias=False,
            name="value",
        )
        self.projection_kernel = self.add_weight(
            "projection_kernel",
            shape=[self.num_heads, self.size_per_head, output_size],
            initializer="glorot_uniform",
            dtype=self.dtype,
            trainable=True,
        )

        super().build(input_shape)

    def get_config(self):
        return {
            "size_per_head": self.size_per_head,
            "num_heads": self.num_heads,
            "output_size": self.output_size,
        }

    def call(
        self, query, key, value=None, mask=None, training=None,
    ):
        """Apply attention mechanism to query_input and source_input.

    Args:
      query_input: A tensor with shape [batch_size, length_query, hidden_size].
      source_input: A tensor with shape [batch_size, length_source,
        hidden_size]

    Returns:
      Attention layer output with shape [batch_size, length_query, hidden_size]
    """
        # Linearly project the query, key and value using different learned
        # projections. Splitting heads is automatically done during the linear
        # projections --> [batch_size, length, num_heads, dim_per_head].

        if value is None:
            value = key

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        projection = self.projection_kernel

        attention_output = multi_head_attention(
            query, key, value, projection, mask=mask
        )

        return attention_output


################################################################
# functions
################################################################


def multi_head_attention(query, key, value, projection, mask=None):

    depth = tf.cast(tf.shape(query)[-1], tf.float32)

    query /= tf.sqrt(depth)

    # Calculate dot product attention
    logits = tf.einsum("BTNH,BFNH->BNFT", key, query)

    # apply mask
    if mask is not None:
        logits += mask

    # Note that softmax internally performs math operations using float32
    # for numeric stability. When training with float16, we keep the input
    # and output in float16 for better performance.
    attention = tf.nn.softmax(logits, name="attention_weights")

    concated_output = tf.einsum("BNFT,BTNH->BFNH", attention, value)

    # Run the outputs through another linear projection layer. Recombining heads
    # is automatically done --> [batch_size, length, hidden_size]
    attention_output = tf.einsum("BFNH,NHD->BFD", concated_output, projection)

    return attention_output
