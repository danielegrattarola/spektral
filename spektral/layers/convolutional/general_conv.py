from spektral.layers import MessagePassing


class GeneralConv(MessagePassing):
    def __init__(self,
                 channels,
                 batch_norm=True,
                 dropout=0.0,
                 aggregate='sum',
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
        super().__init__(aggregate=aggregate,
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
        self.channels = self.output_dim = channels

    def build(self, input_shape):
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
        self.built = True

    def call(self, inputs, **kwargs):
        x, a, _ = self.get_inputs(inputs)



    def message(self, x, **kwargs):
        pass


    def get_config(self):
        config = {
            'channels': self.channels,
        }
        base_config = super().get_config()
        base_config.pop('aggregate')  # Remove it because it's defined by constructor
        return {**base_config, **config}