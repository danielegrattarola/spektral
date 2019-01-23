from __future__ import absolute_import

import numpy as np
from keras import activations, initializers, regularizers, constraints
from keras import backend as K
from keras.constraints import Constraint
from keras.layers import Layer, Average, Concatenate


class Affinity(Layer):
    """
    Wrapper for affinity functions, used in graph matching.

    **Mode**: batch.

    **Input**

    - The input to this layer can be any combination of pairs of adjacency matrices,
    node attributes, and edge attributes as follows:
        - list of length 2, with source and target adjacency matrices.
        Shapes: `[(batch_size, num_nodes, num_nodes), (batch_size, num_nodes, num_nodes)]`;
        - list of length 4, with source and target adjacency matrices, and only
        one type of attributes (node or edge). Shapes: `[(batch_size, num_nodes, num_nodes),
        (batch_size, num_nodes, num_nodes), (batch_size, num_nodes, num_features),
        (batch_size, num_nodes, num_features)]`.
        In this case, specify which of the two types of attribute is being passed
        by setting the corresponding `*_features_dim` flag appropriately.
        - list of length 6, with source and target adjacency matrices and features.
        Shapes: `[(batch_size, num_nodes, num_nodes),
        (batch_size, num_nodes, num_nodes), (batch_size, num_nodes, num_node_features),
        (batch_size, num_nodes, num_node_features), (batch_size, num_nodes, num_edge_features),
        (batch_size, num_nodes, num_edge_features)]`.

    **Output**

    - rank 2 tensor of shape `(input_dim_1, input_dim_1)`

    :param affinity_function: a function computing affinity tensors between
    graphs. The function will be called as:
    ```
    affinity_function(adj_src, adj_target, nf_src, nf_target, ef_src, ef_target, N=num_nodes, F=node_features_dim, S=edge_features_dim)
    ```.
    :param num_nodes: number of nodes in the graphs. It will be passed as 
    keyword argument to the affinity function (with key `N`);
    :param node_features_dim: number of node attributes. It will be passed as
    keyword argument to the affinity function (with key `F`);
    :param edge_features_dim: number of edge attributes. It will be passed as
    keyword argument to the affinity function (with key `S`);
    :param kwargs: optional arguments for `Layer`.
    """
    def __init__(self,
                 affinity_function,
                 num_nodes=None,
                 node_features_dim=None,
                 edge_features_dim=None,
                 **kwargs):
        self.affinity_function = affinity_function
        self.num_nodes = num_nodes
        self.node_features_dim = node_features_dim
        self.edge_features_dim = edge_features_dim
        super(Affinity, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        adj_src, adj_target, nf_src, nf_target, ef_src, ef_target = (None,) * 6
        if len(inputs) == 2:
            adj_src, adj_target = inputs
        elif len(inputs) == 4:
            if self.node_features_dim is not None and self.edge_features_dim is None:
                adj_src, adj_target, nf_src, nf_target = inputs
            elif self.node_features_dim is None and self.edge_features_dim is not None:
                adj_src, adj_target, ef_src, ef_target = inputs
            else:
                raise ValueError('Only four input tensors were passed, but '
                                 'it was not possible to interpret their '
                                 'meaning. If you passed node features, set '
                                 'num_node_features accordingly and leave '
                                 'num_edge_features=None (and vice-versa for '
                                 'edge features).')
        elif len(inputs) == 6:
            adj_src, adj_target, nf_src, nf_target, ef_src, ef_target = inputs
        else:
            raise ValueError('An even number of input tensors between 2 and 6 '
                             'is expected, representing input and output '
                             'adjacency matrices (mandatory), node features, '
                             'and edge features.')
        return self.affinity_function(adj_src, adj_target,
                                      nf_src, nf_target,
                                      ef_src, ef_target,
                                      N=self.num_nodes,
                                      F=self.node_features_dim,
                                      S=self.edge_features_dim)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self, **kwargs):
        config = {}
        base_config = super(Affinity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InnerProduct(Layer):
    """
    Computes the inner product between elements of a given 2d tensor \(x\): 
    $$
        \\langle x, x \\rangle = xx^T.
    $$

    **Mode**: single.

    **Input**

    - rank 2 tensor of shape `(input_dim_1, input_dim_2)` (e.g. node features
    of shape `(num_nodes, num_features)`);

    **Output**

    - rank 2 tensor of shape `(input_dim_1, input_dim_1)`

    :param trainable_kernel: add a trainable square matrix between the inner
    product (i.e., `x.dot(w).dot(x.T)`);
    :param activation: activation function to use;
    :param kernel_initializer: initializer for the kernel matrix;
    :param kernel_regularizer: regularization applied to the kernel;
    :param activity_regularizer: regularization applied to the output;
    :param kernel_constraint: constraint applied to the kernel;
    """
    def __init__(self,
                 trainable_kernel=False,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(InnerProduct, self).__init__(**kwargs)
        self.trainable_kernel = trainable_kernel
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if self.trainable_kernel:
            features_dim = input_shape[-1]
            self.kernel = self.add_weight(shape=(features_dim, features_dim),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        self.built = True

    def call(self, inputs):
        if self.trainable_kernel:
            output = K.dot(K.dot(inputs, self.kernel), K.transpose(inputs))
        else:
            output = K.dot(inputs, K.transpose(inputs))
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            return (None, None)
        else:
            return input_shape[:-1] + (input_shape[-2], )

    def get_config(self, **kwargs):
        config = {}
        base_config = super(InnerProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MinkowskiProduct(Layer):
    """
    Computes the hyperbolic inner product between elements of a given 2d tensor
    \(x\): 
    $$
        \\langle x, x \\rangle = x \,
        \\begin{pmatrix}
            I_{d\\times d} & 0 \\\\ 0 & -1
        \\end{pmatrix} \\,x^T.
    $$

    **Mode**: single.

    **Input**

    - rank 2 tensor of shape `(input_dim_1, input_dim_2)` (e.g. node features
    of shape `(num_nodes, num_features)`);

    **Output**

    - rank 2 tensor of shape `(input_dim_1, input_dim_1)`

    :param input_dim_1: first dimension of the input tensor; set this if you
    encounter issues with shapes in your model, in order to provide an explicit
    output shape for your layer.
    :param activation: activation function to use;
    :param activity_regularizer: regularization applied to the output;
    """
    def __init__(self,
                 input_dim_1=None,
                 activation=None,
                 activity_regularizer=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MinkowskiProduct, self).__init__(**kwargs)
        self.input_dim_1 = input_dim_1
        self.activation = activations.get(activation)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.built = True

    def call(self, inputs):
        F = K.int_shape(inputs)[-1]
        minkowski_prod_mat = np.eye(F)
        minkowski_prod_mat[-1, -1] = -1.
        minkowski_prod_mat = K.constant(minkowski_prod_mat)
        output = K.dot(inputs, minkowski_prod_mat)
        output = K.dot(output, K.transpose(inputs))
        output = K.clip(output, -10e9, -1.)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            if self.input_dim_1 is None:
                return (None, None)
            else:
                return (self.input_dim_1, self.input_dim_1)
        else:
            return input_shape[:-1] + (input_shape[-2], )

    def get_config(self, **kwargs):
        config = {}
        base_config = super(MinkowskiProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CCMProjection(Layer):
    """
    Projects a tensor to a CCM depending on the value of `r`. Optionally, 
    `r` can be learned via backpropagation.

    **Input**

    - tensor of shape `(batch_size, input_dim)`.

    **Output**

    - tensor of shape `(batch_size, input_dim)`, where each sample along the
    0th axis is projected to the CCM.

    :param r: radius of the CCM. If r is a number, then use it as fixed
    radius. If `r='spherical'`, use a trainable weight as radius, with a
    positivity constraint. If `r='hyperbolic'`, use a trainable weight
    as radius, with a negativity constraint. If `r=None`, use a trainable
    weight as radius, with no constraints (points will be projected to the
    correct manifold based on the sign of the weight).
    :param kernel_initializer: initializer for the kernel matrix;
    :param kernel_regularizer: regularization applied to the kernel matrix;
    :param kernel_constraint: constraint applied to the kernel matrix.
    """
    def __init__(self,
                 r=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None, **kwargs):
        super(CCMProjection, self).__init__(**kwargs)
        self.radius = r
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        if self.radius == 'spherical':
            self.kernel_constraint = self.Pos()
        elif self.radius == 'hyperbolic':
            self.kernel_constraint = self.Neg()
        else:
            self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if self.radius in {'spherical', 'hyperbolic'} or self.radius is None:
            self.radius = self.add_weight(shape=(),
                                          initializer=self.kernel_initializer,
                                          name='radius',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        else:
            self.radius = K.constant(self.radius)
        self.built = True

    def call(self, inputs):
        zero = K.constant(0.)

        # Spherical clip
        spherical_clip = self.radius * K.l2_normalize(inputs, -1)
        # Hyperbolic clip
        free_components = inputs[..., :-1]
        bound_component = K.sqrt(K.sum(free_components ** 2, -1)[..., None] + (self.radius ** 2))
        hyperbolic_clip = K.concatenate((free_components, bound_component), -1)

        lt_cond = K.less(self.radius, zero)
        lt_check = K.switch(lt_cond, hyperbolic_clip, inputs)

        gt_cond = K.greater(self.radius, zero)
        output = K.switch(gt_cond, spherical_clip, lt_check)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self, **kwargs):
        config = {}
        base_config = super(CCMProjection, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    class Pos(Constraint):
        """Constrains a scalar weight to be positive.
        """

        def __call__(self, w):
            return K.maximum(w, K.epsilon())

    class Neg(Constraint):
        """Constrains a scalar weight to be negative.
        """

        def __call__(self, w):
            return K.minimum(w, -K.epsilon())


class CCMMembership(Layer):
    """
    Computes the membership of the given points to a constant-curvature
    manifold of radius `r`, as: 
    $$
        \\mu(x) = \\mathrm{exp}\\left(\\cfrac{-\\big( \\langle \\vec x, \\vec x \\rangle - r^2 \\big)^2}{2\\sigma^2}\\right).
    $$

    If `r=0`, then \(\\mu(x) = 1\).
    If more than one radius is given, inputs are evenly split across the 
    last dimension and membership is computed for each radius-slice pair.
    The output membership is returned according to the `mode` option.

    **Input**

    - tensor of shape `(batch_size, input_dim)`;

    **Output**

    - tensor of shape `(batch_size, output_size)`, where `output_size` is
    computed according to the `mode` option;.

    :param r: int ot list, radia of the CCMs.
    :param mode: 'average' to return the average membership across CCMs, or
    'concat' to return the membership for each CCM concatenated;
    :param sigma: spread of the membership curve;
    """
    def __init__(self, r=1., mode='average', sigma=1., **kwargs):
        super(CCMMembership, self).__init__(**kwargs)
        if isinstance(r, int) or isinstance(r, float):
            self.r = [r]
        elif isinstance(r, list) or isinstance(r, tuple):
            self.r = r
        else:
            raise TypeError('r must be either a single value, or a list/tuple '
                            'of values.')
        possible_modes = {'average', 'concat'}
        if mode not in possible_modes:
            raise ValueError('Possible modes: {}'.format(possible_modes))
        self.mode = mode
        self.sigma = sigma

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        output_part = []
        manifold_size = K.int_shape(inputs)[-1] // len(self.r)

        for idx, r_ in enumerate(self.r):
            start = idx * manifold_size
            stop = start + manifold_size
            part = inputs[..., start:stop]
            sign = np.sign(r_)
            if sign == 0.:
                # This is weird but necessary to make the layer differentiable
                output_pre = K.sum(inputs, -1, keepdims=True) * 0. + 1.
            else:
                free_components = part[..., :-1] ** 2
                bound_component = sign * part[..., -1:] ** 2
                all_components = K.concatenate((free_components, bound_component), -1)
                ext_product = K.sum(all_components, -1, keepdims=True)
                output_pre = K.exp(-(ext_product - sign * r_ ** 2) ** 2 / (2 * self.sigma ** 2))

            output_part.append(output_pre)

        if len(output_part) >= 2:
            if self.mode == 'average':
                output = Average()(output_part)
            elif self.mode == 'concat':
                output = Concatenate()(output_part)
            else:
                raise ValueError()  # Never gets here
        else:
            output = output_part[0]

        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:-1] + (1, )
        return output_shape

    def get_config(self, **kwargs):
        config = {}
        base_config = super(CCMMembership, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GaussianSampling(Layer):
    """
    A layer for sampling from a Gaussian distribution using the re-parametrization
    trick of Kingma et al., 2014.

    **Input**

    - tensor of shape `(batch_size, input_dim)` representing the means;
    - tensor of shape `(batch_size, input_dim)` representing the log-variances;

    **Output**

    - tensor of shape `(batch_size, input_dim)`, obtained by sampling from a
    Gaussian distribution with the given means and log-variances, using the
    re-parametrization trick;
        
    :param mean: mean of the Gaussian noise;
    :param std: standard deviation of the Gaussian noise.
    """
    def __init__(self, mean=0., std=1.0, **kwargs):
        super(GaussianSampling, self).__init__(**kwargs)
        self.mean = mean
        self.std = std

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.built = True

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = K.random_normal(shape=K.shape(z_log_var),
                                  mean=self.mean,
                                  stddev=self.std)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self, **kwargs):
        config = {}
        base_config = super(GaussianSampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Bias(Layer):
    """
    A layer for adding a trainable bias vector (wrapper for `K.bias_add`).

    **Input**

    - tensor of shape `(batch_size, input_dim_1, ..., input_dim_n)`;

    **Output**

    - tensor of shape `(batch_size, input_dim_1, ..., input_dim_n)`;

    :param bias_initializer: initializer for the bias;
    :param bias_regularizer: regularizer for the bias;
    :param bias_constraint: constraint for the bias;

    """
    def __init__(self,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        super(Bias, self).__init__(**kwargs)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.bias = self.add_weight(shape=input_shape[1:],
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
        self.built = True

    def call(self, inputs):
        return K.bias_add(inputs, self.bias)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self, **kwargs):
        config = {}
        base_config = super(Bias, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

