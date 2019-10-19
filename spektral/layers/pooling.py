import tensorflow as tf
from keras import backend as K, activations
from keras import regularizers, constraints, initializers
from keras.engine import Layer

from spektral.layers import ops, filter_dot


################################################################################
# Pooling layers
################################################################################
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
    - If `return_mask=True`, the binary mask used for pooling, with shape
    `(n_graphs * k, )`.

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
        super().__init__(**kwargs)
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
        super().build(input_shape)

    def call(self, inputs):
        if len(inputs) == 3:
            X, A, I = inputs
            self.data_mode = 'graph'
        else:
            X, A = inputs
            I = tf.zeros(tf.shape(X)[:1], dtype=tf.int32)
            self.data_mode = 'single'
        if K.ndim(I) == 2:
            I = I[:, 0]

        A_is_sparse = K.is_sparse(A)

        # Get mask
        y = K.dot(X, K.l2_normalize(self.kernel))
        N = K.shape(X)[-2]
        indices = ops.segment_top_k(y[:, 0], I, self.ratio, self.top_k_var)
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
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SAGPool(Layer):
    """
    A self-attention graph pooling layer as presented by
    [Lee et al. (2019)](https://arxiv.org/abs/1904.08082) and
    [Knyazev et al. (2019](https://arxiv.org/abs/1905.02850).

    This layer computes the following operations:

    $$
    y = GNN(X, A); \\;\\;\\;\\;
    \\textrm{idx} = \\textrm{rank}(y, k); \\;\\;\\;\\;
    \\bar X = (X \\odot \\textrm{tanh}(y))_{\\textrm{idx}}; \\;\\;\\;\\;
    \\bar A = A^2_{\\textrm{idx}, \\textrm{idx}}
    $$

    where \( \\textrm{rank}(y, k) \) returns the indices of the top k values of
    \( y \), and \( p \) is a learnable parameter vector of size \(F\).
    The gating operation \( \\textrm{tanh}(y) \) can be replaced with a sigmoid.

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
    - If `return_mask=True`, the binary mask used for pooling, with shape
    `(n_graphs * k, )`.

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
        super().__init__(**kwargs)
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
        super().build(input_shape)

    def call(self, inputs):
        if len(inputs) == 3:
            X, A, I = inputs
            self.data_mode = 'graph'
        else:
            X, A = inputs
            I = tf.zeros(tf.shape(X)[:1], dtype=tf.int32)
            self.data_mode = 'single'
        if K.ndim(I) == 2:
            I = I[:, 0]

        A_is_sparse = K.is_sparse(A)

        # Get mask
        y = K.dot(X, self.kernel)
        y = filter_dot(A, y)
        N = K.shape(X)[-2]
        indices = ops.segment_top_k(y[:, 0], I, self.ratio, self.top_k_var)
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
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MinCutPool(Layer):
    """
    A minCUT pooling layer as presented by [Bianchi et al.](https://arxiv.org/abs/1907.00481).

    **Mode**: single, batch.

    This layer computes a soft clustering \(S\) of the input graphs using a MLP,
    and reduces graphs as follows:

    $$
        A^{pool} = S^T A S; X^{pool} = S^T X;
    $$

    Besides training the MLP, two additional unsupervised loss terms are
    minimized to ensure that the cluster assignment solves the minCUT
    optimization problem.
    The layer can be used without a supervised loss, to compute node clustering
    simply by minimizing the unsupervised loss.

    **Input**

    - node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - adjacency matrix of shape `(n_nodes, n_nodes)` (with optional `batch`
    dimension);

    **Output**

    - reduced node features of shape `(k, n_features)`;
    - reduced adjacency matrix of shape `(k, k)`;
    - reduced graph IDs with shape `(k, )` (graph batch mode);
    - If `return_mask=True`, the soft assignment matrix used for pooling, with
    shape `(n_nodes, k)`.

    **Arguments**

    - `k`: number of nodes to keep;
    - `h`: number of units in the hidden layer;
    - `return_mask`: boolean, whether to return the cluster assignment matrix,
    - `kernel_initializer`: initializer for the kernel matrix;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.
    """
    def __init__(self,
                 k,
                 h=None,
                 return_mask=True,
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
        self.k = k
        self.h = h
        self.return_mask = return_mask
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        F = input_shape[0][-1]

        # Optional hidden layer
        if self.h is None:
            H_ = F
        else:
            H_ = self.h
            self.kernel_in = self.add_weight(shape=(F, H_),
                                             name='kernel_in',
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             constraint=self.kernel_constraint)

            if self.use_bias:
                self.bias_in = self.add_weight(shape=(H_,),
                                               name='bias_in',
                                               initializer=self.bias_initializer,
                                               regularizer=self.bias_regularizer,
                                               constraint=self.bias_constraint)

        # Output layer
        self.kernel_out = self.add_weight(shape=(H_, self.k),
                                          name='kernel_out',
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias_out = self.add_weight(shape=(self.k,),
                                            name='bias_out',
                                            initializer=self.bias_initializer,
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)

        super().build(input_shape)

    def call(self, inputs):
        # Note that I is useless, because thee layer cannot be used in graph
        # batch mode.
        if len(inputs) == 3:
            X, A, I = inputs
        else:
            X, A = inputs
            I = None

        # Check if the layer is operating in batch mode (X and A have rank 3)
        batch_mode = K.ndim(A) == 3

        # Optionally compute hidden layer
        if self.h is None:
            Hid = X
        else:
            Hid = K.dot(X, self.kernel_in)
            if self.use_bias:
                Hid = K.bias_add(Hid, self.bias_in)
            if self.activation is not None:
                Hid = self.activation(Hid)

        # Compute cluster assignment matrix
        S = K.dot(Hid, self.kernel_out)
        if self.use_bias:
            S = K.bias_add(S, self.bias_out)
        S = activations.softmax(S, axis=-1)  # Apply softmax to get cluster assignments

        # MinCut regularization
        A_pooled = ops.matmul_AT_B_A(S, A)
        num = tf.trace(A_pooled)

        D = ops.degree_matrix(A)
        den = tf.trace(ops.matmul_AT_B_A(S, D))
        cut_loss = -(num / den)
        if batch_mode:
            cut_loss = K.mean(cut_loss)
        self.add_loss(cut_loss)

        # Orthogonality regularization
        SS = ops.matmul_AT_B(S, S)
        I_S = tf.eye(self.k)
        ortho_loss = tf.norm(
            SS / tf.norm(SS, axis=(-1, -2)) - I_S / tf.norm(I_S), axis=(-1, -2)
        )
        if batch_mode:
            ortho_loss = K.mean(cut_loss)
        self.add_loss(ortho_loss)

        # Pooling
        X_pooled = ops.matmul_AT_B(S, X)
        A_pooled = tf.linalg.set_diag(A_pooled, tf.zeros(K.shape(A_pooled)[:-1]))  # Remove diagonal
        A_pooled = ops.normalize_A(A_pooled)

        output = [X_pooled, A_pooled]

        if I is not None:
            I_mean = tf.segment_mean(I, I)
            I_pooled = ops.repeat(I_mean, tf.ones_like(I_mean) * self.k)
            output.append(I_pooled)

        if self.return_mask:
            output.append(S)

        return output

    def compute_output_shape(self, input_shape):
        X_shape = input_shape[0]
        A_shape = input_shape[1]
        X_shape_out = X_shape[:-2] + (self.k,) + X_shape[-1:]
        A_shape_out = A_shape[:-2] + (self.k, self.k)

        output_shape = [X_shape_out, A_shape_out]

        if len(input_shape) == 3:
            I_shape_out = A_shape[:-2] + (self.k, )
            output_shape.append(I_shape_out)

        if self.return_mask:
            S_shape_out = A_shape[:-1] + (self.k, )
            output_shape.append(S_shape_out)

        return output_shape

    def get_config(self):
        config = {
            'k': self.k,
            'h': self.h,
            'return_mask': self.return_mask,
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


class DiffPool(Layer):
    """
    A DiffPool layer as presented by [Ying et al.](https://arxiv.org/abs/1806.08804).

    **Mode**: single, batch.

    This layer computes a soft clustering \(S\) of the input graphs using a GNN,
    and reduces graphs as follows:

    $$
        A^{pool} = S^T A S; X^{pool} = S^T X;
    $$

    Besides training the GNN, two additional unsupervised loss terms are
    minimized. The layer also applies a GNN to the input features, and returns
    the updated graph signal (the number of output channels is controlled by
    the `channels` parameter).
    The layer can be used without a supervised loss, to compute node clustering
    simply by minimizing the unsupervised loss.

    **Input**

    - node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - adjacency matrix of shape `(n_nodes, n_nodes)` (with optional `batch`
    dimension);

    **Output**

    - reduced node features of shape `(k, channels)`;
    - reduced adjacency matrix of shape `(k, k)`;
    - reduced graph IDs with shape `(k, )` (graph batch mode);
    - If `return_mask=True`, the soft assignment matrix used for pooling, with
    shape `(n_nodes, k)`.

    **Arguments**

    - `k`: number of nodes to keep;
    - `channels`: number of output channels (if None, the number of output
    channels is assumed to be the same as the input);
    - `return_mask`: boolean, whether to return the cluster assignment matrix,
    - `kernel_initializer`: initializer for the kernel matrix;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    """

    def __init__(self,
                 k,
                 channels=None,
                 return_mask=True,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.k = k
        self.channels = channels
        self.return_mask = return_mask
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.mixed_mode = False

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        F = input_shape[0][-1]

        if self.channels is None:
            self.channels = F

        self.kernel_emb = self.add_weight(shape=(F, self.channels),
                                          name='kernel_emb',
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)

        self.kernel_pool = self.add_weight(shape=(F, self.k),
                                           name='kernel_pool',
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        super().build(input_shape)

    def call(self, inputs):
        # Note that I is useless, because thee layer cannot be used in graph
        # batch mode.
        if len(inputs) == 3:
            X, A, I = inputs
        else:
            X, A = inputs
            I = None

        N = K.shape(A)[-1]
        # Check if the layer is operating in batch mode (X and A have rank 3)
        mode = ops.autodetect_mode(A, X)
        self.reduce_loss = mode in (ops._modes['M'], ops._modes['B'])

        # Get normalized adjacency
        if K.is_sparse(A):
            I_ = tf.sparse.eye(N, dtype=A.dtype)
            A_ = tf.sparse.add(A, I_)
        else:
            I_ = tf.eye(N, dtype=A.dtype)
            A_ = A + I_
        fltr = ops.normalize_A(A_)

        # Node embeddings
        Z = K.dot(X, self.kernel_emb)
        Z = ops.filter_dot(fltr, Z)
        if self.activation is not None:
            Z = self.activation(Z)

        # Compute cluster assignment matrix
        S = K.dot(X, self.kernel_pool)
        S = ops.filter_dot(fltr, S)
        S = activations.softmax(S, axis=-1)  # softmax applied row-wise

        # Link prediction loss
        S_gram = ops.matmul_A_BT(S, S)
        if K.is_sparse(A):
            LP_loss = tf.sparse.add(A, -S_gram)  # A/tf.norm(A) - S_gram/tf.norm(S_gram)
        else:
            LP_loss = A - S_gram
        LP_loss = tf.norm(LP_loss, axis=(-1, -2))
        if self.reduce_loss:
            LP_loss = K.mean(LP_loss)
        self.add_loss(LP_loss)

        # Entropy loss
        entr = tf.negative(tf.reduce_sum(tf.multiply(S, K.log(S + K.epsilon())), axis=-1))
        entr_loss = K.mean(entr, axis=-1)
        if self.reduce_loss:
            entr_loss = K.mean(entr_loss)
        self.add_loss(entr_loss)

        # Pooling
        X_pooled = ops.matmul_AT_B(S, Z)
        A_pooled = ops.matmul_AT_B_A(S, A)

        if K.ndim(A_pooled) == 3:
            self.mixed_mode = True

        output = [X_pooled, A_pooled]

        if I is not None:
            I_mean = tf.segment_mean(I, I)
            I_pooled = ops.repeat(I_mean, tf.ones_like(I_mean) * self.k)
            output.append(I_pooled)

        if self.return_mask:
            output.append(S)

        return output

    def compute_output_shape(self, input_shape):
        X_shape = input_shape[0]
        A_shape = input_shape[1]
        X_shape_out = X_shape[:-2] + (self.k, self.channels)
        if self.reduce_loss:
            A_shape_out = X_shape[:-2] + (self.k, self.k)
        else:
            A_shape_out = A_shape[:-2] + (self.k, self.k)

        output_shape = [X_shape_out, A_shape_out]

        if len(input_shape) == 3:
            I_shape_out = A_shape[:-2] + (self.k,)
            output_shape.append(I_shape_out)

        if self.return_mask:
            S_shape_out = A_shape[:-1] + (self.k,)
            output_shape.append(S_shape_out)

        return output_shape

    def get_config(self):
        config = {
            'k': self.k,
            'channels': self.channels,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


################################################################################
# Global pooling layers
################################################################################
class GlobalPooling(Layer):
    def __init__(self, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.supports_masking = True
        self.pooling_op = None

    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = 'graph'
        else:
            if len(input_shape) == 2:
                self.data_mode = 'single'
            else:
                self.data_mode = 'batch'
        super().build(input_shape)

    def call(self, inputs):
        if self.data_mode == 'graph':
            X = inputs[0]
            I = inputs[1]
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X = inputs

        if self.data_mode == 'graph':
            return self.pooling_op(X, I)
        else:
            return K.sum(X, axis=-2, keepdims=(self.data_mode == 'single'))

    def compute_output_shape(self, input_shape):
        if self.data_mode == 'single':
            return (1,) + input_shape[-1:]
        elif self.data_mode == 'batch':
            return input_shape[:-2] + input_shape[-1:]
        else:
            return input_shape[0]  # Input shape is a list of shapes for X and I

    def get_config(self):
        return super().get_config()


class GlobalSumPool(GlobalPooling):
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
        super().__init__(**kwargs)
        self.pooling_op = tf.segment_sum


class GlobalAvgPool(GlobalPooling):
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
        super().__init__(**kwargs)
        self.pooling_op = tf.segment_mean


class GlobalMaxPool(GlobalPooling):
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
        super().__init__(**kwargs)
        self.pooling_op = tf.segment_max


class GlobalAttentionPool(GlobalPooling):
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
        super().__init__(**kwargs)
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
            if K.ndim(I) == 2:
                I = I[:, 0]
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
        config = {
            'channels': self.channels,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalAttnSumPool(GlobalPooling):
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
        super().__init__(**kwargs)
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
            output = tf.segment_sum(output, I)

        return output

    def get_config(self):
        config = {
            'attn_kernel_initializer': self.attn_kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'attn_kernel_regularizer': self.attn_kernel_regularizer,
            'attn_kernel_constraint': self.attn_kernel_constraint,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

