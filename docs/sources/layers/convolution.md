<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional.py#L11)</span>
### GraphConv

```python
spektral.layers.GraphConv(channels, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```


A graph convolutional layer as presented by
[Kipf & Welling (2016)](https://arxiv.org/abs/1609.02907).

**Mode**: single, mixed, batch.

This layer computes:
$$  
Z = \sigma( \tilde{A} XW + b)
$$
where \(X\) is the node features matrix, \(\tilde{A}\) is the normalized
Laplacian, \(W\) is the convolution kernel, \(b\) is a bias vector, and
\(\sigma\) is the activation function.

**Input**

- Node features of shape `(n_nodes, n_features)` (with optional `batch`
dimension);
- Normalized Laplacian of shape `(n_nodes, n_nodes)` (with optional `batch`
dimension); see `spektral.utils.convolution.localpooling_filter`.

**Output**

- Node features with the same shape of the input, but the last dimension
changed to `channels`.

**Arguments**

- `channels`: integer, number of output channels;
- `activation`: activation function to use;
- `use_bias`: whether to add a bias to the linear transformation;
- `kernel_initializer`: initializer for the kernel matrix;
- `bias_initializer`: initializer for the bias vector;
- `kernel_regularizer`: regularization applied to the kernel matrix;  
- `bias_regularizer`: regularization applied to the bias vector;
- `activity_regularizer`: regularization applied to the output;
- `kernel_constraint`: constraint applied to the kernel matrix;
- `bias_constraint`: constraint applied to the bias vector.

**Usage**  

```py
# Load data
A, X, _, _, _, _, _, _ = citation.load_data('cora')

# Preprocessing operations
fltr = utils.localpooling_filter(A)

# Model definition
X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)
output = GraphConv(channels)([X_in, fltr_in])
```

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional.py#L147)</span>
### ChebConv

```python
spektral.layers.ChebConv(channels, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```


A Chebyshev convolutional layer as presented by
[Defferrard et al. (2016)](https://arxiv.org/abs/1606.09375).

**Mode**: single, mixed, batch.

Given a list of Chebyshev polynomials \(T = [T_{1}, ..., T_{K}]\), 
this layer computes:
$$
Z = \sigma( \sum \limits_{k=1}^{K} T_{k} X W  + b)
$$
where \(X\) is the node features matrix, \(W\) is the convolution kernel, 
\(b\) is the bias vector, and \(\sigma\) is the activation function.

**Input**

- Node features of shape `(n_nodes, n_features)` (with optional `batch`
dimension);
- A list of Chebyshev polynomials of shape `(num_nodes, num_nodes)` (with
optional `batch` dimension); see `spektral.utils.convolution.chebyshev_filter`.

**Output**

- Node features with the same shape of the input, but the last dimension
changed to `channels`.

**Arguments**

- `channels`: integer, number of output channels;
- `activation`: activation function to use;
- `use_bias`: boolean, whether to add a bias to the linear transformation;
- `kernel_initializer`: initializer for the kernel matrix;
- `bias_initializer`: initializer for the bias vector;
- `kernel_regularizer`: regularization applied to the kernel matrix;  
- `bias_regularizer`: regularization applied to the bias vector;
- `activity_regularizer`: regularization applied to the output;
- `kernel_constraint`: constraint applied to the kernel matrix;
- `bias_constraint`: constraint applied to the bias vector.

**Usage**
```py
# Load data
A, X, _, _, _, _, _, _ = citation.load_data('cora')

# Preprocessing operations
fltr = utils.chebyshev_filter(A, K)

# Model definition
X_in = Input(shape=(F, ))
fltr_in = [Input((N, ), sparse=True) for _ in range(K + 1)]
output = ChebConv(channels)([X_in] + fltr_in)
```

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional.py#L287)</span>
### GraphSageConv

```python
spektral.layers.GraphSageConv(channels, aggregate_method='mean', activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```


A GraphSage layer as presented by [Hamilton et al. (2017)](https://arxiv.org/abs/1706.02216).

**Mode**: single.

This layer computes:
$$
Z = \sigma \big( \big[ \textrm{AGGREGATE}(X) \| X \big] W + b \big)
$$
where \(X\) is the node features matrix, \(W\) is a trainable kernel,
\(b\) is a bias vector, and \(\sigma\) is the activation function.
\(\textrm{AGGREGATE}\) is an aggregation function as described in the
original paper, that works by aggregating each node's neighbourhood
according to some rule. The supported aggregation methods are: sum, mean,
max, min, and product.

**Input**

- Node features of shape `(n_nodes, n_features)`;
- Binary adjacency matrix of shape `(n_nodes, n_nodes)`.

**Output**

- Node features with the same shape of the input, but the last dimension
changed to `channels`.

**Arguments**

- `channels`: integer, number of output channels;
- `aggregate_method`: str, aggregation method to use (`'sum'`, `'mean'`,
`'max'`, `'min'`, `'prod'`);
- `activation`: activation function to use;
- `use_bias`: whether to add a bias to the linear transformation;
- `kernel_initializer`: initializer for the kernel matrix;
- `bias_initializer`: initializer for the bias vector;
- `kernel_regularizer`: regularization applied to the kernel matrix;
- `bias_regularizer`: regularization applied to the bias vector;
- `activity_regularizer`: regularization applied to the output;
- `kernel_constraint`: constraint applied to the kernel matrix;
- `bias_constraint`: constraint applied to the bias vector.

**Usage**

```py
X_in = Input(shape=(F, ))
A_in = Input((N, ), sparse=True)
output = GraphSageConv(channels)([X_in, A_in])
```

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional.py#L985)</span>
### ARMAConv

```python
spektral.layers.ARMAConv(channels, T=1, K=1, recurrent=False, gcn_activation='relu', dropout_rate=0.0, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```


A graph convolutional layer with ARMA(K, K-1) filters, as presented by
[Bianchi et al. (2019)](https://arxiv.org/abs/1901.01343).

**Mode**: single, mixed, batch.

This layer computes:
$$
Z = \frac{1}{K}\sum \limits_{k=1}^K \bar{X}_k^{(T)},
$$
where \(K\) is the order of the ARMA(K, K-1) filter, and where:
$$
\bar{X}_k^{(t + 1)} =  \sigma\left(\tilde{L}\bar{X}^{(t)}W^{(t)} + XV^{(t)}\right)
$$
is a graph convolutional skip layer implementing a recursive approximation
of an ARMA(1, 0) filter, \(\tilde{L}\) is  normalized graph Laplacian with
a rescaled spectrum, \(\bar{X}^{(0)} = X\), and \(W, V\) are trainable
kernels.

**Input**

- node features of shape `(n_nodes, n_features)` (with optional `batch`
dimension);
- Normalized Laplacian of shape `(n_nodes, n_nodes)` (with optional `batch`
dimension); see the [ARMA node classification example](https://github.com/danielegrattarola/spektral/blob/master/examples/node_classification_arma.py)

**Output**

- node features with the same shape of the input, but the last dimension
changed to `channels`.

**Arguments**

- `channels`: integer, number of output channels;
- `T`: depth of each ARMA(1, 0) approximation (number of recursive updates);
- `K`: order of the full ARMA(K, K-1) filter (combination of K ARMA(1, 0)
filters);
- `recurrent`: whether to share each head's weights like a recurrent net;
- `gcn_activation`: activation function to use to compute the ARMA filter;
- `dropout_rate`: dropout rate for Laplacian and output layer;
- `activation`: activation function to use;
- `use_bias`: whether to add a bias to the linear transformation;
- `kernel_initializer`: initializer for the kernel matrix;
- `bias_initializer`: initializer for the bias vector;
- `kernel_regularizer`: regularization applied to the kernel matrix;
- `bias_regularizer`: regularization applied to the bias vector;
- `activity_regularizer`: regularization applied to the output;
- `kernel_constraint`: constraint applied to the kernel matrix;
- `bias_constraint`: constraint applied to the bias vector.

**Usage**
```py
# Load data
A, X, _, _, _, _, _, _ = citation.load_data('cora')

# Preprocessing operations
fltr = utils.normalized_adjacency(A)

# Model definition
X_in= Input(shape=(F, ), sparse=True)
fltr_in = Input((N, ))
output = ARMAConv(channels)([X_in, fltr_in])
```

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional.py#L418)</span>
### EdgeConditionedConv

```python
spektral.layers.EdgeConditionedConv(channels, kernel_network=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```


An edge-conditioned convolutional layer as presented by [Simonovsky and
Komodakis (2017)](https://arxiv.org/abs/1704.02901).

**Mode**: batch.

For each node \(i\), this layer computes:
$$
Z_i =  \frac{1}{\mathcal{N}(i)} \sum\limits_{j \in \mathcal{N}(i)} F(E_{ji}) X_{j} + b
$$
where \(\mathcal{N}(i)\) represents the one-step neighbourhood of node \(i\),
\(F\) is a neural network that outputs the convolution kernel as a
function of edge attributes, \(E\) is the edge attributes matrix, and \(b\)
is a bias vector.

**Input**

- node features of shape `(batch, n_nodes, n_node_features)`;
- adjacency matrices of shape `(batch, n_nodes, num_nodes)`;
- edge features of shape `(batch, n_nodes, n_nodes, n_edge_features)`;

**Output**

- node features with the same shape of the input, but the last dimension
changed to `channels`.

**Arguments**

- `channels`: integer, number of output channels;
- `kernel_network`: a list of integers describing the hidden structure of
the kernel-generating network (i.e., the ReLU layers before the linear
output);
- `activation`: activation function to use;
- `use_bias`: boolean, whether to add a bias to the linear transformation;
- `kernel_initializer`: initializer for the kernel matrix;
- `bias_initializer`: initializer for the bias vector;
- `kernel_regularizer`: regularization applied to the kernel matrix;  
- `bias_regularizer`: regularization applied to the bias vector;
- `activity_regularizer`: regularization applied to the output;
- `kernel_constraint`: constraint applied to the kernel matrix;
- `bias_constraint`: constraint applied to the bias vector.

**Usage**
```py
X_in = Input(shape=(N, F))
A_in = Input(shape=(N, N))
E_in = Input(shape=(N, N, S))
output = EdgeConditionedConv(channels)([X_in, A_in, E_in])
```

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional.py#L605)</span>
### GraphAttention

```python
spektral.layers.GraphAttention(channels, attn_heads=1, attn_heads_reduction='concat', dropout_rate=0.5, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', attn_kernel_initializer='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, attn_kernel_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, attn_kernel_constraint=None)
```


A graph attention layer as presented by
[Velickovic et al. (2017)](https://arxiv.org/abs/1710.10903).

**Mode**: single, mixed, batch.

This layer computes a convolution similar to `layers.GraphConv`, but
uses the attention mechanism to weight the adjacency matrix instead of
using the normalized Laplacian.

**Input**

- node features of shape `(n_nodes, n_features)` (with optional `batch`
dimension);
- adjacency matrices of shape `(n_nodes, n_nodes)` (with optional `batch`
dimension);

**Output**

- node features with the same shape of the input, but the last dimension
changed to `channels`.

**Arguments**

- `channels`: integer, number of output channels;
- `attn_heads`: number of attention heads to use;
- `attn_heads_reduction`: how to reduce the outputs of the attention heads 
(can be either 'concat' or 'average');
- `dropout_rate`: internal dropout rate;
- `activation`: activation function to use;
- `use_bias`: boolean, whether to add a bias to the linear transformation;
- `kernel_initializer`: initializer for the kernel matrix;
- `attn_kernel_initializer`: initializer for the attention kernel matrices;
- `bias_initializer`: initializer for the bias vector;
- `kernel_regularizer`: regularization applied to the kernel matrix;  
- `attn_kernel_regularizer`: regularization applied to the attention kernel 
matrices;
- `bias_regularizer`: regularization applied to the bias vector;
- `activity_regularizer`: regularization applied to the output;
- `kernel_constraint`: constraint applied to the kernel matrix;
- `attn_kernel_constraint`: constraint applied to the attention kernel
matrices;
- `bias_constraint`: constraint applied to the bias vector.

**Usage**

```py
# Load data
A, X, _, _, _, _, _, _ = citation.load_data('cora')

# Preprocessing operations
A = utils.add_eye(A).toarray()  # Add self-loops

# Model definition
X_in = Input(shape=(F, ))
A_in = Input((N, ))
output = GraphAttention(channels)([X_in, A_in])
```

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional.py#L838)</span>
### GraphConvSkip

```python
spektral.layers.GraphConvSkip(channels, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```


A graph convolutional layer as presented by
[Kipf & Welling (2016)](https://arxiv.org/abs/1609.02907), with the addition
of a skip connection.

**Mode**: single, mixed, batch.

This layer computes:
$$
Z = \sigma(A X W_1 + X W_2 + b)
$$
where \(X\) is the node features matrix, \(A\) is the normalized laplacian,
\(W_1\) and \(W_2\) are the convolution kernels, \(b\) is a bias vector,
and \(\sigma\) is the activation function.

**Input**

- node features of shape `(n_nodes, n_features)` (with optional `batch`
dimension);
- Normalized adjacency matrix of shape `(n_nodes, n_nodes)` (with optional
`batch` dimension); see `spektral.utils.convolution.normalized_adjacency`.

**Output**

- node features with the same shape of the input, but the last dimension
changed to `channels`.

**Arguments**

- `channels`: integer, number of output channels;
- `activation`: activation function to use;
- `use_bias`: whether to add a bias to the linear transformation;
- `kernel_initializer`: initializer for the kernel matrix;
- `bias_initializer`: initializer for the bias vector;
- `kernel_regularizer`: regularization applied to the kernel matrix;
- `bias_regularizer`: regularization applied to the bias vector;
- `activity_regularizer`: regularization applied to the output;
- `kernel_constraint`: constraint applied to the kernel matrix;
- `bias_constraint`: constraint applied to the bias vector.

**Usage**
```py
# Load data
A, X, _, _, _, _, _, _ = citation.load_data('cora')

# Preprocessing operations
fltr = utils.normalized_adjacency(A)

# Model definition
X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)
output = GraphConvSkip(channels)([X_in, fltr_in])
```

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional.py#L1316)</span>
### APPNP

```python
spektral.layers.APPNP(channels, mlp_channels, alpha=0.2, H=1, K=1, mlp_activation='relu', dropout_rate=0.0, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```


A graph convolutional layer implementing the APPNP operator, as presented by
[Klicpera et al. (2019)](https://arxiv.org/abs/1810.05997).

**Mode**: single, mixed, batch.

**Input**

- node features of shape `(n_nodes, n_features)` (with optional `batch`
dimension);
- Normalized adjacency matrix of shape `(n_nodes, n_nodes)` (with optional
`batch` dimension); see `spektral.utils.convolution.normalized_adjacency`.

**Output**

- node features with the same shape of the input, but the last dimension
changed to `channels`.

**Arguments**

- `channels`: integer, number of output channels;
- `mlp_channels`: integer, number of hidden units for the MLP layers;
- `alpha`: teleport probability;
- `H`: number of MLP layers;
- `K`: number of power iterations;
- `mlp_activation`: activation for the MLP layers;
- `dropout_rate`: dropout rate for Laplacian and MLP layers;
- `activation`: activation function to use;
- `use_bias`: whether to add a bias to the linear transformation;
- `kernel_initializer`: initializer for the kernel matrix;
- `bias_initializer`: initializer for the bias vector;
- `kernel_regularizer`: regularization applied to the kernel matrix;
- `bias_regularizer`: regularization applied to the bias vector;
- `activity_regularizer`: regularization applied to the output;
- `kernel_constraint`: constraint applied to the kernel matrix;
- `bias_constraint`: constraint applied to the bias vector.

**Usage**
```py
# Load data
A, X, _, _, _, _, _, _ = citation.load_data('cora')

# Preprocessing operations
I = sp.identity(A.shape[0], dtype=A.dtype)
fltr = utils.normalize_adjacency(A + I)

# Model definition
X_in = Input(shape=(F, ))
fltr_in = Input((N, ))
output = APPNP(channels, mlp_channels)([X_in, fltr_in])
```

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional.py#L1515)</span>
### GINConv

```python
spektral.layers.GINConv(channels, mlp_channels=16, n_hidden_layers=0, epsilon=None, mlp_activation='relu', activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```


A Graph Isomorphism Network (GIN) as presented by
[Xu et al. (2018)](https://arxiv.org/abs/1810.00826).

**Mode**: single.

This layer computes for each node \(i\):
$$
Z_i = \textrm{MLP} ( (1 + \epsilon) \cdot X_i + \sum\limits_{j \in \mathcal{N}(i)} X_j)
$$
where \(X\) is the node features matrix and \(\textrm{MLP}\) is a
multi-layer perceptron.

**Input**

- Node features of shape `(n_nodes, n_features)` (with optional `batch`
dimension);
- Normalized and rescaled Laplacian of shape `(n_nodes, n_nodes)` (with
optional `batch` dimension);

**Output**

- Node features with the same shape of the input, but the last dimension
changed to `channels`.

**Arguments**

- `channels`: integer, number of output channels;
- `mlp_channels`: integer, number of channels in the inner MLP;
- `n_hidden_layers`: integer, number of hidden layers in the MLP (default 0)
- `epsilon`: unnamed parameter, see
[Xu et al. (2018)](https://arxiv.org/abs/1810.00826), and the equation above.
This parameter can be learned by setting `epsilon=None`, or it can be set
to a constant value, which is what happens by default (0). In practice, it
is safe to leave it to 0.
- `mlp_activation`: activation function for the MLP,
- `activation`: activation function to use;
- `use_bias`: whether to add a bias to the linear transformation;
- `kernel_initializer`: initializer for the kernel matrix;
- `bias_initializer`: initializer for the bias vector;
- `kernel_regularizer`: regularization applied to the kernel matrix;
- `bias_regularizer`: regularization applied to the bias vector;
- `activity_regularizer`: regularization applied to the output;
- `kernel_constraint`: constraint applied to the kernel matrix;
- `bias_constraint`: constraint applied to the bias vector.

**Usage**

```py
# Load data
A, X, _, _, _, _, _, _ = citation.load_data('cora')

# Preprocessing operations
fltr = utils.normalized_laplacian(A)
fltr = utils.rescale_laplacian(X, lmax=2)

# Model definition
X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)
output = GINConv(channels)([X_in, fltr_in])
```
