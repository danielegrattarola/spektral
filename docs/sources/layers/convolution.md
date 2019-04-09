<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional.py#L8)</span>
### GraphConv

```python
spektral.layers.GraphConv(channels, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```


A graph convolutional layer as presented by [Kipf & Welling (2016)](https://arxiv.org/abs/1609.02907).

**Mode**: single, mixed, batch.

This layer computes the transformation:
$$  
Z = \sigma(AXW + b)
$$
where \(X\) is the node features matrix, \(A\) is the normalized Laplacian,
\(W\) is the convolution kernel, \(b\) is a bias vector, and \(\sigma\) is 
the activation function.

**Input**

- node features of shape `(batch, num_nodes, num_features)`, depending on the
mode;
- Laplacians of shape `(batch, num_nodes, num_nodes)`, depending on the mode.
The Laplacians can be computed from the adjacency matrices like in the
original paper using `utils.convolution.localpooling_filter`.

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
fltr = localpooling_filter(adj)  # Can be any pre-processing
...
X = Input(shape=(num_nodes, num_features))
filter = Input((num_nodes, num_nodes))
Z = GraphConv(channels, activation='relu')([X, filter])
...
model.fit([node_features, fltr], y)
```

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional.py#L141)</span>
### ChebConv

```python
spektral.layers.ChebConv(channels, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```


A Chebyshev convolutional layer as presented by [Defferrard et al. (2016)](https://arxiv.org/abs/1606.09375).

**Mode**: single, mixed, batch.

Given a list of Chebyshev polynomials \(T = [T_{1}, ..., T_{K}]\), 
this layer computes the transformation:
$$
Z = \sigma( \sum \limits_{k=1}^{K} T_{k} X W  + b)
$$
where \(X\) is the node features matrix, \(W\) is the convolution kernel, 
\(b\) is the bias vector, and \(\sigma\) is the activation function.

**Input**

- node features of shape `(batch, num_nodes, num_features)`, depending on the
mode;
- a list of Chebyshev filters of shape `(batch, num_nodes, num_nodes)`,
depending on the mode.
The filters can be generated from the adjacency matrices using
`utils.convolution.chebyshev_filter`.

**Output**

- node features with the same shape of the input, but the last dimension
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
fltr = chebyshev_filter(adj, K)
...
X = Input(shape=(num_nodes, num_features))
filter = Input((num_nodes, num_nodes))
Z = GraphConv(channels, activation='relu')([X, filter])
...
model.fit([node_features, fltr], y)
```

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional.py#L846)</span>
### ARMAConv

```python
spektral.layers.ARMAConv(channels, ARMA_D, ARMA_K=None, recurrent=False, gcn_activation='relu', dropout_rate=0.0, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```


A graph convolutional layer with ARMA(H, K) filters, as presented by
[Bianchi et al. (2019)](https://arxiv.org/abs/1901.01343).

**Mode**: single, mixed, batch.

This layer computes the transformation:
$$
X^{out} = \text{avgpool}\left(\sum \limits_{k=1}^K \bar{X}_k^{(T)} \right),
$$
where:
$$
\bar{X}_k^{(t + 1)} =  \sigma\left(\tilde{L}\bar{X}^{(t)}W^{(t)} + XV^{(t)}\right)
$$
is a graph convolutional skip layer implementing the recursive update to
approximate the ARMA filter, \(\tilde{L}\) is the Laplacian modified to
have a spectrum in \([0,,2]\), \(\bar{X}^{(0)} = X\), and \(W, V\) are
trainable kernels.

**Input**

- node features of shape `(batch, num_nodes, num_features)`, depending on the
mode;
- normalized Laplacians of shape `(batch, num_nodes, num_nodes)`, depending
on the mode.

**Output**

- node features with the same shape of the input, but the last dimension
changed to `channels`.

**Arguments**

- `channels`: integer, number of output channels;
- `ARMA_K`: order of the ARMA filter (combination of K ARMA_1 filters);
- `ARMA_D`: depth of each ARMA_1 filter (number of recursive updates);
- `recurrent`: whether to share each head's weights like a recurrent net;
- `gcn_activation`: activation function to use to compute the ARMA filter;
- `dropout_rate`: dropout rate for laplacian and output layer
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
fltr = localpooling_filter(adj)
...
X = Input(shape=(num_nodes, num_features))
filter = Input((num_nodes, num_nodes))
Z = ARMAConv(channels, activation='relu')([X, filter])
...
model.fit([node_features, fltr], y)
```

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional.py#L279)</span>
### EdgeConditionedConv

```python
spektral.layers.EdgeConditionedConv(channels, kernel_network=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```


An edge-conditioned convolutional layer as presented by [Simonovsky and
Komodakis (2017)](https://arxiv.org/abs/1704.02901).

**Mode**: batch.

This layer computes a transformation of the input \(X\), s.t. for each node
\(i\) we have:
$$
X^{out}_i =  \frac{1}{\mathcal{N}(i)} \sum\limits_{j \in \mathcal{N}(i)} F(E_{ji}) X_{j} + b
$$
where \(\mathcal{N}(i)\) represents the one-step neighbourhood of node \(i\),
\(F\) is a neural network that outputs the convolution kernel as a
function of edge attributes, \(E\) is the edge attributes matrix, and \(b\)
is a bias vector.

**Input**

- node features of shape `(batch, num_nodes, num_node_features)`, depending
on the mode;
- adjacency matrices of shape `(batch, num_nodes, num_nodes)`, depending on
the mode.
- edge features of shape `(batch, num_nodes, num_nodes, num_edge_features)`,
depending on the mode.

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
adj = add_eye_batch(adj)
...
nf = Input(shape=(num_nodes, num_node_features))
a = Input(shape=(num_nodes, num_nodes))
ef = Input(shape=(num_nodes, num_nodes, num_edge_features))
Z = EdgeConditionedConv(32, num_nodes, num_edge_features)([nf, a, ef])
...
model.fit([node_features, adj, edge_features], y)
```

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional.py#L474)</span>
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

- node features of shape `(num_nodes, num_features)`;
- adjacency matrices of shape `(num_nodes, num_nodes)`;

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
adj = normalize_sum_to_unity(adj)
...
X = Input(shape=(num_nodes, num_features))
A = Input((num_nodes, num_nodes))
Z = GraphAttention(channels, activation='relu')([X, A])
...
model.fit([node_features, fltr], y)
```

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional.py#L699)</span>
### GraphConvSkip

```python
spektral.layers.GraphConvSkip(channels, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```


A graph convolutional layer as presented by
[Kipf & Welling (2016)](https://arxiv.org/abs/1609.02907), with the addition
of skip connections.

**Mode**: single, mixed, batch.

This layer computes the transformation:
$$
Z = \sigma(A X W_1 + X_0 W_2 + b)
$$
where \(X\) is the node features matrix, \(X_0\) is the node features matrix
for the skip connection, \(A\) is the normalized laplacian,
\(W_1\) and \(W_2\) are the convolution kernels, \(b\) is a bias vector,
and \(\sigma\) is the activation function.

**Input**

- node features of shape `(batch, num_nodes, num_features)`, depending on the
mode;
- node features for the skip connection of shape
`(batch, num_nodes, num_features)`, depending on the mode;
- Laplacians of shape `(batch, num_nodes, num_nodes)`, depending on the mode.
The Laplacians can be computed from the adjacency matrices using
`utils.convolution.localpooling_filter`.

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
X = Input(shape=(num_nodes, num_features))
X_0 = Input(shape=(num_nodes, num_features))
filter = Input((num_nodes, num_nodes))
Z = GraphConvSkip(channels, activation='relu')([X, X_0, filter])
```

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional.py#L1173)</span>
### APPNP

```python
spektral.layers.APPNP(channels, mlp_channels, alpha=0.2, H=1, K=1, mlp_activation='relu', dropout_rate=0.0, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```


A graph convolutional layer implementing the APPNP operator, as presented by
[Klicpera et al. (2019)](https://arxiv.org/abs/1810.05997).
Implementation by Filippo Bianchi.

**Mode**: single, mixed, batch.

**Input**

- node features of shape `(batch, num_nodes, num_features)`, depending on the
mode;
- normalized Laplacians of shape `(batch, num_nodes, num_nodes)`, depending
on the mode.

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
I = sp.identity(adj.shape[0], dtype=adj.dtype)
fltr = utils.normalize_adjacency(adj + I)
...
X = Input(shape=(num_nodes, num_features))
filter = Input((num_nodes, num_nodes))
Z = APPNP(channels, mlp_channels)([X, filter])
...
model.fit([node_features, fltr], y)
```
