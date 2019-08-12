<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/pooling.py#L12)</span>
### TopKPool

```python
spektral.layers.TopKPool(ratio, return_mask=False, sigmoid_gating=False, kernel_initializer='glorot_uniform', kernel_regularizer=None, activity_regularizer=None, kernel_constraint=None)
```


A gPool/Top-K layer as presented by
[Gao & Ji (2017)](http://proceedings.mlr.press/v97/gao19a/gao19a.pdf) and
[Cangea et al.](https://arxiv.org/abs/1811.01287).

This layer computes the following operations:

$$
y = \cfrac{Xp}{\| p \|}; \;\;\;\;
\textrm{idx} = \textrm{rank}(y, k); \;\;\;\;
\bar X = (X \odot \textrm{tanh}(y))_{\textrm{idx}}; \;\;\;\;
\bar A = A^2_{\textrm{idx}, \textrm{idx}}
$$

where \( \textrm{rank}(y, k) \) returns the indices of the top k values of
\( y \), and \( p \) is a learnable parameter vector of size \(F\).
Note that the the gating operation \( \textrm{tanh}(y) \) (Cangea et al.)
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

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/pooling.py#L169)</span>
### MinCutPool

```python
spektral.layers.MinCutPool(k, h=None, return_mask=True, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```


A minCUT pooling layer as presented by [Bianchi et al.](https://arxiv.org/abs/1907.00481).

**Mode**: single, batch.

This layer computes a soft clustering \(S\) of the input graphs using a MLP,
and reduces graphs as follows:

$$
A^{pool} = S^T A S; X^{pool} = S^T X;
$$

Besides training the MLP, two additional unsupervised loss terms to ensure
that the cluster assignment solves the minCUT optimization problem.
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

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/pooling.py#L386)</span>
### GlobalSumPool

```python
spektral.layers.GlobalSumPool()
```


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


----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/pooling.py#L451)</span>
### GlobalAvgPool

```python
spektral.layers.GlobalAvgPool()
```


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

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/pooling.py#L515)</span>
### GlobalMaxPool

```python
spektral.layers.GlobalMaxPool()
```


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

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/pooling.py#L579)</span>
### GlobalAttentionPool

```python
spektral.layers.GlobalAttentionPool(channels=32, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```


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

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/pooling.py#L701)</span>
### GlobalAttnSumPool

```python
spektral.layers.GlobalAttnSumPool(attn_kernel_initializer='glorot_uniform', kernel_regularizer=None, attn_kernel_regularizer=None, attn_kernel_constraint=None)
```


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
