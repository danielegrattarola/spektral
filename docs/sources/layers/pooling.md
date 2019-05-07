<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/pooling.py#L10)</span>
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

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/pooling.py#L73)</span>
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

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/pooling.py#L135)</span>
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

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/pooling.py#L197)</span>
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

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/pooling.py#L314)</span>
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
