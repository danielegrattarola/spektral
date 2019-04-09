<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/pooling.py#L6)</span>
### GlobalAttentionPool

```python
spektral.layers.GlobalAttentionPool(channels=32, kernel_regularizer=None)
```


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

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/pooling.py#L82)</span>
### NodeAttentionPool

```python
spektral.layers.NodeAttentionPool(attn_kernel_initializer='glorot_uniform', kernel_regularizer=None, attn_kernel_regularizer=None, attn_kernel_constraint=None)
```


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
