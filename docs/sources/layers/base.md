<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/base.py#L7)</span>
### InnerProduct

```python
spektral.layers.InnerProduct(trainable_kernel=False, activation=None, kernel_initializer='glorot_uniform', kernel_regularizer=None, activity_regularizer=None, kernel_constraint=None)
```


Computes the inner product between elements of a given 2d tensor \(x\): 
$$
\langle x, x \rangle = xx^T.
$$

**Mode**: single.

**Input**

- rank 2 tensor of shape `(input_dim_1, input_dim_2)` (e.g. node features
of shape `(num_nodes, num_features)`);

**Output**

- rank 2 tensor of shape `(input_dim_1, input_dim_1)`


**Arguments**  

- ` trainable_kernel`: add a trainable square matrix between the inner
product (i.e., `x.dot(w).dot(x.T)`);

- ` activation`: activation function to use;

- ` kernel_initializer`: initializer for the kernel matrix;

- ` kernel_regularizer`: regularization applied to the kernel;

- ` activity_regularizer`: regularization applied to the output;

- ` kernel_constraint`: constraint applied to the kernel;

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/base.py#L83)</span>
### MinkowskiProduct

```python
spektral.layers.MinkowskiProduct(input_dim_1=None, activation=None, activity_regularizer=None)
```


Computes the hyperbolic inner product between elements of a given 2d tensor
\(x\): 
$$
\langle x, x \rangle = x \,
\begin{pmatrix}
I_{d\times d} & 0 \\ 0 & -1
\end{pmatrix} \,x^T.
$$

**Mode**: single.

**Input**

- rank 2 tensor of shape `(input_dim_1, input_dim_2)` (e.g. node features
of shape `(num_nodes, num_features)`);

**Output**

- rank 2 tensor of shape `(input_dim_1, input_dim_1)`


**Arguments**  

- ` input_dim_1`: first dimension of the input tensor; set this if you
encounter issues with shapes in your model, in order to provide an explicit
output shape for your layer.

- ` activation`: activation function to use;

- ` activity_regularizer`: regularization applied to the output;
