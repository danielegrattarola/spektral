<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/base.py#L105)</span>
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

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/base.py#L181)</span>
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

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/base.py#L254)</span>
### CCMProjection

```python
spektral.layers.CCMProjection(r=None, kernel_initializer='glorot_uniform', kernel_regularizer=None, kernel_constraint=None)
```


Projects a tensor to a CCM depending on the value of `r`. Optionally, 
`r` can be learned via backpropagation.

**Input**

- tensor of shape `(batch_size, input_dim)`.

**Output**

- tensor of shape `(batch_size, input_dim)`, where each sample along the
0th axis is projected to the CCM.


**Arguments**  

- ` r`: radius of the CCM. If r is a number, then use it as fixed
radius. If `r='spherical'`, use a trainable weight as radius, with a
positivity constraint. If `r='hyperbolic'`, use a trainable weight
as radius, with a negativity constraint. If `r=None`, use a trainable
weight as radius, with no constraints (points will be projected to the
correct manifold based on the sign of the weight).

- ` kernel_initializer`: initializer for the kernel matrix;

- ` kernel_regularizer`: regularization applied to the kernel matrix;

- ` kernel_constraint`: constraint applied to the kernel matrix.

----

<span style="float:right;">[[source]](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/base.py#L347)</span>
### CCMMembership

```python
spektral.layers.CCMMembership(r=1.0, mode='average', sigma=1.0)
```


Computes the membership of the given points to a constant-curvature
manifold of radius `r`, as: 
$$
\mu(x) = \mathrm{exp}\left(\cfrac{-\big( \langle \vec x, \vec x \rangle - r^2 \big)^2}{2\sigma^2}\right).
$$

If `r=0`, then \(\mu(x) = 1\).
If more than one radius is given, inputs are evenly split across the 
last dimension and membership is computed for each radius-slice pair.
The output membership is returned according to the `mode` option.

**Input**

- tensor of shape `(batch_size, input_dim)`;

**Output**

- tensor of shape `(batch_size, output_size)`, where `output_size` is
computed according to the `mode` option;.


**Arguments**  

- ` r`: int ot list, radia of the CCMs.

- ` mode`: 'average' to return the average membership across CCMs, or
'concat' to return the membership for each CCM concatenated;

- ` sigma`: spread of the membership curve;
