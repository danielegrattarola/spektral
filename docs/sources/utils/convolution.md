### degree


```python
spektral.utils.degree(adj)
```



Computes the degree matrix of the given adjacency matrix.

**Arguments**  

- ` adj`: rank 2 array or sparse matrix

**Return**  
 The degree matrix in sparse DIA format

----

### degree_power


```python
spektral.utils.degree_power(adj, pow)
```



Computes \(D^{p}\) from the given adjacency matrix. Useful for computing
normalised Laplacians.

**Arguments**  

- ` adj`: rank 2 array or sparse matrix

- ` pow`: exponent to which elevate the degree matrix

**Return**  
 The exponentiated degree matrix in sparse DIA format

----

### normalized_adjacency


```python
spektral.utils.normalized_adjacency(adj, symmetric=True)
```



Normalizes the given adjacency matrix using the degree matrix as either
\(D^{-1}A\) or \(D^{-1/2}AD^{-1/2}\) (symmetric normalization).

**Arguments**  

- ` adj`: rank 2 array or sparse matrix;

- ` symmetric`: boolean, compute symmetric normalization;

**Return**  
 The normalized adjacency matrix.

----

### laplacian


```python
spektral.utils.laplacian(adj)
```



Computes the Laplacian of the given adjacency matrix as \(D - A\).

**Arguments**  

- ` adj`: rank 2 array or sparse matrix;

**Return**  
 The Laplacian.

----

### normalized_laplacian


```python
spektral.utils.normalized_laplacian(adj, symmetric=True)
```



Computes a  normalized Laplacian of the given adjacency matrix as
\(I - D^{-1}A\) or \(I - D^{-1/2}AD^{-1/2}\) (symmetric normalization).

**Arguments**  

- ` adj`: rank 2 array or sparse matrix;

- ` symmetric`: boolean, compute symmetric normalization;

**Return**  
 The normalized Laplacian.

----

### localpooling_filter


```python
spektral.utils.localpooling_filter(adj, symmetric=True)
```



Computes the local pooling filter from the given adjacency matrix, as 
described by Kipf & Welling (2017).

**Arguments**  

- ` adj`: a np.array or scipy.sparse matrix of rank 2 or 3;

- ` symmetric`: boolean, whether to normalize the matrix as
\(D^{-\frac{1}{2}}AD^{-\frac{1}{2}}\) or as \(D^{-1}A\);

**Return**  
 The filter matrix, as dense np.array.

----

### chebyshev_filter


```python
spektral.utils.chebyshev_filter(adj, k, symmetric=True)
```



Computes the Chebyshev filter from the given adjacency matrix, as described
in Defferrard et at. (2016).

**Arguments**  

- ` adj`: a np.array or scipy.sparse matrix;

- ` k`: integer, the order up to which to compute the Chebyshev polynomials;

- ` symmetric`: boolean, whether to normalize the matrix as
\(D^{-\frac{1}{2}}AD^{-\frac{1}{2}}\) or as \(D^{-1}A\);

**Return**  
 A list of k+1 filter matrices, as np.arrays.
