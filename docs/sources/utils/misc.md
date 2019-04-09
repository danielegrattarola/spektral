### batch_iterator


```python
spektral.utils.batch_iterator(data, batch_size=32, epochs=1, shuffle=True)
```



Iterates over the data for the given number of epochs, yielding batches of
size `batch_size`.

**Arguments**  

- ` data`: np.array or list of np.arrays with equal first dimension.

- ` batch_size`: number of samples in a batch

- ` epochs`: number of times to iterate over the data

- ` shuffle`: whether to shuffle the data at the beginning of each epoch
:yield: a batch of samples (or tuple of batches if X had more than one 
array). 

----

### set_trainable


```python
spektral.utils.set_trainable(model, toset)
```



Sets the trainable parameters of a Keras model and all its layers to toset.

**Arguments**  

- ` model`: a Keras Model

- ` toset`: boolean

**Return**  
 None

----

### pad_jagged_array


```python
spektral.utils.pad_jagged_array(x, target_shape, dtype=<class 'float'>)
```



Given a jagged array of arbitrary dimensions, zero-pads all elements in the
array to match the provided `target_shape`.

**Arguments**  

- ` x`: a np.array of dtype object, containing np.arrays of varying 
dimensions

- ` target_shape`: a tuple or list s.t. target_shape[i] >= x.shape[i]
for each x in X.
If `target_shape[i] = -1`, it will be automatically converted to X.shape[i], 
so that passing a target shape of e.g. (-1, n, m) will leave the first 
dimension of each element untouched (note that the creation of the output
array may fail if the result is again a jagged array). 

- ` dtype`: the dtype of the returned np.array

**Return**  
 A zero-padded np.array of shape `(X.shape[0], ) + target_shape`

----

### add_eye


```python
spektral.utils.add_eye(x)
```



Adds the identity matrix to the given matrix.

**Arguments**  

- ` x`: a rank 2 np.array or scipy.sparse matrix

**Return**  
 A rank 2 np.array as described above

----

### sub_eye


```python
spektral.utils.sub_eye(x)
```



Subtracts the identity matrix to the given matrix.

**Arguments**  

- ` x`: a rank 2 np.array or scipy.sparse matrix

**Return**  
 A rank 2 np.array as described above
