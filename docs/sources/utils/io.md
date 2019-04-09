### load_binary


```python
spektral.utils.load_binary(filename)
```



Loads a pickled file.

**Arguments**  

- ` filename`: a string or file-like object

**Return**  
 The loaded object

----

### dump_binary


```python
spektral.utils.dump_binary(obj, filename)
```



Pickles and saves an object to file.

**Arguments**  

- ` obj`: the object to save

- ` filename`: a string or file-like object

----

### load_csv


```python
spektral.utils.load_csv(filename)
```



Loads a csv file with pandas.

**Arguments**  

- ` filename`: a string or file-like object

**Return**  
 The loaded csv

----

### dump_csv


```python
spektral.utils.dump_csv(df, filename, convert=False)
```



Dumps a pd.DataFrame to csv.

**Arguments**  

- ` df`: the pd.DataFrame to save or equivalent object

- ` filename`: a string or file-like object

- ` convert`: whether to attempt to convert the given object to
pd.DataFrame before saving the csv.

----

### load_dot


```python
spektral.utils.load_dot(filename, force_graph=True)
```



Loads a graph saved in .dot format.

**Arguments**  

- ` filename`: a string or file-like object

- ` force_graph`: whether to force a conversion to nx.Graph after loading.
This may be useful in the case of .dot files being loaded as nx.MultiGraph.

**Return**  
 The loaded graph

----

### dump_dot


```python
spektral.utils.dump_dot(obj, filename)
```



Dumps a nx.Graph to .dot file

**Arguments**  

- ` obj`: the nx.Graph (or equivalent) to save

- ` filename`: a string or file-like object

----

### load_npy


```python
spektral.utils.load_npy(filename)
```



Loads a file saved by np.save.

**Arguments**  

- ` filename`: a string or file-like object

**Return**  
 The loaded object

----

### dump_npy


```python
spektral.utils.dump_npy(obj, filename, zipped=False)
```



Saves an object to file using the numpy format.

**Arguments**  

- ` obj`: the object to save

- ` filename`: a string or file-like object

- ` zipped`: boolean, whether to save the object in the zipped format .npz
rather than .npy

----

### load_txt


```python
spektral.utils.load_txt(filename)
```



Loads a txt file using np.genfromtxt.

**Arguments**  

- ` filename`: a string or file-like object

**Return**  
 The loaded object

----

### dump_txt


```python
spektral.utils.dump_txt(obj, filename)
```



Saves an object to text file using np.savetxt.

**Arguments**  

- ` obj`: the object to save

- ` filename`: a string or file-like object

----

### load_sdf


```python
spektral.utils.load_sdf(filename, amount=None)
```



Load an .sdf file and return a list of molecules in the internal SDF format.

**Arguments**  

- ` filename`: target SDF file

- ` amount`: only load the first `amount` molecules from the file

**Return**  
 A list of molecules in the internal SDF format (see documentation).
