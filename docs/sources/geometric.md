This module provides some functions to work with Riemannian geometry, and requires the [CDG](https://github.com/dan-zam/cdg) library to be installed on the system.

### exp_map


```python
spektral.geometric.manifold.exp_map(x, r, tangent_point=None)
```



Let \(\mathcal{M}\) be a CCM of radius `r`, and \(T_{p}\mathcal{M}\) the
tangent plane of the CCM at point \(p\) (`tangent_point`).
This function maps a point `x` on the tangent plane to the CCM, using the
Riemannian exponential map.

**Arguments**  

- ` x`: np.array, point on the tangent plane (intrinsic coordinates);

- ` r`: float, radius of the CCM;

- ` tangent_point`: np.array, origin of the tangent plane on the CCM
(extrinsic coordinates); if `None`, defaults to `[0., ..., 0., r]`.

**Return**  
 The exp-map of x to the CCM (extrinsic coordinates).

----

### log_map


```python
spektral.geometric.manifold.log_map(x, r, tangent_point=None)
```



Let \(\mathcal{M}\) be a CCM of radius `r` and \(T_{p}\mathcal{M}\) the
tangent plane of the CCM at point \(p\) (`tangent_point`).
This function maps a point `x` on the CCM to the tangent plane, using the
Riemannian logarithmic map.

**Arguments**  

- ` x`: np.array, point on the CCM (extrinsic coordinates);

- ` r`: float, radius of the CCM;

- ` tangent_point`: np.array, origin of the tangent plane on the CCM
(extrinsic coordinates); if 'None', defaults to `[0., ..., 0., r]`.

**Return**  
 The log-map of x to the tangent plane (intrinsic coordinates).

----

### belongs


```python
spektral.geometric.manifold.belongs(x, r)
```



Boolean membership to CCM of radius `r`.

**Arguments**  

- ` x`: np.array, coordinates are assumed to be in the last axis;

- ` r`: float, the radius of the CCM;

**Return**  
 Boolean np.array, True if the points are on the CCM.

----

### clip


```python
spektral.geometric.manifold.clip(x, r, axis=-1)
```



Clips points in the ambient space to a CCM of radius `r`.

**Arguments**  

- ` x`: np.array, coordinates are assumed to be in the last axis;

- ` r`: float, the radius of the CCM;

- ` axis`: axis along which to clip points in the hyperbolic case (`r < 0`);

**Return**  
 Np.array of same shape as x.

----

### get_distance


```python
spektral.geometric.manifold.get_distance(r)
```




**Arguments**  

- ` r`: float, the radius of the CCM;

**Return**  
 The callable distance function for the CCM of radius `r`.

----

### ccm_uniform


```python
spektral.geometric.stat.ccm_uniform(size, dim=3, r=0.0, low=-1.0, high=1.0, projection='upper')
```



Samples points from a uniform distribution on a constant-curvature manifold.
If `r=0`, then points are sampled from a uniform distribution in the ambient
space.
If a list of radii is passed instead of a single scalar, then the sampling
is repeated for each value in the list and the results are concatenated
along the last axis (e.g., see [Grattarola et al. (2018)](https://arxiv.org/abs/1805.06299)).

**Arguments**  

- ` size`: number of points to sample;

- ` dim`: dimension of the ambient space;

- ` r`: floats or list of floats, radii of the CCMs;

- ` low`: lower bound of the uniform distribution from which to sample;

- ` high`: upper bound of the uniform distribution from which to sample;

- ` projection`: 'upper', 'lower', or 'both'. Whether to project points
always on the upper or lower branch of the hyperboloid, or on both based
on the sign of the last coordinate.

**Return**  
 If `r` is a scalar, np.array of shape (size, dim). If `r` is a
list, np.array of shape (size, len(r) * dim).

----

### ccm_normal


```python
spektral.geometric.stat.ccm_normal(size, dim=3, r=0.0, tangent_point=None, loc=0.0, scale=1.0)
```



Samples points from a Gaussian distribution on a constant-curvature manifold.
If `r=0`, then points are sampled from a Gaussian distribution in the
ambient space.
If a list of radii is passed instead of a single scalar, then the sampling
is repeated for each value in the list and the results are concatenated
along the last axis (e.g., see [Grattarola et al. (2018)](https://arxiv.org/abs/1805.06299)).

**Arguments**  

- ` size`: number of points to sample;

- ` tangent_point`: np.array, origin of the tangent plane on the CCM
(extrinsic coordinates); if 'None', defaults to `[0., ..., 0., r]`.

- ` r`: floats or list of floats, radii of the CCMs;

- ` dim`: dimension of the ambient space;

- ` loc`: mean of the Gaussian on the tangent plane;

- ` scale`: standard deviation of the Gaussian on the tangent plane;

**Return**  
 If `r` is a scalar, np.array of shape (size, dim). If `r` is a
list, np.array of shape (size, len(r) * dim).

----

### get_ccm_distribution


```python
spektral.geometric.stat.get_ccm_distribution(name)
```




**Arguments**  

- ` name`: 'uniform' or 'normal', name of the distribution.

**Return**  
 The callable function for sampling on a generic CCM;
