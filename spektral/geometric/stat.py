import numpy as np
from spektral.geometric.manifold import spherical_clip, exp_map


# Uniform ######################################################################
def spherical_uniform(size, dim=3, r=1.):
    """
    Samples points from a uniform distribution on a spherical manifold.
    Uniform sampling on the sphere can be achieved by sampling from a Gaussian
    in the ambient space of the CCM, and then projecting the samples onto the
    sphere.
    :param size: number of points to sample;
    :param dim: dimension of the ambient space;
    :param r: positive float, the radius of the CCM;
    :return: np.array of shape (size, dim).
    """
    samples = np.random.normal(0, 1, (size, dim))
    samples = spherical_clip(samples, r=r)
    return samples


def hyperbolic_uniform(size, dim=3, r=-1., low=-1., high=1., projection='upper'):
    """
    Samples points from a uniform distribution on a hyperbolic manifold. Uniform
    sampling on a hyperbolic CCM can be achieved by sampling from a uniform
    distribution in the ambient space of the CCM, and then projecting the
    samples onto the CCM.
    :param size: number of points to sample;
    :param dim: dimension of the ambient space;
    :param r: negative float, the radius of the CCM;
    :param low: lower bound of the uniform distribution from which to sample;
    :param high: upper bound of the uniform distribution from which to sample;
    :param projection: 'upper', 'lower', or 'both'. Whether to project points 
    always on the upper or lower branch of the hyperboloid, or on both based 
    on the sign of the last coordinate. 
    :return: np.array of shape (size, dim).
    """
    samples = np.random.uniform(low, high, (size, dim))
    if projection == 'both':
        sign = np.sign(samples[..., -1:])
    elif projection == 'upper':
        sign = 1
    elif projection == 'lower':
        sign = -1
    else:
        raise NotImplementedError('Possible projection modes: \'both\', '
                                  '\'upper\', \'lower\'.')
    samples[..., -1:] = sign * np.sqrt((samples[..., :-1] ** 2).sum(-1, keepdims=True) + r ** 2)

    return samples


def _ccm_uniform(size, dim=3, r=0., low=-1., high=1., projection='upper'):
    """
    Samples points from a uniform distribution on a constant-curvature manifold.
    If `r=0`, then points are sampled from a uniform distribution in the ambient
    space.
    :param size: number of points to sample;
    :param dim: dimension of the ambient space;
    :param r: float, the radius of the CCM;
    :param low: lower bound of the uniform distribution from which to sample;
    :param high: upper bound of the uniform distribution from which to sample;
    :param projection: 'upper', 'lower', or 'both'. Whether to project points 
    always on the upper or lower branch of the hyperboloid, or on both based 
    on the sign of the last coordinate.
    :return: np.array of shape (size, dim).
    """
    if r < 0.:
        return hyperbolic_uniform(size, dim=dim, r=r, low=low, high=high,
                                  projection=projection)
    elif r > 0.:
        return spherical_uniform(size, dim=dim, r=r)
    else:
        return np.random.uniform(low, high, (size, dim))


def ccm_uniform(size, dim=3, r=0., low=-1., high=1., projection='upper'):
    """
    Samples points from a uniform distribution on a constant-curvature manifold.
    If `r=0`, then points are sampled from a uniform distribution in the ambient
    space.
    If a list of radii is passed instead of a single scalar, then the sampling
    is repeated for each value in the list and the results are concatenated
    along the last axis (e.g., see [Grattarola et al. (2018)](https://arxiv.org/abs/1805.06299)).
    :param size: number of points to sample;
    :param dim: dimension of the ambient space;
    :param r: floats or list of floats, radii of the CCMs;
    :param low: lower bound of the uniform distribution from which to sample;
    :param high: upper bound of the uniform distribution from which to sample;
    :param projection: 'upper', 'lower', or 'both'. Whether to project points
    always on the upper or lower branch of the hyperboloid, or on both based
    on the sign of the last coordinate.
    :return: if `r` is a scalar, np.array of shape (size, dim). If `r` is a
    list, np.array of shape (size, len(r) * dim).
    """
    if isinstance(r, int) or isinstance(r, float):
        r = [r]
    elif isinstance(r, list) or isinstance(r, tuple):
        r = r
    else:
        raise TypeError('Radius must be either a single value, a list'
                        'of values (or a tuple).')
    to_ret = []
    for r_ in r:
        to_ret.append(_ccm_uniform(size, dim=dim, r=r_, low=low, high=high,
                                   projection=projection))
    return np.concatenate(to_ret, -1)


# Normal #######################################################################
def spherical_normal(size, tangent_point, r, dim=3, loc=0., scale=1.):
    """
    Samples points from a normal distribution on a spherical manifold.
    Normal sampling on the sphere works by sampling from a Gaussian on the
    tangent plane, and then projecting the sampled points onto the sphere using
    the Riemannian exponential map.
    :param size: number of points to sample;
    :param tangent_point: np.array, origin of the tangent plane on the CCM
    (extrinsic coordinates);
    :param dim: dimension of the ambient space;
    :param r: positive float, the radius of the CCM;
    :param loc: mean of the Gaussian on the tangent plane;
    :param scale: standard deviation of the Gaussian on the tangent plane;
    :return: np.array of shape (size, dim).
    """
    samples = np.random.normal(loc=loc, scale=scale, size=(size, dim - 1))
    samples = exp_map(samples, r, tangent_point)
    return samples


def hyperbolic_normal(size, tangent_point, r, dim=3, loc=0., scale=1.):
    """
    Samples points from a normal distribution on a hyperbolic manifold.
    Normal sampling on a hyperbolic CCM works by sampling from a Gaussian on the
    tangent plane, and then projecting the sampled points onto the CCM using
    the Riemannian exponential map.
    :param size: number of points to sample;
    :param tangent_point: np.array, origin of the tangent plane on the CCM
    (extrinsic coordinates);
    :param r: positive float, the radius of the CCM;
    :param dim: dimension of the ambient space;
    :param loc: mean of the Gaussian on the tangent plane;
    :param scale: standard deviation of the Gaussian on the tangent plane;
    :return: np.array of shape (size, dim).
    """
    samples = np.random.normal(loc=loc, scale=scale, size=(size, dim - 1))
    return exp_map(samples, r, tangent_point)


def _ccm_normal(size, dim=3, r=0., tangent_point=None, loc=0., scale=1.):
    """
    Samples points from a Gaussian distribution on a constant-curvature manifold.
    If `r=0`, then points are sampled from a Gaussian distribution in the
    ambient space.
    :param size: number of points to sample;
    :param tangent_point: np.array, origin of the tangent plane on the CCM
    (extrinsic coordinates); if 'None', defaults to `[0., ..., 0., r]`.
    :param r: float, the radius of the CCM;
    :param dim: dimension of the ambient space;
    :param loc: mean of the Gaussian on the tangent plane;
    :param scale: standard deviation of the Gaussian on the tangent plane;
    :return: np.array of shape (size, dim).
    """
    if tangent_point is None:
        tangent_point = np.zeros((dim, ))
        tangent_point[-1] = np.abs(r)
    if r < 0.:
        return hyperbolic_normal(size, tangent_point, r, dim=dim, loc=loc, scale=scale)
    elif r > 0.:
        return spherical_normal(size, tangent_point, r, dim=dim, loc=loc, scale=scale)
    else:
        return np.random.normal(loc, scale, (size, dim))


def ccm_normal(size, dim=3, r=0., tangent_point=None, loc=0., scale=1.):
    """
    Samples points from a Gaussian distribution on a constant-curvature manifold.
    If `r=0`, then points are sampled from a Gaussian distribution in the
    ambient space.
    If a list of radii is passed instead of a single scalar, then the sampling
    is repeated for each value in the list and the results are concatenated
    along the last axis (e.g., see [Grattarola et al. (2018)](https://arxiv.org/abs/1805.06299)).
    :param size: number of points to sample;
    :param tangent_point: np.array, origin of the tangent plane on the CCM
    (extrinsic coordinates); if 'None', defaults to `[0., ..., 0., r]`.
    :param r: floats or list of floats, radii of the CCMs;
    :param dim: dimension of the ambient space;
    :param loc: mean of the Gaussian on the tangent plane;
    :param scale: standard deviation of the Gaussian on the tangent plane;
    :return: if `r` is a scalar, np.array of shape (size, dim). If `r` is a
    list, np.array of shape (size, len(r) * dim).
    """
    if isinstance(r, int) or isinstance(r, float):
        r = [r]
    elif isinstance(r, list) or isinstance(r, tuple):
        r = r
    else:
        raise TypeError('Radius must be either a single value, a list'
                        'of values (or a tuple).')

    if tangent_point is None:
        tangent_point = [None] * len(r)
    elif isinstance(tangent_point, np.ndarray):
        tangent_point = [tangent_point]
    elif isinstance(tangent_point, list) or isinstance(tangent_point, tuple):
        pass
    else:
        raise TypeError('tangent_point must be either a single point or a'
                        'list of points.')

    if len(r) != len(tangent_point):
        raise ValueError('r and tangent_point must have the same length')

    to_ret = []
    for r_, tp_ in zip(r, tangent_point):
        to_ret.append(_ccm_normal(size, dim=dim, r=r_, tangent_point=tp_,
                                  loc=loc, scale=scale))
    return np.concatenate(to_ret, -1)


# Generic ######################################################################
def get_ccm_distribution(name):
    """
    :param name: 'uniform' or 'normal', name of the distribution.
    :return: the callable function for sampling on a generic CCM;
    """
    if name == 'uniform':
        return ccm_uniform
    elif name == 'normal':
        return ccm_normal
    else:
        raise ValueError('Possible distributions: \'uniform\', \'normal\'')
