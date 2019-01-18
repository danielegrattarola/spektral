import numpy as np
from cdg.embedding.manifold import SphericalManifold, HyperbolicManifold


# Euclidean manifold ###########################################################
def euclidean_distance(x, y):
    """
    Euclidean distance between points. Can be used as user-defined metric for
    sklearn.neighbors.DistanceMetric.
    :param x: one-dimensional np.array;
    :param y: one-dimensional np.array;
    :return: distance between the given points.
    """
    return np.linalg.norm(x - y, axis=-1, keepdims=True)


# Spherical manifold ###########################################################
def is_spherical(x, r=1.):
    """
    Boolean membership to spherical manifold.
    :param x: np.array, coordinates are assumed to be in the last axis;
    :param r: positive float, the radius of the CCM;
    :return: boolean np.array, True if the points are on the CCM.
    """
    return (x ** 2).sum(-1).astype(np.float32) == r ** 2


def spherical_clip(x, r=1.):
    """
    Clips points in the ambient space to a spherical CCM of radius `r`.
    :param x: np.array, coordinates are assumed to be in the last axis;
    :param r: positive float, the radius of the CCM;
    :return: np.array of same shape as x.
    """
    x = x.copy()
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return r * (x / norm)


def spherical_distance(x, y):
    """
    Geodesic distance between points on a spherical CCM. Can be used as
    user-defined metric for sklearn.neighbors.DistanceMetric.
    :param x: one-dimensional np.array;
    :param y: one-dimensional np.array;
    :return: distance between the given points.
    """
    return np.arccos(np.clip(np.dot(x, y.T), -1, 1))


# Hyperbolic manifold ##########################################################
def is_hyperbolic(x, r=-1.):
    """
    Boolean membership to hyperbolic manifold.
    :param x: np.array, coordinates are assumed to be in the last axis;
    :param r: negative float, the radius of the CCM;
    :return: boolean np.array, True if the points are on the CCM.
    """
    return ((x[..., :-1] ** 2).sum(-1) - x[..., -1] ** 2).astype(np.float32) == - r ** 2


def hyperbolic_clip(x, r=-1., axis=-1):
    """
    Clips points in the ambient space to a hyperbolic CCM of radius `r`, by f
    orcing the `axis` coordinate of the points to be
    \(X_{axis} = \sqrt{\sum\limits_{i \neq {axis}} X_{i}^{2} + r^{2}}\).
    :param x: np.array, coordinates are assumed to be in the last axis;
    :param r: negative float, the radius of the CCM;
    :param axis: int, the axis along which to clip;
    :return: np.array of same shape as x.
    """
    x = x.copy()
    free_components_idxs = np.delete(np.arange(x.shape[-1]), axis)
    x[..., axis] = np.sqrt(np.sum(x[..., free_components_idxs] ** 2, -1) + (r ** 2))
    return x


def hyperbolic_inner(x, y):
    """
    Computes the inner product between points in the pseudo-euclidean
    ambient space of a hyperbolic manifold.
    Works also for 2D arrays of points.
    :param x: np.array, coordinates are assumed to be in the last axis;
    :param y: np.array, coordinates are assumed to be in the last axis;
    :return: the inner product matrix.
    """
    minkowski_ipm = np.eye(x.shape[-1])
    minkowski_ipm[-1, -1] = -1
    inner = x.dot(minkowski_ipm).dot(y.T)
    return np.clip(inner, -np.inf, -1)


def hyperbolic_distance(x, y):
    """
    Geodesic distance between points on a hyperbolic CCM. Can be used as
    user-defined metric for sklearn.neighbors.DistanceMetric.
    :param x: one-dimensional np.array;
    :param y: one-dimensional np.array;
    :return: the computed distance.
    """
    inner = hyperbolic_inner(x, y)
    return np.arccosh(-inner)


# Generic CCM ##################################################################
def exp_map(x, r, tangent_point=None):
    """
    Let \(\mathcal{M}\) be a CCM of radius `r`, and \(T_{p}\mathcal{M}\) the
    tangent plane of the CCM at point \(p\) (`tangent_point`).
    This function maps a point `x` on the tangent plane to the CCM, using the
    Riemannian exponential map.
    :param x: np.array, point on the tangent plane (intrinsic coordinates);
    :param r: float, radius of the CCM;
    :param tangent_point: np.array, origin of the tangent plane on the CCM
    (extrinsic coordinates); if `None`, defaults to `[0., ..., 0., r]`.
    :return: the exp-map of x to the CCM (extrinsic coordinates).
    """
    extrinsic_dim = x.shape[-1] + 1
    if tangent_point is None:
        tangent_point = np.zeros((extrinsic_dim,))
        tangent_point[-1] = np.abs(r)
    if isinstance(tangent_point, np.ndarray):
        if tangent_point.shape != (extrinsic_dim,) and tangent_point.shape != (1, extrinsic_dim):
            raise ValueError('Expected tangent_point of shape ({0},) or (1, {0}), got {1}'.format(extrinsic_dim, tangent_point.shape))
        if tangent_point.ndim == 1:
            tangent_point = tangent_point[np.newaxis, ...]
        if not belongs(tangent_point, r)[0]:
            raise ValueError('Tangent point must belong to manifold {}'.format(tangent_point))
    else:
        raise TypeError('tangent_point must be np.array or None')

    if r > 0.:
        return SphericalManifold.exp_map(tangent_point, x)
    elif r < 0.:
        return HyperbolicManifold.exp_map(tangent_point, x)
    else:
        return x


def log_map(x, r, tangent_point=None):
    """
    Let \(\mathcal{M}\) be a CCM of radius `r` and \(T_{p}\mathcal{M}\) the
    tangent plane of the CCM at point \(p\) (`tangent_point`).
    This function maps a point `x` on the CCM to the tangent plane, using the
    Riemannian logarithmic map.
    :param x: np.array, point on the CCM (extrinsic coordinates);
    :param r: float, radius of the CCM;
    :param tangent_point: np.array, origin of the tangent plane on the CCM
    (extrinsic coordinates); if 'None', defaults to `[0., ..., 0., r]`.
    :return: the log-map of x to the tangent plane (intrinsic coordinates).
    """
    extrinsic_dim = x.shape[-1]
    if tangent_point is None:
        tangent_point = np.zeros((extrinsic_dim,))
        tangent_point[-1] = np.abs(r)
    if isinstance(tangent_point, np.ndarray):
        if tangent_point.shape != (extrinsic_dim,) and tangent_point.shape != (1, extrinsic_dim):
            raise ValueError('Expected tangent_point of shape ({0},) or (1, {0}), got {1}'.format(extrinsic_dim, tangent_point.shape))
        if tangent_point.ndim == 1:
            tangent_point = tangent_point[np.newaxis, ...]
        if not belongs(tangent_point, r)[0]:
            raise ValueError('Tangent point must belong to manifold {}'.format(tangent_point))
    else:
        raise TypeError('tangent_point must be np.ndarray or None')

    if r > 0.:
        return SphericalManifold.log_map(tangent_point, x)
    elif r < 0.:
        return HyperbolicManifold.log_map(tangent_point, x)
    else:
        return x


def belongs(x, r):
    """
    Boolean membership to CCM of radius `r`.
    :param x: np.array, coordinates are assumed to be in the last axis;
    :param r: float, the radius of the CCM;
    :return: boolean np.array, True if the points are on the CCM.
    """
    if r > 0.:
        return is_spherical(x, r)
    elif r < 0.:
        return is_hyperbolic(x, r)
    else:
        return np.ones(x.shape[:-1]).astype(np.float32)


def clip(x, r, axis=-1):
    """
    Clips points in the ambient space to a CCM of radius `r`.
    :param x: np.array, coordinates are assumed to be in the last axis;
    :param r: float, the radius of the CCM;
    :param axis: axis along which to clip points in the hyperbolic case (`r < 0`);
    :return: np.array of same shape as x.
    """
    if r > 0.:
        return spherical_clip(x, r)
    elif r < 0.:
        return hyperbolic_clip(x, r, axis=axis)
    else:
        return x


def get_distance(r):
    """
    :param r: float, the radius of the CCM;
    :return: the callable distance function for the CCM of radius `r`.
    """
    if r > 0.:
        return spherical_distance
    elif r < 0.:
        return hyperbolic_distance
    else:
        return euclidean_distance
