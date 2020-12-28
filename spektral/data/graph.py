import numpy as np
import scipy.sparse as sp


class Graph:
    """
    A container to represent a graph. The data associated with the Graph is
    stored in its attributes:

        - `x`, for the node features;
        - `a`, for the adjacency matrix;
        - `e`, for the edge attributes;
        - `y`, for the node or graph labels;

    All of these default to `None` if you don't specify them in the constructor.
    If you want to read all non-None attributes at once, you can call the
    `numpy()` method, which will return all data in a tuple (with the order
    defined above).

    Graphs also have the following attributes that are computed automatically
    from the data:

    - `n_nodes`: number of nodes;
    - `n_edges`: number of edges;
    - `n_node_features`: size of the node features, if available;
    - `n_edge_features`: size of the edge features, if available;
    - `n_labels`: size of the labels, if available;

    Any additional `kwargs` passed to the constructor will be automatically
    assigned as instance attributes of the graph.

    Data can be stored in Numpy arrays or Scipy sparse matrices, and labels can
    also be scalars.

    Spektral usually assumes that the different data matrices have specific
    shapes, although this is not strictly enforced to allow more flexibility.
    In general, node attributes should have shape `(n_nodes, n_node_features)` and the adjacency
    matrix should have shape `(n_nodes, n_nodes)`.

    Edge attributes can be stored in a dense format as arrays of shape
    `(n_nodes, n_nodes, n_edge_features)` or in a sparse format as arrays of shape `(n_edges, n_edge_features)`
    (so that you don't have to store all the zeros for missing edges). Most
    components of Spektral will know how to deal with both situations
    automatically.

    Labels can refer to the entire graph (shape `(n_labels, )`) or to each
    individual node (shape `(n_nodes, n_labels)`).

    **Arguments**

    - `x`: np.array, the node features (shape `(n_nodes, n_node_features)`);
    - `a`: np.array or scipy.sparse matrix, the adjacency matrix (shape `(n_nodes, n_nodes)`);
    - `e`: np.array, the edge features (shape `(n_nodes, n_nodes, n_edge_features)` or `(n_edges, n_edge_features)`);
    - `y`: np.array, the node or graph labels (shape `(n_nodes, n_labels)` or `(n_labels, )`);


    """

    def __init__(self, x=None, a=None, e=None, y=None, **kwargs):
        self.x = x
        self.a = a
        self.e = e
        self.y = y

        # Read extra kwargs
        for k, v in kwargs.items():
            self[k] = v

    def numpy(self):
        return tuple(ret for ret in [self.x, self.a, self.e, self.y] if ret is not None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __contains__(self, key):
        return key in self.keys

    def __repr__(self):
        return "Graph(n_nodes={}, n_node_features={}, n_edge_features={}, n_labels={})".format(
            self.n_nodes, self.n_node_features, self.n_edge_features, self.n_labels
        )

    @property
    def n_nodes(self):
        if self.x is not None:
            return self.x.shape[-2]
        elif self.a is not None:
            return self.a.shape[-1]
        else:
            return None

    @property
    def n_edges(self):
        if sp.issparse(self.a):
            return self.a.nnz
        elif isinstance(self.a, np.ndarray):
            return np.nonzero(self.a)
        else:
            return None

    @property
    def n_node_features(self):
        if self.x is not None:
            return self.x.shape[-1]
        else:
            return None

    @property
    def n_edge_features(self):
        if self.e is not None:
            return self.e.shape[-1]
        else:
            return None

    @property
    def n_labels(self):
        if self.y is not None:
            shp = np.shape(self.y)
            return 1 if len(shp) == 0 else shp[-1]
        else:
            return None

    @property
    def keys(self):
        keys = [
            key
            for key in self.__dict__.keys()
            if self[key] is not None and not key.startswith("__")
        ]
        return keys
