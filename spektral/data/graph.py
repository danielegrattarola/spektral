class Graph:
    """
    A container to represent a graph with:
        - node features;
        - adjacency matrix;
        - edge attributes;
        - node or graph labels;

    See the [data representation page](https://graphneural.network/data/) for
    more info.

    This class exposes the following attributes:

    - `N`: number of nodes;
    - `F`: size of the node features;
    - `S`: size of the edge features;

    **Arguments**

    - `x`: np.array, the node features (shape `(N, F)`);
    - `adj`: np.array or scipy.sparse matrix, the adjacency matrix (shape `(N, N)`);
    - `edge_attr`: np.array, the edge features (shape `(N, N, S)`);
    - `y`: np.array, the node or graph labels (shape `(N, n_labels)` or
           `(n_labels, )`);


    """
    def __init__(self, x=None, adj=None, edge_attr=None, y=None, **kwargs):
        self.x = x
        self.adj = adj
        self.edge_attr = edge_attr
        self.y = y
        # Read extra kwargs
        for k, v in kwargs.items():
            self[k] = v

        self.N = None if self.x is None else self.x.shape[-2]
        self.F = None if self.x is None else self.x.shape[-1]
        self.S = None if self.edge_attr is None else self.edge_attr.shape[-1]

    def numpy(self):
        return tuple(ret for ret in [self.x, self.adj, self.edge_attr, self.y]
                     if ret is not None)

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __repr__(self):
        return 'Graph(N={}, F={}, S={}, y={}'\
               .format(self.N, self.F, self.S, self.y)