class LayerPreprocess(object):
    """
    Applies the `preprocess` function of a convolutional Layer to the adjacency
    matrix.

    **Arguments**

    - `layer_class`: the class of a layer from `spektral.layers.convolutional`,
    or any Layer that implements a `preprocess(adj)` method.
    """

    def __init__(self, layer_class):
        self.layer_class = layer_class

    def __call__(self, graph):
        if graph.a is not None and hasattr(self.layer_class, "preprocess"):
            graph.a = self.layer_class.preprocess(graph.a)

        return graph
