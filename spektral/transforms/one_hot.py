from spektral.utils import label_to_one_hot, one_hot


class OneHotLabels(object):
    """
    One-hot encodes the graph labels along the innermost dimension (also if they
    are simple scalars).

    Either `depth` or `labels` must be passed as argument.

    **Arguments**

    - `depth`: int, the size of the one-hot vector (labels are intended as
    indices for a vector of size `depth`);
    - `labels`: list or tuple, the possible values that the labels can take
    (labels are one-hot encoded according to where they are found in`labels`).
    """

    def __init__(self, depth=None, labels=None):
        self.depth = depth
        self.labels = labels
        if self.depth is None and self.labels is None:
            raise ValueError("Must specify either depth or labels.")

    def __call__(self, graph):
        if self.labels is not None:
            graph.y = label_to_one_hot(graph.y, self.labels)
        else:
            graph.y = one_hot(graph.y, self.depth)

        return graph
