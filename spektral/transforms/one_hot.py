from spektral.utils import one_hot, label_to_one_hot


class OneHotLabels(object):
    def __init__(self, depth=None, labels=None):
        self.depth = depth
        self.labels = labels
        if self.depth is None and self.labels is None:
            raise ValueError('Must specify either depth or labels.')

    def __call__(self, graph):
        if self.labels is not None:
            graph.y = label_to_one_hot(graph.y, self.labels)
        else:
            graph.y = one_hot(graph.y, self.depth)

        return graph
