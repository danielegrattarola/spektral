from spektral.layers.ops import sp_matrix_to_sp_tensor


class AdjToSpTensor(object):
    def __call__(self, graph):
        if graph.adj is not None:
            graph.adj = sp_matrix_to_sp_tensor(graph.adj)

        return graph
