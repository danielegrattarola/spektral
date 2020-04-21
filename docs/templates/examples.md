# Examples

This is a collection of example scripts that you can use as template to solve your own tasks. 

## Node classification
- [Node classification on citation networks with GCN](https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_gcn.py);
- [Node classification on citation networks with ChebNets](https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_cheby.py);
- [Node classification on citation networks with GAT](https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_gat.py);
- [Node classification on citation networks with ARMA](https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_arma.py);
- [Node classification on citation networks with SimpleGCN](https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_simple_gc.py);
- [Node classification on the Open Graph Benchmark dataset (ogbn-proteins)](https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/ogbn-proteins_gcn.py);

## Graph-level prediction

Batch mode:

- [Classification of synthetic graphs with GAT](https://github.com/danielegrattarola/spektral/blob/master/examples/graph_prediction/delaunay_batch.py);
- [Regression of molecular properties on QM9 with ECC](https://github.com/danielegrattarola/spektral/blob/master/examples/graph_prediction/qm9_batch.py);

Disjoint mode: 

- [Classification of synthetic graphs with TopK pooling](https://github.com/danielegrattarola/spektral/blob/master/examples/graph_prediction/BDGC_disjoint.py);
- [Regression of molecular properties on QM9 with ECC](https://github.com/danielegrattarola/spektral/blob/master/examples/graph_prediction/qm9_disjoint.py);

## Graph signal classification
- [Graph signal classification on MNIST (mixed mode)](https://github.com/danielegrattarola/spektral/blob/master/examples/other/graph_signal_classification_mnist.py);

## Other applications
- [Node clustering on citation networks with minCUT pooling (unsupervised)](https://github.com/danielegrattarola/spektral/blob/master/examples/other/node_clustering_mincut.py);

The following notebooks are available on Kaggle with more visualizations (maintained by [@kmader](https://github.com/kmader)):

- [MNIST Graph Deep Learning](https://www.kaggle.com/kmader/mnist-graph-deep-learning);
- [MNIST Graph Pooling](https://www.kaggle.com/kmader/mnist-graph-nn-with-pooling);
