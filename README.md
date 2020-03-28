<img src="https://danielegrattarola.github.io/spektral/img/logo_dark.svg" width="50%"/>

## Welcome to Spektral
Spektral is a Python library for graph deep learning, based on the Keras API and TensorFlow 2.
The main goal of this project is to provide a simple but flexible framework for creating graph neural networks (GNNs).

You can use Spektral for classifying the nodes of a network, predicting molecular properties, generating new graphs with GANs, clustering nodes, predicting links, and any other task where data is described by graphs. 

Spektral implements some of the most popular layers for graph deep learning, including: 

- [Graph Convolutional Networks (GCN)](https://arxiv.org/abs/1609.02907)
- [Chebyshev networks (ChebNets)](https://arxiv.org/abs/1606.09375)
- [GraphSAGE](https://arxiv.org/abs/1706.02216)
- [ARMA convolutions](https://arxiv.org/abs/1901.01343)
- [Edge-Conditioned Convolutions (ECC)](https://arxiv.org/abs/1704.02901)
- [Graph attention networks (GAT)](https://arxiv.org/abs/1710.10903)
- [Approximated Personalized Propagation of Neural Predictions (APPNP)](https://arxiv.org/abs/1810.05997)
- [Graph Isomorphism Networks (GIN)](https://arxiv.org/abs/1810.00826)

You can also find [pooling layers](https://danielegrattarola.github.io/spektral/layers/pooling/), including:

- [DiffPool](https://arxiv.org/abs/1806.08804)
- [MinCUT pooling](https://arxiv.org/abs/1907.00481)
- [Top-K pooling](http://proceedings.mlr.press/v97/gao19a/gao19a.pdf)
- [Self-Attention Graph (SAG) pooling](https://arxiv.org/abs/1904.08082)
- Global sum, average, and max pooling
- [Global gated attention pooling](https://arxiv.org/abs/1511.05493)

Spektral also includes lots of utilities for your graph deep learning projects.  

See how to [get started with Spektral](https://danielegrattarola.github.io/spektral/getting-started/) and have a look at the [examples](https://danielegrattarola.github.io/spektral/examples/) for some templates.

The source code of the project is available on [Github](https://github.com/danielegrattarola/spektral).  
Read the documentation [here](https://spektral.graphneural.network).

## Installation
Spektral is compatible with Python 3.5+, and is tested on Ubuntu 16.04+ and MacOS. 
Other Linux distros should work as well, but Windows is not supported for now. 

To install the required dependencies on Ubuntu run:

```bash
sudo apt install graphviz libgraphviz-dev libcgraph6
```

Some optional features of Spektral also depend on [RDKit](http://www.rdkit.org/docs/index.html), 
a library for cheminformatics and molecule manipulation (available through 
Anaconda).

The simplest way to install Spektral is from PyPi: 

```bash
pip install spektral
```

To install Spektral from source, run this in a terminal:

```bash
git clone https://github.com/danielegrattarola/spektral.git
cd spektral
python setup.py install  # Or 'pip install .'
```

To install Spektral on [Google Colab](https://colab.research.google.com/):

```jupyter
! apt install graphviz libgraphviz-dev libcgraph6
! pip install spektral
```

### TensorFlow 1 and Keras
Starting from version 0.3, Spektral only supports TensorFlow 2 and `tf.keras`.
The old version of Spektral, which is based on TensorFlow 1 and the stand-alone Keras library, is still available on the `tf1` branch on GitHub and can be installed from source:

```bash
git clone https://github.com/danielegrattarola/spektral.git
cd spektral
git checkout tf1
python setup.py install  # Or 'pip install .'
```

In the future, the TF1-compatible version of Spektral (<0.2) will receive bug fixes, but all new features will only support TensorFlow 2.   

## Contributing
Spektral is an open source project available [on Github](https://github.com/danielegrattarola/spektral), and contributions of all types are welcome. Feel free to open a pull request if you have something interesting that you want to add to the framework.
