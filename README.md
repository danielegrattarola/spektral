<img src="https://danielegrattarola.github.io/spektral/img/logo_dark.svg" style="max-width: 400px; width: 100%;"/>

# Welcome to Spektral
Spektral is a Python library for graph deep learning, based on the Keras API and TensorFlow 2.
The main goal of this project is to provide a simple but flexible framework for creating graph neural networks (GNNs).

You can use Spektral for classifying the users of a social network, predicting molecular properties, generating new graphs with GANs, clustering nodes, predicting links, and any other task where data is described by graphs. 

Spektral implements some of the most popular layers for graph deep learning, including: 

- [Graph Convolutional Networks (GCN)](https://arxiv.org/abs/1609.02907)
- [Chebyshev convolutions](https://arxiv.org/abs/1606.09375)
- [GraphSAGE](https://arxiv.org/abs/1706.02216)
- [ARMA convolutions](https://arxiv.org/abs/1901.01343)
- [Edge-Conditioned Convolutions (ECC)](https://arxiv.org/abs/1704.02901)
- [Graph attention networks (GAT)](https://arxiv.org/abs/1710.10903)
- [Approximated Personalized Propagation of Neural Predictions (APPNP)](https://arxiv.org/abs/1810.05997)
- [Graph Isomorphism Networks (GIN)](https://arxiv.org/abs/1810.00826)
- [Diffusional Convolutions](https://arxiv.org/abs/1707.01926)

and many others (see [convolutional layers](https://graphneural.network/layers/convolution/)).

You can also find [pooling layers](https://graphneural.network/layers/pooling/), including:

- [MinCut pooling](https://arxiv.org/abs/1907.00481)
- [DiffPool](https://arxiv.org/abs/1806.08804)
- [Top-K pooling](http://proceedings.mlr.press/v97/gao19a/gao19a.pdf)
- [Self-Attention Graph (SAG) pooling](https://arxiv.org/abs/1904.08082)
- Global pooling
- [Global gated attention pooling](https://arxiv.org/abs/1511.05493)
- [SortPool](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf)

Spektral also includes lots of utilities for representing, manipulating, and transforming graphs in your graph deep learning projects.

See how to [get started with Spektral](https://graphneural.network/getting-started/) and have a look at the [examples](https://danielegrattarola.github.io/spektral/examples/) for some templates.

The source code of the project is available on [Github](https://github.com/danielegrattarola/spektral).  
Read the documentation [here](https://graphneural.network).  

If you want to cite Spektral in your work, refer to our paper: 

> [Graph Neural Networks in TensorFlow and Keras with Spektral](https://arxiv.org/abs/2006.12138)<br>
> Daniele Grattarola and Cesare Alippi

## Installation
Spektral is compatible with Python 3.5+, and is tested on Ubuntu 16.04+ and MacOS. 
Other Linux distros should work as well, but Windows is not supported for now. 

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

```
! pip install spektral
```

## New in Spektral 1.0

The 1.0 release of Spektral is an important milestone for the library and brings many new features and improvements. 

If you have already used Spektral in your projects, the only major change that you need to be aware of is the new `datasets` API.

This is a summary of the new features and changes: 

- The new `Graph` and `Dataset` containers standardize how Spektral handles data. 
**This does not impact your models**, but makes it easier to use your data in Spektral.
- The new `Loader` class hides away all the complexity of creating graph batches. 
Whether you want to write a custom training loop or use Keras' famous model-dot-fit approach, you only need to worry about the training logic and not the data. 
- The new `transforms` module implements a wide variety of common operations on graphs, that you can now `apply()` to your datasets. 
- The new `GeneralConv` and `GeneralGNN` classes let you build models that are, well... general. Using state-of-the-art results from recent literature means that you don't need to worry about which layers or architecture to choose. The defaults will work well everywhere. 
- New datasets: QM7 and ModelNet10/40, and a new wrapper for OGB datasets. 
- Major clean-up of the library's structure and dependencies.
- New examples and tutorials.



## Contributing
Spektral is an open-source project available [on Github](https://github.com/danielegrattarola/spektral), and contributions of all types are welcome. 
Feel free to open a pull request if you have something interesting that you want to add to the framework.

The contribution guidelines are available [here](https://github.com/danielegrattarola/spektral/blob/master/CONTRIBUTING.md) and a list of feature requests is available [here](https://github.com/danielegrattarola/spektral/projects/1).
