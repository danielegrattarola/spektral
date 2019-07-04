[pypi-image]: https://badge.fury.io/py/spektral.svg
[pypi-url]: https://pypi.org/project/spektral/
[coverage-image]: https://codecov.io/gh/danielegrattarola/spektral/branch/develop/graph/badge.svg
[coverage-url]: https://codecov.io/github/danielegrattarola/spektral?branch=develop
[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]: https://github.com/rusty1s/pytorch_geometric/blob/master/CONTRIBUTING.md


<img src="https://danielegrattarola.github.io/spektral/img/logo_dark.svg" width="50%"/>

[![PyPI Version][pypi-image]][pypi-url]
[![Code Coverage][coverage-image]][coverage-url]
[![Contributing][contributing-image]][contributing-url]


|WARNING|
|:------|
| Spektral is still a work in progress and may change substantially before the first proper release. The API is not mature enough to be considered stable, but we'll try to keep breaking changes to a minimum. Drop me an email if you want to help out with the development.|

## Welcome to Spektral
Spektral is a framework for relational representation learning, built in Python and based on the Keras API.
The main purpose of this project is to provide a simple, fast, and scalable environment for fast experimentation.

Spektral contains a wide set of tools to build graph neural networks, and implements some of the most popular layers for graph deep learning so that you only need to worry about creating your models.

Spektral is an open source project available on [Github](https://github.com/danielegrattarola/spektral).  
Read the documentation on [the official website](https://danielegrattarola.github.io/spektral).

## Relational Representation Learning
An important trait of human intelligence is the ability to model the world in terms of entities and relations, exploiting the knowledge that entities are connected (called the __relational inductive bias__) in order to make informed decisions.  
This relational representation of the world allows us to model a great deal of phenomena using graphs, from social networks to sentences, from the interactions of bodies in a gravitational field to the bonds between atoms in a molecule. 

Given their flexibility for representing knowledge in a structured way, graphs have been historically ubiquitous in computer science, and early efforts in artificial intelligence relied heavily on relational representations (e.g., Monte Carlo methods, Bayesian networks, etc.).  
Modern machine learning approaches, on the other hand, seemed to have diverted from such __hand-engineered__ representations in favour of learning from raw data, with deep learning methods leading the way on the quasi-totality of learning tasks.    
However, by placing artificial intelligence in the framework of relational inductive bias, as proposed by [Battaglia et al.](https://arxiv.org/abs/1806.01261), we see that even the most modern deep learning methods are designed to exploit the relational inductive biases of particular types of data. For instance, convolutional neural networks are based on exploiting the relation of __locality__ in grid-structured data, and recurrent neural networks are designed to exploit the __sequentiality__ of time series (i.e., chains of time steps).

Adding to this, in recent years graph neural networks (GNNs) have been proposed, in several formulations, as a general framework for exploiting arbitrary relational inductive biases on arbitrarily defined entities and relations, giving rise to the field of __relational representation learning__ (RRL).  
In other words, RRL consists of developing models that are able to deal with graphs natively, taking their topology and attributes into account when making a prediction, exactly like we do when we reason about the universe.

To read more about RRL and recent developments in the literature, an excellent starting point is [this paper by DeepMind](https://arxiv.org/abs/1806.01261).

## Why Keras?
The core GNN modules of Spektral are based on Keras, rather than libraries like TensorFlow or PyTorch.  
Because it's built as a Keras extension, Spektral works with all the different backends offered by Keras, so that you can quickly start experimenting with RRL 
without having to deal with the distracting low-level details.   
This also means that Spektral inherits all the core design principles of Keras, and that it can be seamlessly integrated in your Keras or TensorFlow projects. 

While Keras has a slightly higher computational overhead with respect to the "pure" deep learning frameworks, the speed of implementation of Keras's models largely makes up for the disadvantage. At the same time, Keras offers the same granular control as the other frameworks via direct access to the backend (`from keras import backend as K`), and native functions from the backend frameworks can directly be used at any point in a Keras model.

Spektral's accessory modules are built in Numpy/Scipy, so everything should work at almost-C-like speed and without compatibility issues.

## Installation
Spektral is developed with Python 3 in mind, although some modules may work as expected also in Python 2. However, [you should consider
switching to Python 3](https://python3statement.org/) if you haven't already.

The framework is tested for Ubuntu 16.04 and 18.04, but is should also work on other Linux distros and MacOS. Core functionalities should work on Windows, as well, although it is not fully supported for now. 

To install the required dependencies on Ubuntu run:

```bash
$ sudo apt install graphviz libgraphviz-dev libcgraph6
```

Some features of Spektral also require the following optional dependencies:

 - [RDKit](http://www.rdkit.org/docs/index.html), a library for cheminformatics and molecule manipulation (available through Anaconda);
 - [dyfunconn](https://dyfunconn.readthedocs.io/), a library to build functional connectivity networks (available through PyPi);

The simplest way to install Spektral is with PyPi: 

```bash
$ pip install spektral
```

To install Spektral from source, run this in a terminal:

```bash
$ git clone https://github.com/danielegrattarola/spektral.git
$ cd spektral
$ python setup.py install  # Or 'pip install .'
```

Note that the `setup.py` script will not attempt to install a backend for Keras, in order to not mess up any previous installation. 
It will, however, install Keras and its dependencies via PyPi (which may include the CPU version of TensorFlow).    
If you are already a Keras user, this should not impact you. If you're just getting started, then you may want to [install the GPU version of Tensorflow](https://www.tensorflow.org/install/) before installing Spektral.

Also note that some features of Spektral may depend explicitly on TensorFlow, although this dependency will be kept to a minimum.

## Contributing
Spektral is an open source project available [on Github](https://github.com/danielegrattarola/spektral), and contributions of all types are welcome. Feel free to open a pull request if you have something interesting that you want to add to the framework.
