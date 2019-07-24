<img src="https://danielegrattarola.github.io/spektral/img/logo_dark.svg" width="50%"/>
<br><br>
<a href="https://pypi.python.org/pypi/spektral/">
    <img src="https://img.shields.io/pypi/pyversions/spektral.svg" style="width: auto !important" />
</a>
<a href="https://pypi.org/project/spektral/">
    <img src="https://badge.fury.io/py/spektral.svg" style="width: auto !important" />
</a>
<a href="https://travis-ci.org/danielegrattarola/spektral">
    <img src="https://travis-ci.org/danielegrattarola/spektral.svg?branch=master" style="width: auto !important" />
</a>
<a href="https://codecov.io/github/danielegrattarola/spektral?branch=develop">
    <img src="https://codecov.io/gh/danielegrattarola/spektral/branch/develop/graph/badge.svg" style="width: auto !important" />
</a>
<a href="https://github.com/danielegrattarola/spektral">
    <img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat" style="width: auto !important" />
</a>

|WARNING|
|:------|
| Spektral is still a work in progress and may change significantly before the first release. The API is not mature enough to be considered stable, but we'll try to keep breaking changes to a minimum.|

## Welcome to Spektral
Spektral is a Python library for relational representation learning, based on the Keras API.
The main goal of this project is to provide a simple but flexible framework for creating graph neural networks.

Spektral implements some of the most popular layers for graph deep learning, so that you only need to worry about creating your models.

The source of the project is available on [Github](https://github.com/danielegrattarola/spektral).  
Read the documentation on [the official website](https://danielegrattarola.github.io/spektral).

## Installation
Spektral is compatible with Python 3.5+, and is tested on Ubuntu 16.04 and 18.04. 
Other Linux distros and MacOS should work as well, but Windows is not supported for now. 

To install the required dependencies on Ubuntu run:

```bash
$ sudo apt install graphviz libgraphviz-dev libcgraph6
```

Some features of Spektral also require the following optional dependencies:

 - [RDKit](http://www.rdkit.org/docs/index.html), a library for cheminformatics and molecule manipulation (available through Anaconda);
 - [dyfunconn](https://dyfunconn.readthedocs.io/), a library to build functional connectivity networks (available through PyPi);

The simplest way to install Spektral is from PyPi: 

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
