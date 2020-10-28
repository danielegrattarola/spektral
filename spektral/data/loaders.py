import copy

import numpy as np
import tensorflow as tf
from scipy import sparse as sp

from spektral.data.utils import prepend_none, output_signature, numpy_to_disjoint, numpy_to_batch, batch_generator
from spektral.layers.ops import sp_matrix_to_sp_tensor

version = tf.__version__.split('.')
major, minor = int(version[0]), int(version[1])
tf_loader_available = major > 2 and minor > 3


class Loader:
    def __init__(self, dataset, batch_size=1, epochs=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self._generator = batch_generator(
            self.dataset, batch_size=self.batch_size, epochs=self.epochs,
            shuffle=self.shuffle)
        self.steps_per_epoch = int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        nxt = self._generator.__next__()
        return self.collate(nxt)

    def tf(self):
        raise NotImplementedError

    def collate(self, data_list):
        raise NotImplementedError

    def _pack(self, data_list):
        return [list(elem) for elem in zip(*[g.numpy() for g in data_list])]


class BatchLoader(Loader):
    def tf(self):
        if not tf_loader_available:
            raise RuntimeError('Calling Loader.tf() requires TensorFlow 2.4 '
                               'or greater.')
        signature = copy.deepcopy(self.dataset.signature)
        for k in signature:
            signature[k]['shape'] = prepend_none(signature[k]['shape'])
        if 'a' in signature:
            # Adjacency matrix in batch mode is dense
            signature['a']['spec'] = tf.TensorSpec
        if 'e' in signature:
            # Edge attributes have an extra None dimension in batch mode
            signature['e']['shape'] = prepend_none(signature['e']['shape'])

        return tf.data.Dataset.from_generator(
            lambda: (_ for _ in self),
            output_signature=output_signature(signature)
        )

    def collate(self, data_list):
        data_packed = self._pack(data_list)
        y = np.array(data_packed[-1])
        ret = numpy_to_batch(*data_packed[:-1])

        return ret, y


class DisjointLoader(Loader):
    def tf(self):
        if not tf_loader_available:
            raise RuntimeError('Calling Loader.tf() requires TensorFlow 2.4 '
                               'or greater.')
        signature = copy.deepcopy(self.dataset.signature)
        if 'y' in signature:
            # Edge attributes have an extra None dimension in batch mode
            signature['y']['shape'] = prepend_none(signature['y']['shape'])

        if 'a' in signature:
            # Adjacency matrix in batch mode is sparse
            signature['a']['spec'] = tf.SparseTensorSpec

        signature['i'] = dict()
        signature['i']['spec'] = tf.TensorSpec
        signature['i']['shape'] = (None, )
        signature['i']['dtype'] = tf.as_dtype(tf.int64)

        return tf.data.Dataset.from_generator(
            lambda: (_ for _ in self),
            output_signature=output_signature(signature)
        )

    def collate(self, data_list):
        data_packed = self._pack(data_list)
        y = np.array(data_packed[-1])
        ret = numpy_to_disjoint(*data_packed[:-1])
        ret = list(ret)
        for i in range(len(ret)):
            if sp.issparse(ret[i]):
                ret[i] = sp_matrix_to_sp_tensor(ret[i])
        ret = tuple(ret)

        return ret, y