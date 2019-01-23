import pytest
from keras.models import Sequential

from spektral.datasets import qm9
from spektral.layers import NodeAttentionPool, GlobalAttentionPool


def test_nap():
    adj, nf, ef, labels = qm9.load_data('numpy', amount=1000)
    N = nf.shape[-2]
    F = nf.shape[-1]

    model = Sequential()
    model.add(NodeAttentionPool(input_shape=(N, F)))

    assert model.output_shape == (None, F)


def test_gap():
    adj, nf, ef, labels = qm9.load_data('numpy', amount=1000)
    N = nf.shape[-2]
    F = nf.shape[-1]
    channels_out = 32

    model = Sequential()
    model.add(GlobalAttentionPool(channels_out, input_shape=(N, F)))

    assert model.output_shape == (None, channels_out)


if __name__ == '__main__':
    pytest.main([__file__, '--disable-pytest-warnings'])
