import pytest
from spektral.datasets import aids, citation, delaunay, ieeg, kaggle_seizure, qm9


def correctly_padded(adj, nf, ef):
    assert adj.ndim == 3
    assert adj.shape[-1] == adj.shape[-2]
    if nf is not None:
        assert nf.ndim == 3
        assert adj.shape[-1] == nf.shape[-2]
    if ef is not None:
        assert ef.ndim == 4
        assert adj.shape[-1] == ef.shape[-2]
        assert adj.shape[-1] == ef.shape[-3]


def test_aids():
    adj, nf, ef, labels = aids.load_data('numpy')
    correctly_padded(adj, nf, ef)
    assert adj.shape[-1] == aids.MAX_K
    assert adj.shape[0] == labels.shape[0]

    # Test that it doesn't crash
    aids.load_data('networkx')


def test_citation():
    pass


def test_delaunay():
    adj, nf, labels = delaunay.load_data('numpy')
    correctly_padded(adj, nf, None)
    assert adj.shape[-1] == delaunay.MAX_K
    assert adj.shape[0] == labels.shape[0]

    # Test that it doesn't crash
    delaunay.load_data('networkx')


def test_ieeg():
    pass


def test_kaggle_seizure():
    datasets = kaggle_seizure.DATASETS
    for d in datasets:
        for p in kaggle_seizure.available_patients(d):
            adj_train, nf_train, ef_train, y_train, \
            adj_test, nf_test, ef_test = kaggle_seizure.load_data(p, d)
            correctly_padded(adj_train, nf_train, ef_train)
            correctly_padded(adj_test, nf_test, ef_test)


def test_qm9():
    adj, nf, ef, labels = qm9.load_data('numpy')
    correctly_padded(adj, nf, ef)
    assert adj.shape[-1] == qm9.MAX_K
    assert adj.shape[0] == labels.shape[0]

    # Test that it doesn't crash
    qm9.load_data('networkx')
    qm9.load_data('sdf')


if __name__ == '__main__':
    pytest.main([__file__, '--disable-pytest-warnings'])