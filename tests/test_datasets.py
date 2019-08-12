from spektral.datasets import delaunay, qm9, citation, mnist


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


def test_citation():
    for dataset_name in ['cora', 'citeseer', 'pubmed']:
        citation.load_data(dataset_name)
        citation.load_data(dataset_name, random_split=True)


def test_delaunay():
    adj, nf, labels = delaunay.generate_data('numpy', classes=[0, 1, 2])
    correctly_padded(adj, nf, None)
    assert adj.shape[0] == labels.shape[0]

    # Test that it doesn't crash
    delaunay.generate_data('networkx')


def test_mnist():
    mnist.load_data()


def test_qm9():
    adj, nf, ef, labels = qm9.load_data('numpy', amount=1000)
    correctly_padded(adj, nf, ef)
    assert adj.shape[0] == labels.shape[0]

    # Test that it doesn't crash
    qm9.load_data('networkx', amount=1000)
    qm9.load_data('sdf', amount=1000)
