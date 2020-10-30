from spektral.datasets import delaunay, qm9, citation, graphsage, mnist, tud
from spektral.data import DisjointLoader, BatchLoader


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


def test_graphsage():
    for dataset_name in ['ppi']:
        # Test only PPI because Travis otherwise fails
        graphsage.load_data(dataset_name)


def test_delaunay():
    adj, nf, labels = delaunay.generate_data(return_type='numpy', classes=[0, 1, 2])
    correctly_padded(adj, nf, None)
    assert adj.shape[0] == labels.shape[0]

    # Test that it doesn't crash
    delaunay.generate_data(return_type='networkx')


def test_mnist():
    mnist.load_data(k=8, noise_level=0.1)


def test_qm9():
    dataset = qm9.QM9(amount=100)
    dl = DisjointLoader(dataset, batch_size=3)
    dl.__next__()

    bl = BatchLoader(dataset, batch_size=3)
    bl.__next__()


def test_tud():
    tud.load_data('PROTEINS', clean=False)
    tud.load_data('ENZYMES', clean=True)
