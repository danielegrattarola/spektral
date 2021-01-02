from spektral import datasets
from spektral.data import BatchLoader, DisjointLoader, SingleLoader

batch_size = 3


def test_citation():
    dataset = datasets.Cora()
    dataset = datasets.Citeseer(random_split=True)
    dataset = datasets.Pubmed(normalize_x=True)
    sl = SingleLoader(dataset)
    sl.load()


def test_graphsage():
    # Test only PPI because Travis otherwise runs into memory errors
    dataset = datasets.PPI()
    sl = SingleLoader(dataset)


def test_mnist():
    dataset = datasets.MNIST(k=8, noise_level=0.1)


def test_qm7():
    dataset = datasets.QM7()
    dl = DisjointLoader(dataset, batch_size=batch_size)
    dl.__next__()

    bl = BatchLoader(dataset, batch_size=batch_size)
    bl.__next__()


def test_qm9():
    dataset = datasets.QM9(amount=100)
    dl = DisjointLoader(dataset, batch_size=batch_size)
    dl.__next__()

    bl = BatchLoader(dataset, batch_size=batch_size)
    bl.__next__()


def test_tud():
    # Edge labels + edge attributes
    dataset = datasets.TUDataset("BZR_MD", clean=False)
    dl = DisjointLoader(dataset, batch_size=batch_size)
    dl.__next__()

    bl = BatchLoader(dataset, batch_size=batch_size)
    bl.__next__()

    # Node labels + node attributes + clean version
    dataset = datasets.TUDataset("ENZYMES", clean=True)
    dl = DisjointLoader(dataset, batch_size=batch_size)
    dl.__next__()

    bl = BatchLoader(dataset, batch_size=batch_size)
    bl.__next__()
