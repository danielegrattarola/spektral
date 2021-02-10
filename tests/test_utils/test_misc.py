from spektral.utils import misc


def test_misc():
    l = [1, [2, 3], [4]]
    flattened = misc.flatten_list(l)
    assert flattened == [1, 2, 3, 4]
