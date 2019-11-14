### load_data


```python
spektral.datasets.citation.load_data(dataset_name='cora', normalize_features=True, random_split=False)
```



Loads a citation dataset (Cora, Citeseer or Pubmed) using the "Planetoid"
splits intialliy defined in [Yang et al. (2016)](https://arxiv.org/abs/1603.08861).
The train, test, and validation splits are given as binary masks.

Node attributes are bag-of-words vectors representing the most common words
in the text document associated to each node.
Two papers are connected if either one cites the other.
Labels represent the class of the paper.


**Arguments**  

- ` dataset_name`: name of the dataset to load (`'cora'`, `'citeseer'`, or
`'pubmed'`);

- ` normalize_features`: if True, the node features are normalized;

- ` random_split`: if True, return a randomized split (20/40/40) instead
of the default Planetoid split.

**Return**  

- Adjacency matrix;
- Node features;
- Labels;
- Three binary masks for train, validation, and test splits.
