"""
This example implements the same GCN example for node classification provided
with the [Open Graph Benchmark](https://ogb.stanford.edu).
See https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py
for the reference implementation.
"""
import numpy as np
from ogb.nodeproppred import NodePropPredDataset, Evaluator
from tensorflow.keras.layers import Input, Dropout, BatchNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from spektral.datasets import ogb
from spektral.layers import GraphConv


def evaluate(X, fltr, y, model, masks, evaluator):
    p = model.predict_on_batch([X, fltr])
    p = p.argmax(-1)[:, None]
    tr_mask, va_mask, te_mask = masks
    tr_auc = evaluator.eval({'y_true': y[tr_mask],
                             'y_pred': p[tr_mask]})['acc']
    va_auc = evaluator.eval({'y_true': y[va_mask],
                             'y_pred': p[va_mask]})['acc']
    te_auc = evaluator.eval({'y_true': y[te_mask],
                             'y_pred': p[te_mask]})['acc']
    return tr_auc, va_auc, te_auc


# Load data
dataset_name = 'ogbn-arxiv'
dataset = NodePropPredDataset(dataset_name)
evaluator = Evaluator(dataset_name)
graph, y = dataset[0]
X, A, _ = ogb.graph_to_numpy(graph)
N = A.shape[0]

# Data splits
idxs = dataset.get_idx_split()
tr_idx, va_idx, te_idx = idxs["train"], idxs["valid"], idxs["test"]
tr_mask = np.zeros(N, dtype=bool)
tr_mask[tr_idx] = True
va_mask = np.zeros(N, dtype=bool)
va_mask[va_idx] = True
te_mask = np.zeros(N, dtype=bool)
te_mask[te_idx] = True
masks = [tr_mask, va_mask, te_mask]

# Parameters
channels = 256
learning_rate = 1e-2
dropout = 0.5
epochs = 200
F = X.shape[1]
n_classes = dataset.num_classes

# Preprocessing operations
fltr = GraphConv.preprocess(A).astype('f4')

# Model definition
X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)
X_1 = GraphConv(channels, activation='relu')([X_in, fltr_in])
X_1 = BatchNormalization()(X_1)
X_1 = Dropout(dropout)(X_1)
X_2 = GraphConv(channels, activation='relu')([X_1, fltr_in])
X_2 = BatchNormalization()(X_2)
X_2 = Dropout(dropout)(X_2)
X_3 = GraphConv(n_classes, activation='softmax')([X_2, fltr_in])

# Build model
model = Model(inputs=[X_in, fltr_in], outputs=X_3)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy())
model.summary()

# Train model
for i in tqdm(range(1, 1 + epochs)):
    tr_loss = model.train_on_batch([X, fltr], y, sample_weight=tr_mask)
    tr_auc, va_auc, te_auc = evaluate(X, fltr, y, model, masks, evaluator)
    tqdm.write(
        'Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Val acc: {:.3f} - Test acc: {:.3f}'
        .format(i, tr_loss, tr_auc, va_auc, te_auc)
    )

# Evaluate model
print('Evaluating model.')
te_loss = model.test_on_batch([X, fltr], y, sample_weight=te_mask)
tr_auc, va_auc, te_auc = evaluate(X, fltr, y, model, masks, evaluator)
print('Done! Loss: {:.2f} - Test acc: {:.3f}'.format(te_loss, te_auc))
