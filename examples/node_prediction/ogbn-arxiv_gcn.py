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

from spektral.datasets.ogb import OGB
from spektral.layers import GraphConv
from spektral.transforms import GCNFilter, AdjToSpTensor

# Load data
dataset_name = 'ogbn-arxiv'
ogb_dataset = NodePropPredDataset(dataset_name)
dataset = OGB(ogb_dataset, transforms=[GCNFilter(), AdjToSpTensor()])
graph = dataset[0]
x, adj, y = graph.x, graph.adj, graph.y
N = dataset.N

# Data splits
idx = ogb_dataset.get_idx_split()
idx_tr, idx_va, idx_te = idx["train"], idx["valid"], idx["test"]
mask_tr = np.zeros(N, dtype=bool)
mask_va = np.zeros(N, dtype=bool)
mask_te = np.zeros(N, dtype=bool)
mask_tr[idx_tr] = True
mask_va[idx_va] = True
mask_te[idx_te] = True
masks = [mask_tr, mask_va, mask_te]

# Parameters
channels = 256
dropout = 0.5                    # Dropout rate for the features
learning_rate = 1e-2             # Learning rate
epochs = 200                     # Number of training epochs
F = dataset.F                    # Original size of node features
n_out = ogb_dataset.num_classes  # OGB labels are sparse indices

# Model definition
X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)
X_1 = GraphConv(channels, activation='relu')([X_in, fltr_in])
X_1 = BatchNormalization()(X_1)
X_1 = Dropout(dropout)(X_1)
X_2 = GraphConv(channels, activation='relu')([X_1, fltr_in])
X_2 = BatchNormalization()(X_2)
X_2 = Dropout(dropout)(X_2)
X_3 = GraphConv(n_out, activation='softmax')([X_2, fltr_in])

# Build model
model = Model(inputs=[X_in, fltr_in], outputs=X_3)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy())
model.summary()


# Evaluation with OGB
evaluator = Evaluator(dataset_name)
def evaluate(X, fltr, y, model, masks, evaluator):
    p = model.predict_on_batch([X, fltr])
    p = p.argmax(-1)[:, None]
    tr_mask, va_mask, te_mask = masks
    tr_auc = evaluator.eval({'y_true': y[tr_mask], 'y_pred': p[tr_mask]})['acc']
    va_auc = evaluator.eval({'y_true': y[va_mask], 'y_pred': p[va_mask]})['acc']
    te_auc = evaluator.eval({'y_true': y[te_mask], 'y_pred': p[te_mask]})['acc']
    return tr_auc, va_auc, te_auc


# Train model
for i in range(1, 1 + epochs):
    tr_loss = model.train_on_batch([x, adj], y, sample_weight=mask_tr)
    tr_auc, va_auc, te_auc = evaluate(x, adj, y, model, masks, evaluator)
    print('Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Val acc: {:.3f} - Test acc: '
          '{:.3f}'.format(i, tr_loss, tr_auc, va_auc, te_auc))

# Evaluate model
print('Evaluating model.')
te_loss = model.test_on_batch([x, adj], y, sample_weight=mask_te)
tr_auc, va_auc, te_auc = evaluate(x, adj, y, model, masks, evaluator)
print('Done! Loss: {:.2f} - Test acc: {:.3f}'.format(te_loss, te_auc))
