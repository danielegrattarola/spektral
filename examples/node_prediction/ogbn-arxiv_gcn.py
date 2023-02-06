"""
This example implements the same GCN example for node classification provided
with the [Open Graph Benchmark](https://ogb.stanford.edu).
See https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py
for the reference implementation.
"""
import numpy as np
import tensorflow as tf
from ogb.nodeproppred import Evaluator, NodePropPredDataset
from tensorflow.keras.layers import BatchNormalization, Dropout, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.datasets.ogb import OGB
from spektral.layers import GCNConv
from spektral.transforms import AdjToSpTensor, GCNFilter

# Load data
dataset_name = "ogbn-arxiv"
ogb_dataset = NodePropPredDataset(dataset_name)
dataset = OGB(ogb_dataset, transforms=[GCNFilter(), AdjToSpTensor()])
graph = dataset[0]
x, adj, y = graph.x, graph.a, graph.y

# Parameters
channels = 256  # Number of channels for GCN layers
dropout = 0.5  # Dropout rate for the features
learning_rate = 1e-2  # Learning rate
epochs = 200  # Number of training epochs

N = dataset.n_nodes  # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = ogb_dataset.num_classes  # OGB labels are sparse indices

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

# Model definition
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True)
x_1 = GCNConv(channels, activation="relu")([x_in, a_in])
x_1 = BatchNormalization()(x_1)
x_1 = Dropout(dropout)(x_1)
x_2 = GCNConv(channels, activation="relu")([x_1, a_in])
x_2 = BatchNormalization()(x_2)
x_2 = Dropout(dropout)(x_2)
x_3 = GCNConv(n_out, activation="softmax")([x_2, a_in])

# Build model
model = Model(inputs=[x_in, a_in], outputs=x_3)
optimizer = Adam(learning_rate=learning_rate)
loss_fn = SparseCategoricalCrossentropy()
model.summary()


# Training function
@tf.function
def train(inputs, target, mask):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target[mask], predictions[mask]) + sum(model.losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# Evaluation with OGB
evaluator = Evaluator(dataset_name)


def evaluate(x, a, y, model, masks, evaluator):
    p = model([x, a], training=False)
    p = p.numpy().argmax(-1)[:, None]
    tr_mask, va_mask, te_mask = masks
    tr_auc = evaluator.eval({"y_true": y[tr_mask], "y_pred": p[tr_mask]})["acc"]
    va_auc = evaluator.eval({"y_true": y[va_mask], "y_pred": p[va_mask]})["acc"]
    te_auc = evaluator.eval({"y_true": y[te_mask], "y_pred": p[te_mask]})["acc"]
    return tr_auc, va_auc, te_auc


# Train model
for i in range(1, 1 + epochs):
    tr_loss = train([x, adj], y, mask_tr)
    tr_acc, va_acc, te_acc = evaluate(x, adj, y, model, masks, evaluator)
    print(
        "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Val acc: {:.3f} - Test acc: "
        "{:.3f}".format(i, tr_loss, tr_acc, va_acc, te_acc)
    )

# Evaluate model
print("Evaluating model.")
tr_acc, va_acc, te_acc = evaluate(x, adj, y, model, masks, evaluator)
print("Done! - Test acc: {:.3f}".format(te_acc))
