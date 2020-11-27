"""
This example shows how to define your own dataset and use it to train a
non-trivial GNN with message-passing and pooling layers.
The script also shows how to implement fast training and evaluation functions
in disjoint mode, with early stopping and accuracy monitoring.

The dataset that we create is a simple synthetic task in which we have random
graphs with randomly-colored nodes. The goal is to classify each graph with the
color that occurs the most on its nodes. For example, given a graph with 2
colors and 3 nodes:

x = [[1, 0],
     [1, 0],
     [0, 1]],

the corresponding target will be [1, 0].
"""

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import Dataset, Graph, DisjointLoader
from spektral.layers import GCSConv, GlobalAvgPool
from spektral.layers.pooling import TopKPool
from spektral.transforms.normalize_adj import NormalizeAdj

################################################################################
# PARAMETERS
################################################################################
learning_rate = 1e-2       # Learning rate
epochs = 400               # Number of training epochs
es_patience = 10           # Patience for early stopping
batch_size = 32            # Batch size


################################################################################
# LOAD DATA
################################################################################
class MyDataset(Dataset):
    """
    A dataset of random colored graphs.
    The task is to classify each graph with the color which occurs the most in
    its nodes.
    The graphs have `n_colors` colors, of at least `n_min` and at most `n_max`
    nodes connected with probability `p`.
    """
    def __init__(self, n_samples, n_colors=3, n_min=10, n_max=100, p=0.1, **kwargs):
        self.n_samples = n_samples
        self.n_colors = n_colors
        self.n_min = n_min
        self.n_max = n_max
        self.p = p
        super().__init__(**kwargs)

    def read(self):
        def make_graph():
            n = np.random.randint(self.n_min, self.n_max)
            colors = np.random.randint(0, self.n_colors, size=n)

            # Node features
            x = np.zeros((n, self.n_colors))
            x[np.arange(n), colors] = 1

            # Edges
            a = np.random.rand(n, n) <= self.p
            a = np.maximum(a, a.T).astype(int)
            a = sp.csr_matrix(a)

            # Labels
            y = np.zeros((self.n_colors, ))
            color_counts = x.sum(0)
            y[np.argmax(color_counts)] = 1

            return Graph(x=x, a=a, y=y)

        # We must return a list of Graph objects
        return [make_graph() for _ in range(self.n_samples)]


dataset = MyDataset(1000, transforms=NormalizeAdj())

# Parameters
F = dataset.n_node_features  # Dimension of node features
n_out = dataset.n_labels     # Dimension of the target

# Train/valid/test split
idxs = np.random.permutation(len(dataset))
split_va, split_te = int(0.8 * len(dataset)), int(0.9 * len(dataset))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
dataset_tr = dataset[idx_tr]
dataset_va = dataset[idx_va]
dataset_te = dataset[idx_te]

loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(dataset_va, batch_size=batch_size)
loader_te = DisjointLoader(dataset_te, batch_size=batch_size)

################################################################################
# BUILD (unnecessarily big) MODEL
################################################################################
X_in = Input(shape=(F, ), name='X_in')
A_in = Input(shape=(None,), sparse=True)
I_in = Input(shape=(), name='segment_ids_in', dtype=tf.int32)

X_1 = GCSConv(32, activation='relu')([X_in, A_in])
X_1, A_1, I_1 = TopKPool(ratio=0.5)([X_1, A_in, I_in])
X_2 = GCSConv(32, activation='relu')([X_1, A_1])
X_2, A_2, I_2 = TopKPool(ratio=0.5)([X_2, A_1, I_1])
X_3 = GCSConv(32, activation='relu')([X_2, A_2])
X_3 = GlobalAvgPool()([X_3, I_2])
output = Dense(n_out, activation='softmax')(X_3)

# Build model
model = Model(inputs=[X_in, A_in, I_in], outputs=output)
opt = Adam(lr=learning_rate)
loss_fn = CategoricalCrossentropy()
acc_fn = CategoricalAccuracy()


################################################################################
# FIT MODEL
################################################################################
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions)
        loss += sum(model.losses)
        acc = acc_fn(target, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, acc


def evaluate(loader, ops_list):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        outs = [o(target, pred) for o in ops_list]
        output.append(outs)
    return np.mean(output, 0)


print('Fitting model')
current_batch = epoch = model_loss = model_acc = 0
best_val_loss = np.inf
best_weights = None
patience = es_patience

for batch in loader_tr:
    outs = train_step(*batch)

    model_loss += outs[0]
    model_acc += outs[1]
    current_batch += 1
    if current_batch == loader_tr.steps_per_epoch:
        model_loss /= loader_tr.steps_per_epoch
        model_acc /= loader_tr.steps_per_epoch
        epoch += 1

        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(loader_va, [loss_fn, acc_fn])
        print('Ep. {} - Loss: {:.2f} - Acc: {:.2f} - Val loss: {:.2f} - Val acc: {:.2f}'
              .format(epoch, model_loss, model_acc, val_loss, val_acc))

        # Check if loss improved for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = es_patience
            print('New best val_loss {:.3f}'.format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                print('Early stopping (best val_loss: {})'.format(best_val_loss))
                break
        model_loss = 0
        model_acc = 0
        current_batch = 0

################################################################################
# EVALUATE MODEL
################################################################################
print('Testing model')
model.set_weights(best_weights)  # Load best model
test_loss, test_acc = evaluate(loader_te, [loss_fn, acc_fn])
print('Done. Test loss: {:.4f}. Test acc: {:.2f}'.format(test_loss, test_acc))
