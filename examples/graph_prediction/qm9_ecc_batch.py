"""
This example shows how to perform regression of molecular properties with the
QM9 database, using a GNN based on edge-conditioned convolutions in batch mode.
"""

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import BatchLoader
from spektral.datasets import QM9
from spektral.layers import ECCConv, GlobalSumPool, GraphMasking

################################################################################
# Config
################################################################################
learning_rate = 1e-3  # Learning rate
epochs = 10  # Number of training epochs
batch_size = 32  # Batch size

################################################################################
# Load data
################################################################################
dataset = QM9(amount=1000)  # Set amount=None to train on whole dataset

# Parameters
F = dataset.n_node_features  # Dimension of node features
S = dataset.n_edge_features  # Dimension of edge features
n_out = dataset.n_labels  # Dimension of the target

# Train/test split
idxs = np.random.permutation(len(dataset))
split = int(0.9 * len(dataset))
idx_tr, idx_te = np.split(idxs, [split])
dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]


################################################################################
# Build model
################################################################################
class Net(Model):
    def __init__(self):
        super().__init__()
        self.masking = GraphMasking()
        self.conv1 = ECCConv(32, activation="relu")
        self.conv2 = ECCConv(32, activation="relu")
        self.global_pool = GlobalSumPool()
        self.dense = Dense(n_out)

    def call(self, inputs):
        x, a, e = inputs
        x = self.masking(x)
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        output = self.global_pool(x)
        output = self.dense(output)

        return output


model = Net()
optimizer = Adam(learning_rate)
model.compile(optimizer=optimizer, loss="mse")

################################################################################
# Fit model
################################################################################
loader_tr = BatchLoader(dataset_tr, batch_size=batch_size, mask=True)
model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, epochs=epochs)

################################################################################
# Evaluate model
################################################################################
print("Testing model")
loader_te = BatchLoader(dataset_te, batch_size=batch_size, mask=True)
loss = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done. Test loss: {}".format(loss))
