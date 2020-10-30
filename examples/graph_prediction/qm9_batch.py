"""
This example shows how to perform regression of molecular properties with the
QM9 database, using a GNN based on edge-conditioned convolutions in batch mode.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import BatchLoader
from spektral.datasets import qm9
from spektral.datasets.qm9 import QM9
from spektral.layers import EdgeConditionedConv, GlobalSumPool
from spektral.utils import label_to_one_hot

################################################################################
# PARAMETERS
################################################################################
learning_rate = 1e-3  # Learning rate
epochs = 10           # Number of training epochs
batch_size = 32       # Batch size

################################################################################
# LOAD DATA
################################################################################
dataset = QM9(amount=1000)  # Set amount=None to train on whole dataset

# Parameters
F = dataset.F          # Dimension of node features
S = dataset.S          # Dimension of edge features
n_out = dataset.n_out  # Dimension of the target

# Train/test split
idxs = np.random.permutation(len(dataset))
split = int(0.9 * len(dataset))
dataset_tr, dataset_te = dataset[:split], dataset[split:]

################################################################################
# BUILD MODEL
################################################################################
X_in = Input(shape=(None, F))
A_in = Input(shape=(None, None))
E_in = Input(shape=(None, None, S))

X_1 = EdgeConditionedConv(32, activation='relu')([X_in, A_in, E_in])
X_2 = EdgeConditionedConv(32, activation='relu')([X_1, A_in, E_in])
X_3 = GlobalSumPool()(X_2)
output = Dense(n_out)(X_3)

# Build model
model = Model(inputs=[X_in, A_in, E_in], outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='mse')
model.summary()

################################################################################
# FIT MODEL
################################################################################
loader_tr = BatchLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
model.fit(loader_tr,
          steps_per_epoch=loader_tr.steps_per_epoch,
          epochs=epochs)

################################################################################
# EVALUATE MODEL
################################################################################
print('Testing model')
loader_te = BatchLoader(dataset_te, batch_size=batch_size)
model_loss = model.evaluate(loader_te)
print('Done. Test loss: {}'.format(model_loss))
