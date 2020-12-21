"""
This example implements the experiments on citation networks from the paper:

Semi-Supervised Classification with Graph Convolutional Networks (https://arxiv.org/abs/1609.02907)
Thomas N. Kipf, Max Welling
"""
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GCNConv
from spektral.transforms import LayerPreprocess, AdjToSpTensor

# Load data
dataset = Citation('cora',
                   transforms=[LayerPreprocess(GCNConv), AdjToSpTensor()])

def mask_to_weights(mask):
      return mask / np.count_nonzero(mask)

weights_tr, weights_va, weights_te = (mask_to_weights(mask) for mask in (
      dataset.mask_tr, dataset.mask_va, dataset.mask_te))

# Parameters
channels = 16          # Number of channels in the first layer
dropout = 0.5          # Dropout rate for the features
l2_reg = 5e-4 / 2      # L2 regularization rate
learning_rate = 1e-2   # Learning rate
epochs = 200           # Number of training epochs
patience = 10          # Patience for early stopping
a_dtype = dataset[0].a.dtype  # Only needed for TF 2.1

N = dataset.n_nodes          # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = dataset.n_labels     # Number of classes

# Model definition
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True, dtype=a_dtype)

do_1 = Dropout(dropout)(x_in)
gc_1 = GCNConv(channels,
               activation='relu',
               kernel_regularizer=l2(l2_reg),
               use_bias=False)([do_1, a_in])
do_2 = Dropout(dropout)(gc_1)
gc_2 = GCNConv(n_out,
               activation='softmax',
               use_bias=False)([do_2, a_in])

# Build model
model = Model(inputs=[x_in, a_in], outputs=gc_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss=CategoricalCrossentropy(reduction='sum'),
              weighted_metrics=['acc'])
model.summary()

# Train model
loader_tr = SingleLoader(dataset, sample_weights=weights_tr)
loader_va = SingleLoader(dataset, sample_weights=weights_va)
model.fit(loader_tr.load(),
          steps_per_epoch=loader_tr.steps_per_epoch,
          validation_data=loader_va.load(),
          validation_steps=loader_va.steps_per_epoch,
          epochs=epochs,
          callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)])

# Evaluate model
print('Evaluating model.')
loader_te = SingleLoader(dataset, sample_weights=weights_te)
eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
