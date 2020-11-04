"""
This example implements the experiments on citation networks from the paper:

Semi-Supervised Classification with Graph Convolutional Networks (https://arxiv.org/abs/1609.02907)
Thomas N. Kipf, Max Welling
"""

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GraphConv
from spektral.transforms import LayerPreprocess, AdjToSpTensor

# Load data
dataset = Citation('cora',
                   transforms=[LayerPreprocess(GraphConv), AdjToSpTensor()])
mask_tr, mask_va, mask_te = dataset.mask_tr, dataset.mask_va, dataset.mask_te

# Parameters
channels = 16          # Number of channels in the first layer
dropout = 0.5          # Dropout rate for the features
l2_reg = 5e-4 / 2      # L2 regularization rate
learning_rate = 1e-2   # Learning rate
epochs = 200           # Number of training epochs
patience = 10          # Patience for early stopping
N = dataset.N          # Number of nodes in the graph
F = dataset.F          # Original size of node features
n_out = dataset.n_out  # Number of classes

# Model definition
X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)

dropout_1 = Dropout(dropout)(X_in)
graph_conv_1 = GraphConv(channels,
                         activation='relu',
                         kernel_regularizer=l2(l2_reg),
                         use_bias=False)([dropout_1, fltr_in])
dropout_2 = Dropout(dropout)(graph_conv_1)
graph_conv_2 = GraphConv(n_out,
                         activation='softmax',
                         use_bias=False)([dropout_2, fltr_in])

# Build model
model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

# Train model
loader_tr = SingleLoader(dataset, sample_weights=mask_tr)
loader_va = SingleLoader(dataset, sample_weights=mask_va)
model.fit(loader_tr.tf(),
          steps_per_epoch=loader_tr.steps_per_epoch,
          validation_data=loader_va.tf(),
          validation_steps=loader_va.steps_per_epoch,
          epochs=epochs,
          callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)])

# Evaluate model
print('Evaluating model.')
loader_te = SingleLoader(dataset, sample_weights=mask_te)
eval_results = model.evaluate(loader_te.tf(), steps=loader_te.steps_per_epoch)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
