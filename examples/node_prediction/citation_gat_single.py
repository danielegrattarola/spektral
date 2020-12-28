"""
This example implements the experiments on citation networks from the paper:

Graph Attention Networks (https://arxiv.org/abs/1710.10903)
Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
"""
import numpy as np
from tensorflow.random import set_seed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.datasets.citation import Citation
from spektral.layers import GATConv
from spektral.transforms import LayerPreprocess, AdjToSpTensor
from spektral.utils.fit_single import fit_single
set_seed(0)

# Load data
dataset = Citation('cora',
                   normalize_x=True,
                   transforms=[LayerPreprocess(GATConv), AdjToSpTensor()])

def mask_to_weights(mask):
    return mask.astype(np.float32) / np.count_nonzero(mask)

weights_tr, weights_va, weights_te = (mask_to_weights(mask) for mask in (
      dataset.mask_tr, dataset.mask_va, dataset.mask_te))

# Parameters
channels = 8           # Number of channels in each head of the first GAT layer
n_attn_heads = 8       # Number of attention heads in first GAT layer
dropout = 0.6          # Dropout rate for the features and adjacency matrix
l2_reg = 2.5e-4        # L2 regularization rate
learning_rate = 5e-3   # Learning rate
epochs = 20000         # Number of training epochs
patience = 100         # Patience for early stopping

N = dataset.n_nodes          # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = dataset.n_labels     # Number of classes

# Model definition
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True)

do_1 = Dropout(dropout)(x_in)
gc_1 = GATConv(channels,
               attn_heads=n_attn_heads,
               concat_heads=True,
               dropout_rate=dropout,
               activation='elu',
               kernel_regularizer=l2(l2_reg),
               attn_kernel_regularizer=l2(l2_reg),
               bias_regularizer=l2(l2_reg),
               )([do_1, a_in])
do_2 = Dropout(dropout)(gc_1)
gc_2 = GATConv(n_out,
               attn_heads=1,
               concat_heads=False,
               dropout_rate=dropout,
               activation='softmax',
               kernel_regularizer=l2(l2_reg),
               attn_kernel_regularizer=l2(l2_reg),
               bias_regularizer=l2(l2_reg),
               )([do_2, a_in])

# Build model
model = Model(inputs=[x_in, a_in], outputs=gc_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss=CategoricalCrossentropy(reduction='sum'),
              weighted_metrics=['acc'])
model.summary()

# Train model
graph = dataset[0]
x = graph.x
a = graph.a
y = graph.y
fit_single(model,
           ((x, a), y, weights_tr),
           ((x, a), y, weights_va),
           epochs=epochs,
           callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)])

# Evaluate model
print('Evaluating model.')
model.reset_metrics()
eval_results = model.test_on_batch((x, a), y, weights_te)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
