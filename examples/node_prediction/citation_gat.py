"""
This example implements the experiments on citation networks from the paper:

Graph Attention Networks (https://arxiv.org/abs/1710.10903)
Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
"""

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GATConv
from spektral.transforms import LayerPreprocess, AdjToSpTensor

# Load data
dataset = Citation('cora',
                   transforms=[LayerPreprocess(GATConv), AdjToSpTensor()])
mask_tr, mask_va, mask_te = dataset.mask_tr, dataset.mask_va, dataset.mask_te

# Parameters
channels = 8           # Number of channels in each head of the first GAT layer
n_attn_heads = 8       # Number of attention heads in first GAT layer
dropout = 0.6          # Dropout rate for the features and adjacency matrix
l2_reg = 5e-6          # L2 regularization rate
learning_rate = 5e-3   # Learning rate
epochs = 20000         # Number of training epochs
patience = 100         # Patience for early stopping
a_dtype = dataset[0].a.dtype  # Only needed for TF 2.1

N = dataset.n_nodes          # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = dataset.n_labels     # Number of classes

# Model definition
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True, dtype=a_dtype)

do_1 = Dropout(dropout)(x_in)
gc_1 = GATConv(channels,
               attn_heads=n_attn_heads,
               concat_heads=True,
               dropout_rate=dropout,
               activation='elu',
               kernel_regularizer=l2(l2_reg),
               attn_kernel_regularizer=l2(l2_reg)
               )([do_1, a_in])
do_2 = Dropout(dropout)(gc_1)
gc_2 = GATConv(n_out,
               attn_heads=1,
               concat_heads=False,
               dropout_rate=dropout,
               activation='softmax',
               kernel_regularizer=l2(l2_reg),
               attn_kernel_regularizer=l2(l2_reg)
               )([do_2, a_in])

# Build model
model = Model(inputs=[x_in, a_in], outputs=gc_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

# Train model
loader_tr = SingleLoader(dataset, sample_weights=mask_tr)
loader_va = SingleLoader(dataset, sample_weights=mask_va)
model.fit(loader_tr.load(),
          steps_per_epoch=loader_tr.steps_per_epoch,
          validation_data=loader_va.load(),
          validation_steps=loader_va.steps_per_epoch,
          epochs=epochs,
          callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)])

# Evaluate model
print('Evaluating model.')
loader_te = SingleLoader(dataset, sample_weights=mask_te)
eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
