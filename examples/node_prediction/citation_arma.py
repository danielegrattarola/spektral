"""
This example implements the experiments on citation networks from the paper:

Graph Neural Networks with convolutional ARMA filters (https://arxiv.org/abs/1901.01343)
Filippo Maria Bianchi, Daniele Grattarola, Cesare Alippi, Lorenzo Livi
"""

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import ARMAConv
from spektral.transforms import LayerPreprocess, AdjToSpTensor

# Load data
dataset = Citation('cora',
                   transforms=[LayerPreprocess(ARMAConv), AdjToSpTensor()])
mask_tr, mask_va, mask_te = dataset.mask_tr, dataset.mask_va, dataset.mask_te

# Parameters
channels = 16          # Number of channels in the first layer
iterations = 1         # Number of iterations to approximate each ARMA(1)
order = 2              # Order of the ARMA filter (number of parallel stacks)
share_weights = True   # Share weights in each ARMA stack
dropout_skip = 0.75    # Dropout rate for the internal skip connection of ARMA
dropout = 0.5          # Dropout rate for the features
l2_reg = 5e-5          # L2 regularization rate
learning_rate = 1e-2   # Learning rate
epochs = 20000         # Number of training epochs
patience = 100         # Patience for early stopping
a_dtype = dataset[0].a.dtype  # Only needed for TF 2.1

N = dataset.n_nodes          # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = dataset.n_labels     # Number of classes

# Model definition
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True, dtype=a_dtype)

gc_1 = ARMAConv(channels,
                iterations=iterations,
                order=order,
                share_weights=share_weights,
                dropout_rate=dropout_skip,
                activation='elu',
                gcn_activation='elu',
                kernel_regularizer=l2(l2_reg))([x_in, a_in])
gc_2 = Dropout(dropout)(gc_1)
gc_2 = ARMAConv(n_out,
                iterations=1,
                order=1,
                share_weights=share_weights,
                dropout_rate=dropout_skip,
                activation='softmax',
                gcn_activation=None,
                kernel_regularizer=l2(l2_reg))([gc_2, a_in])

# Build model
model = Model(inputs=[x_in, a_in], outputs=gc_2)
optimizer = Adam(learning_rate=learning_rate)
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
