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

from spektral.datasets import citation
from spektral.layers import ARMAConv

# Load data
dataset = 'cora'
A, X, y, train_mask, val_mask, test_mask = citation.load_data(dataset)

# Parameters
channels = 16           # Number of channels in the first layer
iterations = 1          # Number of iterations to approximate each ARMA(1)
order = 2               # Order of the ARMA filter (number of parallel stacks)
share_weights = True    # Share weights in each ARMA stack
N = X.shape[0]          # Number of nodes in the graph
F = X.shape[1]          # Original feature dimensionality
n_classes = y.shape[1]  # Number of classes
dropout = 0.5           # Dropout rate applied between layers
dropout_skip = 0.75     # Dropout rate for the internal skip connection of ARMA
l2_reg = 5e-5           # L2 regularization rate
learning_rate = 1e-2    # Learning rate
epochs = 20000          # Number of training epochs
es_patience = 100       # Patience for early stopping

# Preprocessing operations
fltr = ARMAConv.preprocess(A).astype('f4')
X = X.toarray()

# Model definition
X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)

gc_1 = ARMAConv(channels,
                iterations=iterations,
                order=order,
                share_weights=share_weights,
                dropout_rate=dropout_skip,
                activation='elu',
                gcn_activation='elu',
                kernel_regularizer=l2(l2_reg))([X_in, fltr_in])
gc_2 = Dropout(dropout)(gc_1)
gc_2 = ARMAConv(n_classes,
                iterations=1,
                order=1,
                share_weights=share_weights,
                dropout_rate=dropout_skip,
                activation='softmax',
                gcn_activation=None,
                kernel_regularizer=l2(l2_reg))([gc_2, fltr_in])

# Build model
model = Model(inputs=[X_in, fltr_in], outputs=gc_2)
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

# Train model
validation_data = ([X, fltr], y, val_mask)
model.fit([X, fltr],
          y,
          sample_weight=train_mask,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,  # Shuffling data means shuffling the whole graph
          callbacks=[
              EarlyStopping(patience=es_patience,  restore_best_weights=True)
          ])

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([X, fltr],
                              y,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
