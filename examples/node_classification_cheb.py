"""
This example implements the experiments on citation networks using convolutional
layers from the paper:

Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (https://arxiv.org/abs/1606.09375)
Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst
"""

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.datasets import citation
from spektral.layers import ChebConv
from spektral.utils.convolution import chebyshev_filter

# Load data
dataset = 'cora'
A, X, y, train_mask, val_mask, test_mask = citation.load_data(dataset)

# Parameters
channels = 16           # Number of channels in the first layer
cheb_k = 2              # Max degree of the Chebyshev approximation
support = cheb_k + 1    # Total number of filters (k + 1)
N = X.shape[0]          # Number of nodes in the graph
F = X.shape[1]          # Original feature dimensionality
n_classes = y.shape[1]  # Number of classes
dropout = 0.5           # Dropout rate applied to the features
l2_reg = 5e-4           # Regularization rate for l2
learning_rate = 1e-2    # Learning rate for SGD
epochs = 20000          # Number of training epochs
es_patience = 200       # Patience for early stopping

# Preprocessing operations
X = X.toarray()
fltr = chebyshev_filter(A, cheb_k)
fltr = [f.astype('f4') for f in fltr]

# Model definition
X_in = Input(shape=(F, ))
# One input filter for each degree of the Chebyshev approximation
fltr_in = [Input((N, ), sparse=True) for _ in range(support)]

dropout_1 = Dropout(dropout)(X_in)
graph_conv_1 = ChebConv(channels,
                        activation='relu',
                        kernel_regularizer=l2(l2_reg),
                        use_bias=False)([dropout_1] + fltr_in)
dropout_2 = Dropout(dropout)(graph_conv_1)
graph_conv_2 = ChebConv(n_classes,
                        activation='softmax',
                        use_bias=False)([dropout_2] + fltr_in)

# Build model
model = Model(inputs=[X_in] + fltr_in, outputs=graph_conv_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

# Train model
validation_data = ([X] + fltr, y, val_mask)
model.fit([X] + fltr,
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
eval_results = model.evaluate([X] + fltr,
                              y,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
