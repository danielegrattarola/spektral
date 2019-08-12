"""
This example implements the experiments on citation networks using convolutional
layers from the paper:

Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (https://arxiv.org/abs/1606.09375)
Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst
"""

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from spektral.datasets import citation
from spektral.layers import ChebConv
from spektral.utils.convolution import chebyshev_filter

# Load data
dataset = 'cora'
A, X, y, train_mask, val_mask, test_mask = citation.load_data(dataset)

# Parameters
cheb_k = 2              # Max degree of the Chebyshev approximation
support = cheb_k + 1    # Total number of filters (k + 1)
N = X.shape[0]          # Number of nodes in the graph
F = X.shape[1]          # Original feature dimensionality
n_classes = y.shape[1]  # Number of classes
dropout_rate = 0.25     # Dropout rate applied to the input of GCN layers
l2_reg = 5e-4           # Regularization rate for l2
learning_rate = 1e-2    # Learning rate for SGD
epochs = 20000          # Number of training epochs
es_patience = 200       # Patience fot early stopping

# Preprocessing operations
fltr = chebyshev_filter(A, cheb_k)

# Model definition
X_in = Input(shape=(F, ))
# One input filter for each degree of the Chebyshev approximation
fltr_in = [Input((N, ), sparse=True) for _ in range(support)]

dropout_1 = Dropout(dropout_rate)(X_in)
graph_conv_1 = ChebConv(16,
                        activation='relu',
                        kernel_regularizer=l2(l2_reg),
                        use_bias=False)([dropout_1] + fltr_in)
dropout_2 = Dropout(dropout_rate)(graph_conv_1)
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

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_weighted_acc', patience=es_patience),
    ModelCheckpoint('best_model.h5', monitor='val_weighted_acc',
                    save_best_only=True, save_weights_only=True)
]

# Train model
validation_data = ([X] + fltr, y, val_mask)
model.fit([X] + fltr,
          y,
          sample_weight=train_mask,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,  # Shuffling data means shuffling the whole graph
          callbacks=callbacks)

# Load best model
model.load_weights('best_model.h5')

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([X] + fltr,
                              y,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
