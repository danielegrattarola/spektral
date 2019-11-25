"""
This example implements the experiments on citation networks from the paper:

Graph Neural Networks with convolutional ARMA filters (https://arxiv.org/abs/1901.01343)
Filippo Maria Bianchi, Daniele Grattarola, Cesare Alippi, Lorenzo Livi
"""

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from spektral.datasets import citation
from spektral.layers import ARMAConv
from spektral.utils import normalized_laplacian, rescale_laplacian

# Load data
dataset = 'cora'
A, X, y, train_mask, val_mask, test_mask = citation.load_data(dataset)

# Parameters
ARMA_T = 1              # Depth of each ARMA_1 filter
ARMA_K = 2              # Number of parallel ARMA_1 filters
recurrent = True        # Share weights like a recurrent net in each head
N = X.shape[0]          # Number of nodes in the graph
F = X.shape[1]          # Original feature dimensionality
n_classes = y.shape[1]  # Number of classes
dropout_rate = 0.75     # Dropout rate applied to the input of GCN layers
l2_reg = 5e-4           # Regularization rate for l2
learning_rate = 1e-2    # Learning rate for SGD
epochs = 20000          # Number of training epochs
es_patience = 200       # Patience for early stopping

# Preprocessing operations
fltr = normalized_laplacian(A, symmetric=True)
fltr = rescale_laplacian(fltr, lmax=2)

# Model definition
X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)

dropout_1 = Dropout(dropout_rate)(X_in)
graph_conv_1 = ARMAConv(16,
                        iterations=ARMA_T,
                        order=ARMA_K,
                        recurrent=recurrent,
                        gcn_activation='elu',
                        dropout_rate=dropout_rate,
                        activation='elu',
                        kernel_regularizer=l2(l2_reg))([dropout_1, fltr_in])
dropout_2 = Dropout(dropout_rate)(graph_conv_1)
graph_conv_2 = ARMAConv(n_classes,
                        iterations=1,
                        order=1,
                        recurrent=recurrent,
                        gcn_activation=None,
                        dropout_rate=dropout_rate,
                        activation='softmax',
                        kernel_regularizer=l2(l2_reg))([dropout_2, fltr_in])

# Build model
model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_2)
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
validation_data = ([X, fltr], y, val_mask)
model.fit([X, fltr],
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
eval_results = model.evaluate([X, fltr],
                              y,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
