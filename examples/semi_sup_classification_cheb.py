from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from spektral.datasets import citation
from spektral.layers import ChebConv
from spektral.utils.convolution import chebyshev_filter
from spektral.utils.logging import init_logging

# Load data
dataset = 'cora'
adj, node_features, y_train, y_val, y_test, train_mask, val_mask, test_mask = citation.load_data(dataset)

# Parameters
N = node_features.shape[0]    # Number of nodes in the graph
F = node_features.shape[1]    # Original feature dimensionality
n_classes = y_train.shape[1]  # Number of classes
dropout_rate = 0.5            # Dropout rate applied to the input of GCN layers
l2_reg = 5e-4                 # Regularization rate for l2
learning_rate = 1e-2          # Learning rate for SGD
epochs = 2000                 # Number of training epochs
es_patience = 10              # Patience fot early stopping
cheb_k = 2                    # Max degree of the Chebyshev approximation
support = cheb_k + 1          # Total number of filters (k + 1)
log_dir = init_logging()      # Create log directory and file

# Preprocessing operations
node_features = citation.preprocess_features(node_features)
fltr = chebyshev_filter(adj, cheb_k)

# Model definition
X_in = Input(shape=(F, ))
# One input filter for each degree of the Chebyshev approximation
fltr_in = [Input((N, ), sparse=True) for _ in range(support)]

dropout_1 = Dropout(dropout_rate)(X_in)
graph_conv_1 = ChebConv(256,
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
es_callback = EarlyStopping(monitor='val_weighted_acc', patience=es_patience)
tb_callback = TensorBoard(log_dir=log_dir, batch_size=N)
mc_callback = ModelCheckpoint(log_dir + 'best_model.h5', save_best_only=True,
                              save_weights_only=True)

# Train model
validation_data = ([node_features] + fltr, y_val, val_mask)
model.fit([node_features] + fltr,
          y_train,
          sample_weight=train_mask,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,  # Shuffling data means shuffling the whole graph
          callbacks=[es_callback, tb_callback, mc_callback])

# Load best model
model.load_weights(log_dir + 'best_model.h5')

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([node_features] + fltr,
                              y_test,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
