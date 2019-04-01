import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2

from spektral.datasets import mnist
from spektral.layers import GraphConv
from spektral.utils import init_logging, normalized_laplacian


def sp_matrix_to_sp_tensor(x):
    x = x.tocoo()
    return tf.SparseTensor(indices=np.array([x.row, x.col]).T,
                           values=x.data, dense_shape=x.shape)


# Parameters
l2_reg = 5e-4             # Regularization rate for l2
learning_rate = 1e-3      # Learning rate for SGD
batch_size = 32           # Batch size
epochs = 10               # Number of training epochs
es_patience = 200         # Patience fot early stopping
log_dir = init_logging()  # Create log directory and file

# Load data
X_train, y_train, X_val, y_val, X_test, y_test, adj = mnist.load_data()
X_train, X_val, X_test = X_train[..., None], X_val[..., None], X_test[..., None]
N = X_train.shape[-2]      # Number of nodes in the graphs
F = X_train.shape[-1]      # Node features dimensionality
n_out = y_train.shape[-1]  # Dimension of the target

fltr = normalized_laplacian(adj)

# Model definition
X_in = Input(shape=(N, F))
# Pass filter as a fixed tensor, otherwise Keras will complain about inputs of
# different rank.
G_in = Input(tensor=sp_matrix_to_sp_tensor(fltr))

graph_conv = GraphConv(32,
                       activation='elu',
                       kernel_regularizer=l2(l2_reg),
                       use_bias=True)([X_in, G_in])
graph_conv = GraphConv(32,
                       activation='elu',
                       kernel_regularizer=l2(l2_reg),
                       use_bias=True)([graph_conv, G_in])
flatten = Flatten()(graph_conv)
fc = Dense(512, activation='relu')(flatten)
output = Dense(n_out, activation='softmax')(fc)

# Build model
model = Model(inputs=[X_in, G_in], outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=es_patience),
    TensorBoard(log_dir=log_dir, batch_size=batch_size),
    ModelCheckpoint(log_dir + 'best_model.h5', save_best_only=True, save_weights_only=True)
]

# Train model
validation_data = (X_val, y_val)
model.fit(X_train,
          y_train,
          batch_size=batch_size,
          validation_data=validation_data,
          epochs=epochs,
          callbacks=callbacks)

# Load best model
model.load_weights(log_dir + 'best_model.h5')

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate(X_test,
                              y_test,
                              batch_size=batch_size)
print('Done.\n'
      'Test loss: {}\n'
      'Test acc: {}'.format(*eval_results))