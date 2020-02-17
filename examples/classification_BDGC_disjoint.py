"""
This example shows how to perform graph classification with a synthetic
benchmark dataset created by F. M. Bianchi (https://github.com/FilippoMB/Benchmark_dataset_for_graph_classification),
using a GNN with convolutional and pooling blocks in disjoint mode.
"""

import os

import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from spektral.layers import GraphConvSkip, GlobalAvgPool
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.layers.pooling import MinCutPool
from spektral.utils import batch_iterator
from spektral.utils.convolution import normalized_adjacency
from spektral.utils.data import Batch


def evaluate(A_list, X_list, y_list, ops, batch_size):
    batches = batch_iterator([A_list, X_list, y_list], batch_size=batch_size)
    output = []
    for b in batches:
        X, A, I = Batch(b[0], b[1]).get('XAI')
        A = sp_matrix_to_sp_tensor(A)
        y = b[2]
        pred = model([X, A, I])
        outs = [o(pred, y) for o in ops]
        output.append(outs)
    return np.mean(output, 0)


################################################################################
# PARAMETERS
################################################################################
n_channels = 32            # Channels per layer
activ = 'elu'              # Activation in GNN and mincut
GNN_l2 = 1e-4              # l2 regularisation of GNN
pool_l2 = 1e-4             # l2 regularisation for mincut
epochs = 500               # Number of training epochs
es_patience = 50           # Patience for early stopping
learning_rate = 1e-3       # Learning rate
batch_size = 1             # Batch size. NOTE: it MUST be 1 when using MinCutPool and DiffPool
data_url = 'https://github.com/FilippoMB/Benchmark_dataset_for_graph_classification/raw/master/datasets/'
dataset_name = 'easy.npz'  # Dataset ('easy.npz' or 'hard.npz')

################################################################################
# LOAD DATA
################################################################################
# Download graph classification data
if not os.path.exists(dataset_name):
    print('Downloading ' + dataset_name + ' from ' + data_url)
    req = requests.get(data_url + dataset_name)
    with open(dataset_name, 'wb') as out_file:
        out_file.write(req.content)

# Load data
loaded = np.load(dataset_name, allow_pickle=True)
X_train, A_train, y_train = loaded['tr_feat'], list(loaded['tr_adj']), loaded['tr_class']
X_test, A_test, y_test = loaded['te_feat'], list(loaded['te_adj']), loaded['te_class']
X_val, A_val, y_val = loaded['val_feat'], list(loaded['val_adj']), loaded['val_class']

# Preprocessing adjacency matrices for convolution
A_train = [normalized_adjacency(a) for a in A_train]
A_val = [normalized_adjacency(a) for a in A_val]
A_test = [normalized_adjacency(a) for a in A_test]

# Parameters
F = X_train[0].shape[-1]  # Dimension of node features
n_out = y_train[0].shape[-1]  # Dimension of the target
average_N = np.ceil(np.mean([a.shape[-1] for a in A_train]))  # Average number of nodes in dataset

################################################################################
# BUILD MODEL
################################################################################
X_in = Input(shape=(F, ), name='X_in', dtype=tf.float64)
A_in = Input(shape=(None,), sparse=True, dtype=tf.float64)
I_in = Input(shape=(), name='segment_ids_in', dtype=tf.int32)

X_1 = GraphConvSkip(n_channels,
                    activation=activ,
                    kernel_regularizer=l2(GNN_l2))([X_in, A_in])
X_1, A_1, I_1 = MinCutPool(k=int(average_N // 2),
                           activation=activ,
                           kernel_regularizer=l2(pool_l2))([X_1, A_in, I_in])

X_2 = GraphConvSkip(n_channels,
                    activation=activ,
                    kernel_regularizer=l2(GNN_l2))([X_1, A_1])
X_2, A_2, I_2 = MinCutPool(k=int(average_N // 4),
                           activation=activ,
                           kernel_regularizer=l2(pool_l2))([X_2, A_1, I_1])

X_3 = GraphConvSkip(n_channels,
                    activation=activ,
                    kernel_regularizer=l2(GNN_l2))([X_2, A_2])

# Output block
avgpool = GlobalAvgPool()([X_3, I_2])
output = Dense(n_out, activation='softmax')(avgpool)

# Build model
model = Model([X_in, A_in, I_in], output)
model.compile(optimizer='adam',  # Doesn't matter, won't be used
              loss='categorical_crossentropy')
model.summary()

# Training setup
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = model.loss_functions[0]
acc_fn = lambda x, y: K.mean(categorical_accuracy(x, y))


def train_loop(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, acc_fn(targets, predictions)

################################################################################
# FIT MODEL
################################################################################
current_batch = 0
model_loss = 0
model_acc = 0
best_val_loss = np.inf
best_weights = None
patience = es_patience
batches_in_epoch = np.ceil(y_train.shape[0] / batch_size)

print('Fitting model')
batches = batch_iterator([A_train, X_train, y_train], batch_size=batch_size, epochs=epochs)
for b in batches:
    X_, A_, I_ = Batch(b[0], b[1]).get('XAI')
    A_ = sp_matrix_to_sp_tensor(A_)
    y_ = b[2]
    outs = train_loop([X_, A_, I_], y_)

    model_loss += outs[0]
    model_acc += outs[1]
    current_batch += 1
    if current_batch % batches_in_epoch == 0:
        model_loss /= batches_in_epoch
        model_acc /= batches_in_epoch

        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(A_val, X_val, y_val, [loss_fn, acc_fn], batch_size=batch_size)
        print('Ep. {} - Loss: {:.2f} - Acc: {:.2f} - Val loss: {:.2f} - Val acc: {:.2f}'
              .format(current_batch // batches_in_epoch, model_loss, model_acc, val_loss, val_acc))

        # Check if loss improved for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = es_patience
            print('New best val_loss {:.3f}'.format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                print('Early stopping (best val_loss: {})'.format(best_val_loss))
                break
        model_loss = 0
        model_acc = 0

################################################################################
# EVALUATE MODEL
################################################################################
# Load best model
model.set_weights(best_weights)

# Test model
print('Testing model')
test_loss, test_acc = evaluate(A_test, X_test, y_test, [loss_fn, acc_fn], batch_size=batch_size)
print('Done.\n'
      'Test loss: {:.2f}\n'
      'Test acc: {:.2f}'
      .format(test_loss, test_acc))
