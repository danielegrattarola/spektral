"""
This example shows how to perform graph classification with a synthetic
benchmark dataset created by F. M. Bianchi (https://github.com/FilippoMB/Benchmark_dataset_for_graph_classification),
using a GNN with convolutional and pooling blocks in disjoint mode.
Note that the main training loop is written in TensorFlow, because we need to
avoid the restriction imposed by Keras that the input and the output have the
same first dimension. This is the most efficient way of training a GNN in
disjoint mode.
"""

import os

import keras.backend as K
import numpy as np
import requests
import tensorflow as tf
from keras.layers import Input, Dense
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.regularizers import l2

from spektral.layers import GraphConvSkip, GlobalAvgPool
from spektral.layers.ops import sp_matrix_to_sp_tensor_value
from spektral.layers.pooling import MinCutPool
from spektral.utils import batch_iterator
from spektral.utils.convolution import normalized_adjacency
from spektral.utils.data import Batch

np.random.seed(0)
SW_KEY = 'dense_1_sample_weights:0'  # Keras automatically creates a placeholder for sample weights, which must be fed


def evaluate(A_list, X_list, y_list, ops, batch_size):
    batches = batch_iterator([A_list, X_list, y_list], batch_size=batch_size)
    output = []
    for b in batches:
        X, A, I = Batch(b[0], b[1]).get('XAI')
        y = b[2]
        feed_dict = {X_in: X,
                     A_in: sp_matrix_to_sp_tensor_value(A),
                     I_in: I,
                     target: y,
                     SW_KEY: np.ones((1,))}

        outs = sess.run(ops, feed_dict=feed_dict)
        output.append(outs)
    return np.mean(output, 0)


################################################################################
# PARAMETERS
################################################################################
n_channels = 32            # Channels per layer
activ = 'elu'              # Activation in GNN and maxcut / mincut
mincut_H = 16              # Dimension of hidden state in mincut
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
X_in = Input(tensor=tf.placeholder(tf.float32, shape=(None, F), name='X_in'))
A_in = Input(tensor=tf.sparse_placeholder(tf.float32, shape=(None, None)), sparse=True)
I_in = Input(tensor=tf.placeholder(tf.int32, shape=(None,), name='segment_ids_in'))
target = Input(tensor=tf.placeholder(tf.float32, shape=(None, n_out), name='target'))

# Block 1
gc1 = GraphConvSkip(n_channels,
                    activation=activ,
                    kernel_regularizer=l2(GNN_l2))([X_in, A_in])
X_1, A_1, I_1, M_1 = MinCutPool(k=int(average_N // 2),
                                h=mincut_H,
                                activation=activ,
                                kernel_regularizer=l2(pool_l2))([gc1, A_in, I_in])

# Block 2
gc2 = GraphConvSkip(n_channels,
                    activation=activ,
                    kernel_regularizer=l2(GNN_l2))([X_1, A_1])
X_2, A_2, I_2, M_2 = MinCutPool(k=int(average_N // 4),
                                h=mincut_H,
                                activation=activ,
                                kernel_regularizer=l2(pool_l2))([gc2, A_1, I_1])

# Block 3
X_3 = GraphConvSkip(n_channels,
                    activation=activ,
                    kernel_regularizer=l2(GNN_l2))([X_2, A_2])

# Output block
avgpool = GlobalAvgPool()([X_3, I_2])
output = Dense(n_out, activation='softmax')(avgpool)

# Build model
model = Model([X_in, A_in, I_in], output)
model.compile(optimizer='adam',  # Doesn't matter, won't be used
              loss='categorical_crossentropy',
              target_tensors=[target])
model.summary()

# Training setup
sess = K.get_session()
loss = model.total_loss
acc = K.mean(categorical_accuracy(target, model.output))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_step = opt.minimize(loss)
init_op = tf.global_variables_initializer()
sess.run(init_op)

################################################################################
# FIT MODEL
################################################################################
# Run training loop
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
    y_ = b[2]
    tr_feed_dict = {X_in: X_,
                    A_in: sp_matrix_to_sp_tensor_value(A_),
                    I_in: I_,
                    target: y_,
                    SW_KEY: np.ones((1,))}
    outs = sess.run([train_step, loss, acc], feed_dict=tr_feed_dict)

    model_loss += outs[1]
    model_acc += outs[2]
    current_batch += 1
    if current_batch % batches_in_epoch == 0:
        model_loss /= batches_in_epoch
        model_acc /= batches_in_epoch

        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(A_val, X_val, y_val, [loss, acc], batch_size=batch_size)
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
test_loss, test_acc = evaluate(A_test, X_test, y_test, [loss, acc], batch_size=batch_size)
print('Done.\n'
      'Test loss: {:.2f}\n'
      'Test acc: {:.2f}'
      .format(test_loss, test_acc))
