"""
This example shows how to perform graph classification with a synthetic
benchmark dataset created by F. M. Bianchi (https://github.com/FilippoMB/Benchmark_dataset_for_graph_classification),
using a GNN with convolutional and pooling blocks in disjoint mode.
This is a more advanced example that also shows how to do validation and early
stopping. For a beginner-level example, see qm9_disjoint.py.
"""

import os

import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.layers import GraphConvSkip, GlobalAvgPool
from spektral.layers import ops
from spektral.layers.pooling import TopKPool
from spektral.utils.convolution import normalized_adjacency
from spektral.utils.data import batch_iterator, numpy_to_disjoint


def evaluate(A_list, X_list, y_list, ops_list, batch_size):
    batches = batch_iterator([X_list, A_list, y_list], batch_size=batch_size)
    output = []
    for b in batches:
        X, A, I = numpy_to_disjoint(*b[:-1])
        A = ops.sp_matrix_to_sp_tensor(A)
        y = b[-1]
        pred = model([X, A, I], training=False)
        outs = [o(y, pred) for o in ops_list]
        output.append(outs)
    return np.mean(output, 0)


################################################################################
# PARAMETERS
################################################################################
learning_rate = 1e-3       # Learning rate
epochs = 500               # Number of training epochs
es_patience = 50           # Patience for early stopping
batch_size = 16            # Batch size
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

# Preprocessing
A_train = [normalized_adjacency(a) for a in A_train]
A_val = [normalized_adjacency(a) for a in A_val]
A_test = [normalized_adjacency(a) for a in A_test]

# Parameters
F = X_train[0].shape[-1]  # Dimension of node features
n_out = y_train[0].shape[-1]  # Dimension of the target

################################################################################
# BUILD MODEL
################################################################################
X_in = Input(shape=(F, ), name='X_in')
A_in = Input(shape=(None,), sparse=True)
I_in = Input(shape=(), name='segment_ids_in', dtype=tf.int32)

X_1 = GraphConvSkip(32, activation='relu')([X_in, A_in])
X_1, A_1, I_1 = TopKPool(ratio=0.5)([X_1, A_in, I_in])
X_2 = GraphConvSkip(32, activation='relu')([X_1, A_1])
X_2, A_2, I_2 = TopKPool(ratio=0.5)([X_2, A_1, I_1])
X_3 = GraphConvSkip(32, activation='relu')([X_2, A_2])
X_3 = GlobalAvgPool()([X_3, I_2])
output = Dense(n_out, activation='softmax')(X_3)

# Build model
model = Model(inputs=[X_in, A_in, I_in], outputs=output)
opt = Adam(lr=learning_rate)
loss_fn = CategoricalCrossentropy()
acc_fn = CategoricalAccuracy()


@tf.function(
    input_signature=(tf.TensorSpec((None, F), dtype=tf.float64),
                     tf.SparseTensorSpec((None, None), dtype=tf.float32),
                     tf.TensorSpec((None,), dtype=tf.int32),
                     tf.TensorSpec((None, n_out), dtype=tf.float64)),
    experimental_relax_shapes=True)
def train_step(X_, A_, I_, y_):
    with tf.GradientTape() as tape:
        predictions = model([X_, A_, I_], training=True)
        loss = loss_fn(y_, predictions)
        loss += sum(model.losses)
        acc = acc_fn(y_, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, acc


################################################################################
# FIT MODEL
################################################################################
current_batch = 0
epoch = 0
model_loss = 0
model_acc = 0
best_val_loss = np.inf
best_weights = None
patience = es_patience
batches_in_epoch = np.ceil(y_train.shape[0] / batch_size)

print('Fitting model')
batches = batch_iterator([X_train, A_train, y_train],
                         batch_size=batch_size, epochs=epochs)
for b in batches:
    current_batch += 1

    X_, A_, I_ = numpy_to_disjoint(*b[:-1])
    A_ = ops.sp_matrix_to_sp_tensor(A_)
    y_ = b[-1]
    outs = train_step(X_, A_, I_, y_)

    model_loss += outs[0]
    model_acc += outs[1]
    if current_batch == batches_in_epoch:
        epoch += 1
        model_loss /= batches_in_epoch
        model_acc /= batches_in_epoch

        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(A_val, X_val, y_val, [loss_fn, acc_fn], batch_size=batch_size)
        print('Ep. {} - Loss: {:.2f} - Acc: {:.2f} - Val loss: {:.2f} - Val acc: {:.2f}'
              .format(epoch, model_loss, model_acc, val_loss, val_acc))

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
        current_batch = 0

################################################################################
# EVALUATE MODEL
################################################################################
print('Testing model')
model.set_weights(best_weights)  # Load best model
test_loss, test_acc = evaluate(A_test, X_test, y_test, [loss_fn, acc_fn], batch_size=batch_size)
print('Done. Test loss: {:.4f}. Test acc: {:.2f}'.format(test_loss, test_acc))
