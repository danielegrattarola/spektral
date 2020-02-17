"""
This example shows how to perform regression of molecular properties with the
QM9 database, using a simple GNN in disjoint mode .
The main training loop is written in TensorFlow, because we need to avoid the
restriction imposed by Keras that the input and the output have the same first
dimension.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from spektral.datasets import qm9
from spektral.layers import GlobalAvgPool, EdgeConditionedConv
from spektral.utils import Batch, batch_iterator
from spektral.utils import label_to_one_hot

################################################################################
# PARAMETERS
################################################################################
learning_rate = 1e-3  # Learning rate
epochs = 25           # Number of training epochs
batch_size = 32       # Batch size

################################################################################
# LOAD DATA
################################################################################
A, X, E, y = qm9.load_data(return_type='numpy',
                           nf_keys='atomic_num',
                           ef_keys='type',
                           self_loops=True,
                           auto_pad=False,
                           amount=1000)  # Set to None to train on whole dataset
y = y[['cv']].values  # Heat capacity at 298.15K

# Preprocessing
uniq_X = np.unique([v for x in X for v in np.unique(x)])
X = [label_to_one_hot(x, uniq_X) for x in X]
uniq_E = np.unique([v for e in E for v in np.unique(e)])
uniq_E = uniq_E[uniq_E != 0]
E = [label_to_one_hot(e, uniq_E) for e in E]

# Parameters
F = X[0].shape[-1]    # Dimension of node features
S = E[0].shape[-1]    # Dimension of edge features
n_out = y.shape[-1]   # Dimension of the target

# Train/test split
A_train, A_test, \
X_train, X_test, \
E_train, E_test, \
y_train, y_test = train_test_split(A, X, E, y, test_size=0.1)

################################################################################
# BUILD MODEL
################################################################################
X_in = Input(shape=(F,), name='X_in')
A_in = Input(shape=(None,), name='A_in')
E_in = Input(shape=(None, S), name='E_in')
I_in = Input(shape=(), name='segment_ids_in', dtype=tf.int32)

X_1 = EdgeConditionedConv(32, activation='relu')([X_in, A_in, E_in])
X_2 = EdgeConditionedConv(32, activation='relu')([X_1, A_in, E_in])
X_3 = GlobalAvgPool()([X_2, I_in])
output = Dense(n_out)(X_3)

# Build model
model = Model(inputs=[X_in, A_in, E_in, I_in], outputs=output)
model.compile(optimizer='adam',  # Doesn't matter, won't be used
              loss='mse')
model.summary()

# Training setup
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = model.loss_functions[0]


@tf.function(experimental_relax_shapes=True)
def train_loop(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


################################################################################
# FIT MODEL
################################################################################
current_batch = 0
model_loss = 0
batches_in_epoch = np.ceil(len(A_train) / batch_size)

print('Fitting model')
batches_train = batch_iterator([A_train, X_train, E_train, y_train], batch_size=batch_size, epochs=epochs)
for b in batches_train:
    X_, A_, E_, I_ = Batch(b[0], b[1], b[2]).get('XAEI')
    A_ = A_.toarray()  # ECC wants dense inputs
    y_ = b[3]
    outs = train_loop([X_, A_, E_, I_], y_)

    model_loss += outs.numpy()
    current_batch += 1
    if current_batch == batches_in_epoch:
        print('Loss: {}'.format(model_loss / batches_in_epoch))
        model_loss = 0
        current_batch = 0

################################################################################
# EVALUATE MODEL
################################################################################
model_loss = 0
batches_in_epoch = np.ceil(len(A_test) / batch_size)

# Test model
print('Testing model')
batches_test = batch_iterator([A_test, X_test, E_test, y_test], batch_size=batch_size)
for b in batches_test:
    X_, A_, E_, I_ = Batch(b[0], b[1], b[2]).get('XAEI')
    A_ = A_.toarray()
    y_ = b[3]

    predictions = model([X_, A_, E_, I_], training=False)
    model_loss += loss_fn(y_, predictions)

print('Done.\n'
      'Test loss: {}'.format(model_loss / batches_in_epoch))
