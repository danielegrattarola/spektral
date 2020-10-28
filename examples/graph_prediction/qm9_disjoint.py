"""
This example shows how to perform regression of molecular properties with the
QM9 database, using a simple GNN in disjoint mode.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.datasets import qm9
from spektral.layers import EdgeConditionedConv, ops, GlobalSumPool
from spektral.data.utils import numpy_to_disjoint, batch_generator
from spektral.utils import label_to_one_hot

################################################################################
# PARAMETERS
################################################################################
learning_rate = 1e-3  # Learning rate
epochs = 10           # Number of training epochs
batch_size = 32       # Batch size

################################################################################
# LOAD DATA
################################################################################
A, X, E, y = qm9.load_data(return_type='numpy',
                           nf_keys='atomic_num',
                           ef_keys='type',
                           self_loops=False,
                           auto_pad=False,
                           amount=1000)  # Set to None to train on whole dataset
y = y[['cv']].values  # Heat capacity at 298.15K

# Preprocessing
X_uniq = np.unique([v for x in X for v in np.unique(x)])
E_uniq = np.unique([v for e in E for v in np.unique(e)])
X_uniq = X_uniq[X_uniq != 0]
E_uniq = E_uniq[E_uniq != 0]

X = [label_to_one_hot(x, labels=X_uniq) for x in X]
E = [label_to_one_hot(e, labels=E_uniq) for e in E]

# Parameters
F = X[0].shape[-1]   # Dimension of node features
S = E[0].shape[-1]   # Dimension of edge features
n_out = y.shape[-1]  # Dimension of the target

# Train/test split
A_train, A_test, \
X_train, X_test, \
E_train, E_test, \
y_train, y_test = train_test_split(A, X, E, y, test_size=0.1, random_state=0)

################################################################################
# BUILD MODEL
################################################################################
X_in = Input(shape=(F,), name='X_in')
A_in = Input(shape=(None,), sparse=True, name='A_in')
E_in = Input(shape=(S,), name='E_in')
I_in = Input(shape=(), name='segment_ids_in', dtype=tf.int32)

X_1 = EdgeConditionedConv(32, activation='relu')([X_in, A_in, E_in])
X_2 = EdgeConditionedConv(32, activation='relu')([X_1, A_in, E_in])
X_3 = GlobalSumPool()([X_2, I_in])
output = Dense(n_out)(X_3)

# Build model
model = Model(inputs=[X_in, A_in, E_in, I_in], outputs=output)
opt = Adam(lr=learning_rate)
loss_fn = MeanSquaredError()


@tf.function(
    input_signature=(tf.TensorSpec((None, F), dtype=tf.float64),
                     tf.SparseTensorSpec((None, None), dtype=tf.float64),
                     tf.TensorSpec((None, S), dtype=tf.float64),
                     tf.TensorSpec((None,), dtype=tf.int32),
                     tf.TensorSpec((None, n_out), dtype=tf.float64)),
    experimental_relax_shapes=True)
def train_step(X_, A_, E_, I_, y_):
    with tf.GradientTape() as tape:
        predictions = model([X_, A_, E_, I_], training=True)
        loss = loss_fn(y_, predictions)
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
batches_train = batch_generator([X_train, A_train, E_train, y_train],
                                batch_size=batch_size, epochs=epochs)
for b in batches_train:
    X_, A_, E_, I_ = numpy_to_disjoint(*b[:-1])
    A_ = ops.sp_matrix_to_sp_tensor(A_)
    y_ = b[-1]
    outs = train_step(X_, A_, E_, I_, y_)

    model_loss += outs.numpy()
    current_batch += 1
    if current_batch == batches_in_epoch:
        print('Loss: {}'.format(model_loss / batches_in_epoch))
        model_loss = 0
        current_batch = 0

################################################################################
# EVALUATE MODEL
################################################################################
print('Testing model')
model_loss = 0
batches_in_epoch = np.ceil(len(A_test) / batch_size)
batches_test = batch_generator([X_test, A_test, E_test, y_test], batch_size=batch_size)
for b in batches_test:
    X_, A_, E_, I_ = numpy_to_disjoint(*b[:-1])
    A_ = ops.sp_matrix_to_sp_tensor(A_)
    y_ = b[3]

    predictions = model([X_, A_, E_, I_], training=False)
    model_loss += loss_fn(y_, predictions)
model_loss /= batches_in_epoch
print('Done. Test loss: {}'.format(model_loss))
