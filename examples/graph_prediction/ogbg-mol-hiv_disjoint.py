"""
This example shows how to perform molecule classification with the
[Open Graph Benchmark](https://ogb.stanford.edu) `mol-hiv` dataset, using a
simple ECC-based GNN in disjoint mode. The model does not perform really well
but should give you a starting point if you want to implement a more
sophisticated one.
"""

import numpy as np
import tensorflow as tf
from ogb.graphproppred import GraphPropPredDataset, Evaluator
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.datasets import ogb
from spektral.layers import EdgeConditionedConv, ops, GlobalSumPool
from spektral.data.utils import numpy_to_disjoint, batch_generator

################################################################################
# PARAMETERS
################################################################################
learning_rate = 1e-3  # Learning rate
epochs = 10           # Number of training epochs
batch_size = 32       # Batch size

################################################################################
# LOAD DATA
################################################################################
dataset_name = 'ogbg-molhiv'
dataset = GraphPropPredDataset(name=dataset_name)
n_out = dataset.num_tasks

idx = dataset.get_idx_split()
tr_idx, va_idx, te_idx = idx["train"], idx["valid"], idx["test"]

X_tr, A_tr, E_tr, y_tr = ogb.dataset_to_numpy(dataset, tr_idx, dtype='f8')
X_va, A_va, E_va, y_va = ogb.dataset_to_numpy(dataset, va_idx, dtype='f8')
X_te, A_te, E_te, y_te = ogb.dataset_to_numpy(dataset, te_idx, dtype='f8')

F = X_tr[0].shape[-1]
S = E_tr[0].shape[-1]

################################################################################
# BUILD MODEL
################################################################################
X_in = Input(shape=(F,))
A_in = Input(shape=(None,), sparse=True)
E_in = Input(shape=(S,))
I_in = Input(shape=(), dtype=tf.int64)

X_1 = EdgeConditionedConv(32, activation='relu')([X_in, A_in, E_in])
X_2 = EdgeConditionedConv(32, activation='relu')([X_1, A_in, E_in])
X_3 = GlobalSumPool()([X_2, I_in])
output = Dense(n_out, activation='sigmoid')(X_3)

# Build model
model = Model(inputs=[X_in, A_in, E_in, I_in], outputs=output)
opt = Adam(lr=learning_rate)
loss_fn = BinaryCrossentropy()


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
batches_in_epoch = np.ceil(len(A_tr) / batch_size)

print('Fitting model')
batches_train = batch_generator([X_tr, A_tr, E_tr, y_tr],
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
evaluator = Evaluator(name=dataset_name)
y_pred = []
batches_test = batch_generator([X_te, A_te, E_te], batch_size=batch_size)
for b in batches_test:
    X_, A_, E_, I_ = numpy_to_disjoint(*b)
    A_ = ops.sp_matrix_to_sp_tensor(A_)
    p = model([X_, A_, E_, I_], training=False)
    y_pred.append(p.numpy())

y_pred = np.vstack(y_pred)
model_loss = loss_fn(y_te, y_pred)
ogb_score = evaluator.eval({'y_true': y_te, 'y_pred': y_pred})

print('Done. Test loss: {:.4f}. ROC-AUC: {:.2f}'
      .format(model_loss, ogb_score['rocauc']))
