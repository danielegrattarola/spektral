"""
This example shows how to perform graph classification with a simple Graph
Isomorphism Network.
This is an example of TensorFlow 2's imperative style for model declaration.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.datasets import tud
from spektral.layers import GINConv, GlobalAvgPool, ops
from spektral.data.utils import numpy_to_disjoint, batch_generator

################################################################################
# PARAMETERS
################################################################################
learning_rate = 1e-3  # Learning rate
channels = 128        # Hidden units
layers = 3            # GIN layers
epochs = 10           # Number of training epochs
batch_size = 32       # Batch size

################################################################################
# LOAD DATA
################################################################################
a, x, y = tud.load_data('PROTEINS', clean=True)

# Parameters
F = x[0].shape[-1]   # Dimension of node features
n_out = y.shape[-1]  # Dimension of the target

# Train/test split
a_train, a_test, \
x_train, x_test, \
y_train, y_test = train_test_split(a, x, y, test_size=0.1, random_state=0)


################################################################################
# BUILD MODEL
################################################################################
class GIN0(Model):
    def __init__(self, channels, n_layers):
        super().__init__()
        self.conv1 = GINConv(channels, epsilon=0, mlp_hidden=[channels, channels])
        self.convs = []
        for i in range(1, n_layers):
            self.convs.append(
                GINConv(channels, epsilon=0, mlp_hidden=[channels, channels]))
        self.pool = GlobalAvgPool()
        self.dense1 = Dense(channels, activation='relu')
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(n_out, activation='softmax')

    def call(self, inputs, **kwargs):
        x, a, i = inputs
        x = self.conv1([x, a])
        for conv in self.convs:
            x = conv([x, a])
        x = self.pool([x, i])
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)


# Build model
model = GIN0(channels, layers)
opt = Adam(lr=learning_rate)
loss_fn = CategoricalCrossentropy()
acc_fn = CategoricalAccuracy()


@tf.function(
    input_signature=(tf.TensorSpec((None, F), dtype=tf.float64),
                     tf.SparseTensorSpec((None, None), dtype=tf.int64),
                     tf.TensorSpec((None,), dtype=tf.int32),
                     tf.TensorSpec((None, n_out), dtype=tf.float64)),
    experimental_relax_shapes=True)
def train_step(x_, a_, i_, y_):
    with tf.GradientTape() as tape:
        predictions = model([x_, a_, i_], training=True)
        loss = loss_fn(y_, predictions)
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    acc = acc_fn(y_, predictions)
    return loss, acc


################################################################################
# FIT MODEL
################################################################################
current_batch = 0
model_lss = model_acc = 0
batches_in_epoch = np.ceil(len(a_train) / batch_size)

print('Fitting model')
batches_train = batch_generator([x_train, a_train, y_train],
                                batch_size=batch_size, epochs=epochs)
for b in batches_train:
    x_, a_, i_ = numpy_to_disjoint(*b[:-1])
    a_ = ops.sp_matrix_to_sp_tensor(a_)
    y_ = b[-1]
    lss, acc = train_step(x_, a_, i_, y_)

    model_lss += lss.numpy()
    model_acc += acc.numpy()
    current_batch += 1
    if current_batch == batches_in_epoch:
        model_lss /= batches_in_epoch
        model_acc /= batches_in_epoch
        print('Loss: {}. Acc: {}'.format(model_lss, model_acc))
        model_lss = model_acc = 0
        current_batch = 0

################################################################################
# EVALUATE MODEL
################################################################################
print('Testing model')
model_lss = model_acc = 0
batches_in_epoch = np.ceil(len(a_test) / batch_size)
batches_test = batch_generator([x_test, a_test, y_test], batch_size=batch_size)
for b in batches_test:
    x_, a_, i_ = numpy_to_disjoint(*b[:-1])
    a_ = ops.sp_matrix_to_sp_tensor(a_)
    y_ = b[-1]
    predictions = model([x_, a_, i_], training=False)
    model_lss += loss_fn(y_, predictions)
    model_acc += acc_fn(y_, predictions)
model_lss /= batches_in_epoch
model_acc /= batches_in_epoch
print('Done. Test loss: {}. Test acc: {}'.format(model_lss, model_acc))
