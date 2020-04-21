"""
This script is a proof of concept to train GCN as fast as possible and with as
little lines of code as possible.
It uses a custom training function instead of the standard Keras fit(), and
can train GCN for 200 epochs in a few tenths of a second (0.32s on a GTX 1050).
In total, this script has 34 SLOC.
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.datasets import citation
from spektral.layers import GraphConv, ops
from spektral.utils import tic, toc

# Load data
A, X, y, train_mask, val_mask, test_mask = citation.load_data('cora')
fltr = GraphConv.preprocess(A).astype('f4')
fltr = ops.sp_matrix_to_sp_tensor(fltr)
X = X.toarray()

# Define model
X_in = Input(shape=(X.shape[1],))
fltr_in = Input((X.shape[0],), sparse=True)
X_1 = GraphConv(16, 'relu', True, kernel_regularizer=l2(5e-4))([X_in, fltr_in])
X_1 = Dropout(0.5)(X_1)
X_2 = GraphConv(y.shape[1], 'softmax', True)([X_1, fltr_in])

# Build model
model = Model(inputs=[X_in, fltr_in], outputs=X_2)
optimizer = Adam(lr=1e-2)
loss_fn = CategoricalCrossentropy()


# Training step
@tf.function
def train():
    with tf.GradientTape() as tape:
        predictions = model([X, fltr], training=True)
        loss = loss_fn(y[train_mask], predictions[train_mask])
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# Time the execution of 200 epochs of training
train()  # Warm up to ignore tracing times when timing
tic()
for epoch in range(1, 201):
    train()
toc('Spektral - GCN (200 epochs)')
