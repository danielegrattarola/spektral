import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.datasets import mnist
from spektral.layers import GraphConv
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.data.utils import batch_generator

# Parameters
learning_rate = 1e-3  # Learning rate for Adam
batch_size = 32       # Batch size
epochs = 1000         # Number of training epochs
patience = 10         # Patience for early stopping
l2_reg = 5e-4         # Regularization rate for l2

# Load data
x_tr, y_tr, x_va, y_va, x_te, y_te, A = mnist.load_data()
x_tr, x_va, x_te = x_tr[..., None], x_va[..., None], x_te[..., None]
N = x_tr.shape[-2]    # Number of nodes in the graphs
F = x_tr.shape[-1]    # Node features dimensionality
n_out = 10            # Dimension of the target

# Create filter for GCN and convert to sparse tensor
fltr = GraphConv.preprocess(A)
fltr = sp_matrix_to_sp_tensor(fltr)


# Build model
class Net(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GraphConv(32, activation='elu', kernel_regularizer=l2(l2_reg))
        self.conv2 = GraphConv(32, activation='elu', kernel_regularizer=l2(l2_reg))
        self.flatten = Flatten()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(n_out, activation='softmax')

    def call(self, inputs):
        x, fltr = inputs
        x = self.conv1([x, fltr])
        x = self.conv2([x, fltr])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)

        return output


model = Net()
optimizer = Adam(lr=learning_rate)
loss_fn = SparseCategoricalCrossentropy()
acc_fn = SparseCategoricalAccuracy()


# Training step
@tf.function
def train(x, y):
    with tf.GradientTape() as tape:
        predictions = model([x, fltr], training=True)
        loss = loss_fn(y, predictions)
        loss += sum(model.losses)
    acc = acc_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, acc


# Evaluation step
@tf.function
def evaluate(x, y):
    predictions = model([x, fltr], training=False)
    loss = loss_fn(y, predictions)
    loss += sum(model.losses)
    acc = acc_fn(y, predictions)

    return loss, acc


# Setup training
best_val_loss = 99999
current_patience = patience
curent_batch = 0
batches_in_epoch = int(np.ceil(x_tr.shape[0] / batch_size))
batches_tr = batch_generator([x_tr, y_tr], batch_size=batch_size, epochs=epochs)

# Training loop
results_tr = []
results_te = np.zeros(2)
for batch in batches_tr:
    curent_batch += 1

    # Training step
    l, a = train(*batch)
    results_tr.append((l, a))

    if curent_batch == batches_in_epoch:
        batches_va = batch_generator([x_va, y_va], batch_size=batch_size, epochs=1)
        results_va = [evaluate(*batch) for batch in batches_va]
        results_va = np.array(results_va)
        loss_va, acc_va = results_va.mean(0)
        if loss_va < best_val_loss:
            best_val_loss = loss_va
            current_patience = patience
            # Test
            batches_te = batch_generator([x_te, y_te], batch_size=batch_size, epochs=1)
            results_te = [evaluate(*batch) for batch in batches_te]
            results_te = np.array(results_te)
        else:
            current_patience -= 1
            if current_patience == 0:
                print('Early stopping')
                break

        # Print results
        results_tr = np.array(results_tr)
        print('Train loss: {:.4f}, acc: {:.4f} | '
              'Valid loss: {:.4f}, acc: {:.4f} | '
              'Test loss: {:.4f}, acc: {:.4f}'
              .format(*results_tr.mean(0),
                      *results_va.mean(0),
                      *results_te.mean(0)))

        # Reset epoch
        results_tr = []
        curent_batch = 0
