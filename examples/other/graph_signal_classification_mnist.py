import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.data import MixedLoader
from spektral.datasets.mnist import MNIST
from spektral.layers import GCNConv, GlobalSumPool
from spektral.utils.sparse import sp_matrix_to_sp_tensor

# Parameters
batch_size = 32  # Batch size
epochs = 1000  # Number of training epochs
patience = 10  # Patience for early stopping
l2_reg = 5e-4  # Regularization rate for l2

# Load data
data = MNIST()

# The adjacency matrix is stored as an attribute of the dataset.
# Create filter for GCN and convert to sparse tensor.
data.a = GCNConv.preprocess(data.a)
data.a = sp_matrix_to_sp_tensor(data.a)

# Train/valid/test split
data_tr, data_te = data[:-10000], data[-10000:]
np.random.shuffle(data_tr)
data_tr, data_va = data_tr[:-10000], data_tr[-10000:]

# We use a MixedLoader since the dataset is in mixed mode
loader_tr = MixedLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_va = MixedLoader(data_va, batch_size=batch_size)
loader_te = MixedLoader(data_te, batch_size=batch_size)


# Build model
class Net(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GCNConv(32, activation="elu", kernel_regularizer=l2(l2_reg))
        self.conv2 = GCNConv(32, activation="elu", kernel_regularizer=l2(l2_reg))
        self.flatten = GlobalSumPool()
        self.fc1 = Dense(512, activation="relu")
        self.fc2 = Dense(10, activation="softmax")  # MNIST has 10 classes

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)

        return output


# Create model
model = Net()
optimizer = Adam()
loss_fn = SparseCategoricalCrossentropy()


# Training function
@tf.function
def train_on_batch(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
        acc = tf.reduce_mean(sparse_categorical_accuracy(target, predictions))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, acc


# Evaluation function
def evaluate(loader):
    step = 0
    results = []
    for batch in loader:
        step += 1
        inputs, target = batch
        predictions = model(inputs, training=False)
        loss = loss_fn(target, predictions)
        acc = tf.reduce_mean(sparse_categorical_accuracy(target, predictions))
        results.append((loss, acc, len(target)))  # Keep track of batch size
        if step == loader.steps_per_epoch:
            results = np.array(results)
            return np.average(results[:, :-1], 0, weights=results[:, -1])


# Setup training
best_val_loss = 99999
current_patience = patience
step = 0

# Training loop
results_tr = []
for batch in loader_tr:
    step += 1

    # Training step
    inputs, target = batch
    loss, acc = train_on_batch(inputs, target)
    results_tr.append((loss, acc, len(target)))

    if step == loader_tr.steps_per_epoch:
        results_va = evaluate(loader_va)
        if results_va[0] < best_val_loss:
            best_val_loss = results_va[0]
            current_patience = patience
            results_te = evaluate(loader_te)
        else:
            current_patience -= 1
            if current_patience == 0:
                print("Early stopping")
                break

        # Print results
        results_tr = np.array(results_tr)
        results_tr = np.average(results_tr[:, :-1], 0, weights=results_tr[:, -1])
        print(
            "Train loss: {:.4f}, acc: {:.4f} | "
            "Valid loss: {:.4f}, acc: {:.4f} | "
            "Test loss: {:.4f}, acc: {:.4f}".format(
                *results_tr, *results_va, *results_te
            )
        )

        # Reset epoch
        results_tr = []
        step = 0
