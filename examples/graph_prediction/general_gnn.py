"""
This example implements the model from the paper

    > [Design Space for Graph Neural Networks](https://arxiv.org/abs/2011.08843)<br>
    > Jiaxuan You, Rex Ying, Jure Leskovec

using the PROTEINS dataset.

The configuration at the top of the file is the best one identified in the
paper, and should work well for many different datasets without changes.

Note: the results reported in the paper are averaged over 3 random repetitions
with an 80/20 split.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader
from spektral.datasets import TUDataset
from spektral.models import GeneralGNN

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Best config
batch_size = 32
learning_rate = 0.01
epochs = 400

# Read data
data = TUDataset('PROTEINS')

# Train/test split
np.random.shuffle(data)
split = int(0.8 * len(data))
data_tr, data_te = data[:split], data[split:]

# Data loader
loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_te = DisjointLoader(data_te, batch_size=batch_size)

# Create model
model = GeneralGNN(data.n_labels, activation='softmax')
optimizer = Adam(learning_rate)
loss_fn = CategoricalCrossentropy()
acc_fn = CategoricalAccuracy()


# Training function
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_on_batch(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
        acc = acc_fn(target, predictions)

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
        acc = acc_fn(target, predictions)
        results.append((loss, acc, len(target)))  # Keep track of batch size
        if step == loader.steps_per_epoch:
            results = np.array(results)
            return np.average(results[:, :-1], 0, weights=results[:, -1])


# Training loop
epoch = step = 0
results = []
for batch in loader_tr:
    step += 1
    loss, acc = train_on_batch(*batch)
    results.append((loss, acc))
    if step == loader_tr.steps_per_epoch:
        step = 0
        epoch += 1
        results_te = evaluate(loader_te)
        print('Epoch {} - Train loss: {:.3f} - Train acc: {:.3f} - '
              'Test loss: {:.3f} - Test acc: {:.3f}'
              .format(epoch, *np.mean(results, 0), *results_te))

results_te = evaluate(loader_te)
print('Final results - Loss: {:.3f} - Acc: {:.3f}'.format(*results_te))
