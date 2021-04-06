import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader
from spektral.datasets import TUDataset
from spektral.models import GeneralGNN, GNNExplainer

# Config
learning_rate = 1e-2
batch_size = 32
epochs = 100

# Load data
data = TUDataset("PROTEINS")

# Train/test split
np.random.shuffle(data)
split = int(0.8 * len(data))
data_tr, data_te = data[:split], data[split:]

# Data loaders
loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_te = DisjointLoader(data_te, batch_size=batch_size)

# Create model
model = GeneralGNN(data.n_labels, activation="softmax")
optimizer = Adam(learning_rate)
loss_fn = CategoricalCrossentropy()


# Training function
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_on_batch(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    acc = tf.reduce_mean(categorical_accuracy(target, predictions))
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
        acc = tf.reduce_mean(categorical_accuracy(target, predictions))
        results.append((loss, acc, len(target)))  # Keep track of batch size
        if step == loader.steps_per_epoch:
            results = np.array(results)
            return np.average(results[:, :-1], 0, weights=results[:, -1])


# Train model
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
        print(
            "Epoch {} - Train loss: {:.3f} - Train acc: {:.3f} - "
            "Test loss: {:.3f} - Test acc: {:.3f}".format(
                epoch, *np.mean(results, 0), *results_te
            )
        )

# Set up explainer
(x, a, i), y = next(iter(loader_te))
last_idx = len(np.where(i == 0)[0])
x_exp = x[:last_idx]
a_exp = tf.sparse.slice(a, start=[0, 0], size=[last_idx, last_idx])
explainer = GNNExplainer(model, mode="graph", verbose=False, epochs=300)

# Explain prediction for one graph
adj_mask, feat_mask = explainer.explain_node(
    x=x_exp,
    a=a_exp,
    edge_size_reg=0.000001,
    edge_entropy_reg=0.5,
    laplacian_reg=1,
    feat_size_reg=0.0001,
    feat_entropy_reg=0.1,
)

# Plot the result
G = explainer.plot_subgraph(adj_mask, feat_mask)
plt.show()
