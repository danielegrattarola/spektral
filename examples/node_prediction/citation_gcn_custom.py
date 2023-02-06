"""
This script is a proof of concept to train GCN as fast as possible and with as
little lines of code as possible.
It uses a custom training function instead of the standard Keras fit(), and
can train GCN for 200 epochs in a few tenths of a second (~0.20 on a GTX 1050).
"""
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from spektral.datasets.citation import Cora
from spektral.layers import GCNConv
from spektral.models.gcn import GCN
from spektral.transforms import AdjToSpTensor, LayerPreprocess
from spektral.utils import tic, toc

tf.random.set_seed(seed=0)  # make weight initialization reproducible

# Load data
dataset = Cora(normalize_x=True, transforms=[LayerPreprocess(GCNConv), AdjToSpTensor()])
graph = dataset[0]
x, a, y = graph.x, graph.a, graph.y
mask_tr, mask_va, mask_te = dataset.mask_tr, dataset.mask_va, dataset.mask_te

model = GCN(n_labels=dataset.n_labels)
optimizer = Adam(learning_rate=1e-2)
loss_fn = CategoricalCrossentropy()


# Training step
@tf.function
def train():
    with tf.GradientTape() as tape:
        predictions = model([x, a], training=True)
        loss = loss_fn(y[mask_tr], predictions[mask_tr])
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# Time the execution of 200 epochs of training
train()  # Warm up to ignore tracing times when timing
tic()
for epoch in range(1, 201):
    loss = train()
toc("Spektral - GCN (200 epochs)")
print(f"Final loss = {loss}")
