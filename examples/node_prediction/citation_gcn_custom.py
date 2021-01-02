"""
This script is a proof of concept to train GCN as fast as possible and with as
little lines of code as possible.
It uses a custom training function instead of the standard Keras fit(), and
can train GCN for 200 epochs in a few tenths of a second (~0.20 on a GTX 1050).
"""
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.datasets.citation import Cora
from spektral.layers import GCNConv
from spektral.transforms import AdjToSpTensor, LayerPreprocess
from spektral.utils import tic, toc

tf.random.set_seed(seed=0)  # make weight initialization reproducible

# Load data
dataset = Cora(normalize_x=True, transforms=[LayerPreprocess(GCNConv), AdjToSpTensor()])
graph = dataset[0]
x, a, y = graph.x, graph.a, graph.y
mask_tr, mask_va, mask_te = dataset.mask_tr, dataset.mask_va, dataset.mask_te

# Define model
x_in = Input(shape=(dataset.n_node_features,))
a_in = Input((dataset.n_nodes,), sparse=True)
x_1 = Dropout(0.5)(x_in)
x_1 = GCNConv(16, "relu", False, kernel_regularizer=l2(2.5e-4))([x_1, a_in])
x_1 = Dropout(0.5)(x_1)
x_2 = GCNConv(y.shape[1], "softmax", False)([x_1, a_in])

# Build model
model = Model(inputs=[x_in, a_in], outputs=x_2)
optimizer = Adam(lr=1e-2)
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
