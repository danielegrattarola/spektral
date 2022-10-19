"""
This example implements the experiments for node clustering on citation networks
from the paper:

Mincut pooling in Graph Neural Networks (https://arxiv.org/abs/1907.00481)
Filippo Maria Bianchi, Daniele Grattarola, Cesare Alippi
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics.cluster import (
    completeness_score,
    homogeneity_score,
    v_measure_score,
)
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tqdm import tqdm

from spektral.datasets.citation import Cora
from spektral.layers.convolutional import GCSConv
from spektral.layers.pooling import MinCutPool
from spektral.utils.convolution import normalized_adjacency
from spektral.utils.sparse import sp_matrix_to_sp_tensor


@tf.function
def train_step(inputs):
    with tf.GradientTape() as tape:
        _, S_pool = model(inputs, training=True)
        loss = sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return model.losses[0], model.losses[1], S_pool


np.random.seed(1)
epochs = 5000  # Training iterations
lr = 5e-4  # Learning rate

################################################################################
# LOAD DATASET
################################################################################
dataset = Cora()
adj, x, y = dataset[0].a, dataset[0].x, dataset[0].y
a_norm = normalized_adjacency(adj)
a_norm = sp_matrix_to_sp_tensor(a_norm)
F = dataset.n_node_features
y = np.argmax(y, axis=-1)
n_clusters = y.max() + 1

################################################################################
# MODEL
################################################################################
x_in = Input(shape=(F,), name="X_in")
a_in = Input(shape=(None,), name="A_in", sparse=True)

x_1 = GCSConv(16, activation="elu")([x_in, a_in])
x_1, a_1, s_1 = MinCutPool(n_clusters, return_selection=True)([x_1, a_in])

model = Model([x_in, a_in], [x_1, s_1])

################################################################################
# TRAINING
################################################################################
# Setup
inputs = [x, a_norm]
opt = tf.keras.optimizers.Adam(learning_rate=lr)

# Fit model
loss_history = []
nmi_history = []
for _ in tqdm(range(epochs)):
    outs = train_step(inputs)
    outs = [o.numpy() for o in outs]
    loss_history.append((outs[0], outs[1], (outs[0] + outs[1])))
    s_out = np.argmax(outs[2], axis=-1)
    nmi_history.append(v_measure_score(y, s_out))
loss_history = np.array(loss_history)

################################################################################
# RESULTS
################################################################################
_, s_out = model(inputs, training=False)
s_out = np.argmax(s_out, axis=-1)
hom = homogeneity_score(y, s_out)
com = completeness_score(y, s_out)
nmi = v_measure_score(y, s_out)
print("Homogeneity: {:.3f}; Completeness: {:.3f}; NMI: {:.3f}".format(hom, com, nmi))

# Plots
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.plot(loss_history[:, 0], label="Ortho. loss")
plt.plot(loss_history[:, 1], label="MinCUT loss")
plt.plot(loss_history[:, 2], label="Total loss")
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Iteration")

plt.subplot(122)
plt.plot(nmi_history, label="NMI")
plt.legend()
plt.ylabel("NMI")
plt.xlabel("Iteration")

plt.show()
