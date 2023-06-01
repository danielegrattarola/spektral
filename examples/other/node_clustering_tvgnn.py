"""
This example implements the node clustering experiment on citation networks
from the paper:

Total Variation Graph Neural Networks (https://arxiv.org/abs/2211.06218)
Jonas Berg Hansen and Filippo Maria Bianchi
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics.cluster import (
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
)
from tensorflow.keras import Model
from tqdm import tqdm

from spektral.datasets import DBLP
from spektral.datasets.citation import Citation
from spektral.layers import AsymCheegerCutPool, GTVConv
from spektral.utils.sparse import sp_matrix_to_sp_tensor

tf.random.set_seed(1)

################################
# CONFIG/HYPERPARAMETERS
################################
dataset_id = "cora"
mp_channels = 512
mp_layers = 2
mp_activation = "elu"
delta_coeff = 0.311
epsilon = 1e-3
mlp_hidden_channels = 256
mlp_hidden_layers = 1
mlp_activation = "relu"
totvar_coeff = 0.785
balance_coeff = 0.514
learning_rate = 1e-3
epochs = 500

################################
# LOAD DATASET
################################
if dataset_id in ["cora", "citeseer", "pubmed"]:
    dataset = Citation(dataset_id, normalize_x=True)
elif dataset_id == "dblp":
    dataset = DBLP(normalize_x=True)
X = dataset.graphs[0].x
A = dataset.graphs[0].a
Y = dataset.graphs[0].y
y = np.argmax(Y, axis=-1)
n_clust = Y.shape[-1]


################################
# MODEL
################################
class ClusteringModel(Model):
    """
    Defines the general model structure
    """

    def __init__(self, aggr, pool):
        super().__init__()

        self.mp = aggr
        self.pool = pool

    def call(self, inputs):
        x, a = inputs

        out = x
        for _mp in self.mp:
            out = _mp([out, a])

        _, _, s_pool = self.pool([out, a])

        return s_pool


# Define the message-passing layers
MP_layers = [
    GTVConv(
        mp_channels, delta_coeff=delta_coeff, epsilon=1e-3, activation=mp_activation
    )
    for _ in range(mp_layers)
]

# Define the pooling layer
pool_layer = AsymCheegerCutPool(
    n_clust,
    mlp_hidden=[mlp_hidden_channels for _ in range(mlp_hidden_layers)],
    mlp_activation=mlp_activation,
    totvar_coeff=totvar_coeff,
    balance_coeff=balance_coeff,
    return_selection=True,
)

# Instantiate model and optimizer
model = ClusteringModel(aggr=MP_layers, pool=pool_layer)
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)


################################
# TRAINING
################################
@tf.function(input_signature=None)
def train_step(model, inputs):
    with tf.GradientTape() as tape:
        _ = model(inputs, training=True)
        loss = sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return model.losses


A = sp_matrix_to_sp_tensor(A)
inputs = [X, A]
loss_history = []

# Training loop
for _ in tqdm(range(epochs)):
    outs = train_step(model, inputs)

################################
# INFERENCE/RESULTS
################################
S_ = model(inputs, training=False)
s_out = np.argmax(S_, axis=-1)
nmi = normalized_mutual_info_score(y, s_out)
hom = homogeneity_score(y, s_out)
com = completeness_score(y, s_out)
print("Homogeneity: {:.3f}; Completeness: {:.3f}; NMI: {:.3f}".format(hom, com, nmi))
