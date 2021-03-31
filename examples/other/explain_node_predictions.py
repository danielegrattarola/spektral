import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GCNConv
from spektral.models import GNNExplainer
from spektral.models.gcn import GCN
from spektral.transforms import AdjToSpTensor, LayerPreprocess
from spektral.utils import gcn_filter

############ Model setup ###################


learning_rate = 1e-2
seed = 0
epochs = 50
patience = 10
data = "cora"

tf.random.set_seed(seed=seed)  # make weight initialization reproducible

# Load data
dataset = Citation(
    data, normalize_x=True, transforms=[LayerPreprocess(GCNConv), AdjToSpTensor()]
)


# We convert the binary masks to sample weights so that we can compute the
# average loss over the nodes (following original implementation by
# Kipf & Welling)
def mask_to_weights(mask):
    return mask.astype(np.float32) / np.count_nonzero(mask)


weights_tr, weights_va, weights_te = (
    mask_to_weights(mask)
    for mask in (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
)

model = GCN(n_labels=dataset.n_labels, n_input_channels=dataset.n_node_features)
model.compile(
    optimizer=Adam(learning_rate),
    loss=CategoricalCrossentropy(reduction="sum"),
    weighted_metrics=["acc"],
)

# Train model
loader_tr = SingleLoader(dataset, sample_weights=weights_tr)
loader_va = SingleLoader(dataset, sample_weights=weights_va)

model.fit(
    loader_tr.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_data=loader_va.load(),
    validation_steps=loader_va.steps_per_epoch,
    epochs=epochs,
    callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
)


######### Explainer ################


# select the feature matrix and the laplacian matrix
x_exp, a_exp = dataset[0].x, dataset[0].a


# since it is used a laplacian matrix also a
# transformer should be passed
explainer = GNNExplainer(
    model,
    mode="node",
    num_conv_layers=2,
    x=x_exp,
    a=a_exp,
    adj_transf=gcn_filter,
    verbose=False,
    epochs=300,
)

# pick the node to explain
node_idx = 1000

# get the trained masks
adj_mask, feat_mask = explainer.explain_node(
    node_idx=node_idx,
    edge_size_reg=0.000001,
    edge_entropy_reg=0.5,
    laplacian_reg=1,
    feat_size_reg=0.0001,
    feat_entropy_reg=0.1,
)

# plot the result
G = explainer.plot_subgraph(adj_mask, feat_mask)

plt.show()
