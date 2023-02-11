"""
This example implements the experiments on citation networks from the paper:

Simplifying Graph Convolutional Networks (https://arxiv.org/abs/1902.07153)
Felix Wu, Tianyi Zhang, Amauri Holanda de Souza Jr., Christopher Fifty, Tao Yu, Kilian Q. Weinberger

To implement it, we define a custom transform for the adjacency matrix. A
transform is simply a callable object that takes a Graph as input and returns
a Graph.
"""

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GCNConv
from spektral.transforms import LayerPreprocess


class SGCN:
    def __init__(self, K):
        self.K = K

    def __call__(self, graph):
        out = graph.a
        for _ in range(self.K - 1):
            out = out.dot(out)
        out.sort_indices()
        graph.a = out
        return graph


# Load data
K = 2  # Propagation steps for SGCN
dataset = Citation("cora", transforms=[LayerPreprocess(GCNConv), SGCN(K)])
mask_tr, mask_va, mask_te = dataset.mask_tr, dataset.mask_va, dataset.mask_te

# Parameters
l2_reg = 5e-6  # L2 regularization rate
learning_rate = 0.2  # Learning rate
epochs = 20000  # Number of training epochs
patience = 200  # Patience for early stopping
a_dtype = dataset[0].a.dtype  # Only needed for TF 2.1

N = dataset.n_nodes  # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = dataset.n_labels  # Number of classes

# Model definition
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True, dtype=a_dtype)

output = GCNConv(
    n_out, activation="softmax", kernel_regularizer=l2(l2_reg), use_bias=False
)([x_in, a_in])

# Build model
model = Model(inputs=[x_in, a_in], outputs=output)
optimizer = Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", weighted_metrics=["acc"]
)
model.summary()

# Train model
loader_tr = SingleLoader(dataset, sample_weights=mask_tr)
loader_va = SingleLoader(dataset, sample_weights=mask_va)
model.fit(
    loader_tr.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_data=loader_va.load(),
    validation_steps=loader_va.steps_per_epoch,
    epochs=epochs,
    callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
)

# Evaluate model
print("Evaluating model.")
loader_te = SingleLoader(dataset, sample_weights=mask_te)
eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))
