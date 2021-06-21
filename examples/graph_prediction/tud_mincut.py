import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import BatchLoader
from spektral.datasets import TUDataset
from spektral.layers import GCNConv, GlobalSumPool, GraphMasking, MinCutPool

################################################################################
# Config
################################################################################
learning_rate = 1e-3  # Learning rate
epochs = 10  # Number of training epochs
batch_size = 32  # Batch size

################################################################################
# Load data
################################################################################
dataset = TUDataset("PROTEINS", clean=True)

# Parameters
N = max(g.n_nodes for g in dataset)
F = dataset.n_node_features  # Dimension of node features
S = dataset.n_edge_features  # Dimension of edge features
n_out = dataset.n_labels  # Dimension of the target

# Train/test split
idxs = np.random.permutation(len(dataset))
split_va, split_te = int(0.8 * len(dataset)), int(0.9 * len(dataset))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
dataset_tr = dataset[idx_tr]
dataset_va = dataset[idx_va]
dataset_te = dataset[idx_te]

loader_tr = BatchLoader(dataset_tr, batch_size=batch_size, mask=True)
loader_va = BatchLoader(dataset_va, batch_size=batch_size, mask=True)
loader_te = BatchLoader(dataset_te, batch_size=batch_size, mask=True)


################################################################################
# Build model
################################################################################
class Net(Model):
    def __init__(self):
        super().__init__()
        self.mask = GraphMasking()
        self.conv1 = GCNConv(32, activation="relu")
        self.pool = MinCutPool(N // 2)
        self.conv2 = GCNConv(32, activation="relu")
        self.global_pool = GlobalSumPool()
        self.dense1 = Dense(n_out)

    def call(self, inputs):
        x, a = inputs
        x = self.mask(x)
        x = self.conv1([x, a])
        x_pool, a_pool = self.pool([x, a])
        x_pool = self.conv2([x_pool, a_pool])
        output = self.global_pool(x_pool)
        output = self.dense1(output)

        return output


model = Net()
opt = Adam(lr=learning_rate)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["acc"])

################################################################################
# Fit model
################################################################################
model.fit(
    loader_tr.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    epochs=epochs,
    validation_data=loader_va,
    validation_steps=loader_va.steps_per_epoch,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
)

################################################################################
# Evaluate model
################################################################################
print("Testing model")
loss, acc = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done. Test loss: {}. Test acc: {}".format(loss, acc))
