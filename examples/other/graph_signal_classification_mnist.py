import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.regularizers import l2

from spektral.data import PackedBatchLoader
from spektral.datasets.mnist import MNIST
from spektral.layers import GraphConv
from spektral.layers.ops import sp_matrix_to_sp_tensor

# Parameters
batch_size = 32  # Batch size
epochs = 1000    # Number of training epochs
patience = 10    # Patience for early stopping
l2_reg = 5e-4    # Regularization rate for l2

# Load data
data = MNIST()

# The adjacency matrix is stored as an attribute of the dataset.
# Create filter for GCN and convert to sparse tensor.
adj = data.a
adj = GraphConv.preprocess(adj)
adj = sp_matrix_to_sp_tensor(adj)

# Train/valid/test split
data_tr, data_te = data[:-10000], data[-10000:]
np.random.shuffle(data_tr)
data_tr, data_va = data_tr[:-10000], data_tr[-10000:]


# Build model
class Net(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GraphConv(32, activation='elu', kernel_regularizer=l2(l2_reg))
        self.conv2 = GraphConv(32, activation='elu', kernel_regularizer=l2(l2_reg))
        self.flatten = Flatten()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(10, activation='softmax')  # MNIST has 10 classes

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
model.compile('adam', 'sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])


# Evaluation function
def evaluate(loader):
    step = 0
    results = []
    for batch in loader:
        step += 1
        x, y = batch
        l, a = model.test_on_batch([x, adj], y)
        results.append((l, a))
        if step == loader.steps_per_epoch:
            return np.mean(results, 0)


# Setup training
best_val_loss = 99999
current_patience = patience
step = 0

# We can use PackedBatchLoader because we only need to create batches of node
# features with the same dimensions.
loader_tr = PackedBatchLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_va = PackedBatchLoader(data_va, batch_size=batch_size)
loader_te = PackedBatchLoader(data_te, batch_size=batch_size)

# Training loop
results_tr = []
for batch in loader_tr:
    step += 1

    # Training step
    x, y = batch
    l, a = model.train_on_batch([x, adj], y)
    results_tr.append((l, a))

    if step == loader_tr.steps_per_epoch:
        results_va = evaluate(loader_va)
        if results_va[0] < best_val_loss:
            best_val_loss = results_va[0]
            current_patience = patience
            results_te = evaluate(loader_te)
        else:
            current_patience -= 1
            if current_patience == 0:
                print('Early stopping')
                break

        # Print results
        results_tr = np.mean(results_tr, 0)
        print('Train loss: {:.4f}, acc: {:.4f} | '
              'Valid loss: {:.4f}, acc: {:.4f} | '
              'Test loss: {:.4f}, acc: {:.4f}'
              .format(*results_tr, *results_va, *results_te))

        # Reset epoch
        results_tr = []
        step = 0
