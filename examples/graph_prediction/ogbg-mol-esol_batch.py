"""
This example shows how to perform molecule regression with the
[Open Graph Benchmark](https://ogb.stanford.edu) `mol-esol` dataset, using a
simple GIN-based GNN with MinCutPool in batch mode.
Expect unstable training due to the small-ish size of the dataset.
"""

from ogb.graphproppred import GraphPropPredDataset, Evaluator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.datasets import ogb
from spektral.layers import GraphConv, MinCutPool, GlobalSumPool
from spektral.utils import pad_jagged_array

################################################################################
# PARAMETERS
################################################################################
learning_rate = 1e-3  # Learning rate
epochs = 99999        # Number of training epochs
batch_size = 32       # Batch size

################################################################################
# LOAD DATA
################################################################################
dataset_name = 'ogbg-molesol'
dataset = GraphPropPredDataset(name=dataset_name)
n_out = dataset.num_tasks
N = max(g[0]['num_nodes'] for g in dataset)

idx = dataset.get_idx_split()
tr_idx, va_idx, te_idx = idx["train"], idx["valid"], idx["test"]

X, A, _, y = ogb.dataset_to_numpy(dataset, dtype='f8')
A = [a.toarray() for a in A]
F = X[0].shape[-1]
X = pad_jagged_array(X, (N, F))
A = pad_jagged_array(A, (N, N))
X_tr, A_tr, y_tr = X[tr_idx], A[tr_idx], y[tr_idx]
X_va, A_va, y_va = X[va_idx], A[va_idx], y[va_idx]
X_te, A_te, y_te = X[te_idx], A[te_idx], y[te_idx]

################################################################################
# BUILD MODEL
################################################################################
X_in = Input(shape=(N, F))
A_in = Input(shape=(N, N))

X_1 = GraphConv(32, activation='relu')([X_in, A_in])
X_1, A_1 = MinCutPool(N // 2)([X_1, A_in])
X_2 = GraphConv(32, activation='relu')([X_1, A_1])
X_3 = GlobalSumPool()(X_2)
output = Dense(n_out)(X_3)

# Build model
model = Model(inputs=[X_in, A_in], outputs=output)
opt = Adam(lr=learning_rate)
model.compile(optimizer=opt, loss='mse')
model.summary()

################################################################################
# FIT MODEL
################################################################################
model.fit([X_tr, A_tr],
          y_tr,
          batch_size=batch_size,
          validation_data=([X_va, A_va], y_va),
          callbacks=[EarlyStopping(patience=200, restore_best_weights=True)],
          epochs=epochs)

################################################################################
# EVALUATE MODEL
################################################################################
print('Testing model')
evaluator = Evaluator(name=dataset_name)
y_pred = model.predict([X_te, A_te], batch_size=batch_size)
ogb_score = evaluator.eval({'y_true': y_te, 'y_pred': y_pred})

print('Done. RMSE: {:.4f}'.format(ogb_score['rmse']))
