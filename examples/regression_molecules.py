"""
This example shows how to perform regression of molecular properties with the
QM9 database, using a GNN based on edge-conditioned convolutions in batch mode.
"""


import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from spektral.datasets import qm9
from spektral.layers import EdgeConditionedConv, GlobalAttentionPool
from spektral.utils import label_to_one_hot

# Load data
A, X, E, y = qm9.load_data(return_type='numpy',
                           nf_keys='atomic_num',
                           ef_keys='type',
                           self_loops=True,
                           amount=1000)  # Set to None to train on whole dataset
y = y[['cv']].values  # Heat capacity at 298.15K

# Preprocessing
uniq_X = np.unique(X)
X = label_to_one_hot(X, uniq_X)
uniq_E = np.unique(E)
E = label_to_one_hot(E, uniq_E)
y = StandardScaler().fit_transform(y).reshape(-1, y.shape[-1])

# Parameters
N = X.shape[-2]           # Number of nodes in the graphs
F = X.shape[-1]           # Node features dimensionality
S = E.shape[-1]           # Edge features dimensionality
n_out = y.shape[-1]       # Dimensionality of the target
learning_rate = 1e-3      # Learning rate for SGD
epochs = 25               # Number of training epochs
batch_size = 64           # Batch size
es_patience = 5           # Patience fot early stopping

# Train/test split
A_train, A_test, \
X_train, X_test, \
E_train, E_test, \
y_train, y_test = train_test_split(A, X, E, y, test_size=0.1)

# Model definition
X_in = Input(shape=(N, F))
A_in = Input(shape=(N, N))
E_in = Input(shape=(N, N, S))

gc1 = EdgeConditionedConv(32, activation='relu')([X_in, A_in, E_in])
gc2 = EdgeConditionedConv(64, activation='relu')([gc1, A_in, E_in])
pool = GlobalAttentionPool(128)(gc2)
dense1 = Dense(128, activation='relu')(pool)

output = Dense(n_out)(dense1)

# Build model
model = Model(inputs=[X_in, A_in, E_in], outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='mse')
model.summary()

# Callbacks
es_callback = EarlyStopping(monitor='val_loss', patience=es_patience)

# Train model
model.fit([X_train, A_train, E_train],
          y_train,
          batch_size=batch_size,
          validation_split=0.1,
          epochs=epochs,
          callbacks=[es_callback])

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([X_test, A_test, E_test],
                              y_test,
                              batch_size=batch_size)
print('Done.\n'
      'Test loss: {}'.format(eval_results))

# Plot predictions
preds = model.predict([X_test, A_test, E_test])

plt.figure()
plt.scatter(preds, y_test, alpha=0.3)
plt.plot(range(-6, 6), range(-6, 6))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('pred_v_true.png')
