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
from spektral.utils import init_logging, label_to_one_hot

# Load data
adj, nf, ef, y = qm9.load_data(return_type='numpy',
                               nf_keys='atomic_num',
                               ef_keys='type',
                               self_loops=True)
y = y[['cv']].values  # Heat capacity at 298.15K

# Preprocessing
uniq_nf = np.unique(nf)
nf = label_to_one_hot(nf, uniq_nf)
uniq_ef = np.unique(ef)
ef = label_to_one_hot(ef, uniq_ef)
y = StandardScaler().fit_transform(y).reshape(-1, y.shape[-1])

# Parameters
N = nf.shape[-2]          # Number of nodes in the graphs
F = nf.shape[-1]          # Node features dimensionality
S = ef.shape[-1]          # Edge features dimensionality
n_out = y.shape[-1]       # Dimensionality of the target
learning_rate = 1e-3      # Learning rate for SGD
epochs = 25               # Number of training epochs
batch_size = 64           # Batch size
es_patience = 5           # Patience fot early stopping
log_dir = init_logging()  # Create log directory and file

# Train/test split
adj_train, adj_test, \
nf_train, nf_test,   \
ef_train, ef_test,   \
y_train, y_test = train_test_split(adj, nf, ef, y, test_size=0.1)

# Model definition
nf_in = Input(shape=(N, F))
adj_in = Input(shape=(N, N))
ef_in = Input(shape=(N, N, S))

gc1 = EdgeConditionedConv(32, activation='relu')([nf_in, adj_in, ef_in])
gc2 = EdgeConditionedConv(64, activation='relu')([gc1, adj_in, ef_in])
pool = GlobalAttentionPool(128)(gc2)
dense1 = Dense(128, activation='relu')(pool)

output = Dense(n_out)(dense1)

# Build model
model = Model(inputs=[nf_in, adj_in, ef_in], outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='mse')
model.summary()

# Callbacks
es_callback = EarlyStopping(monitor='val_loss', patience=es_patience)

# Train model
model.fit([nf_train, adj_train, ef_train],
          y_train,
          batch_size=batch_size,
          validation_split=0.1,
          epochs=epochs,
          callbacks=[es_callback])

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([nf_test, adj_test, ef_test],
                              y_test,
                              batch_size=batch_size)
print('Done.\n'
      'Test loss: {}'.format(eval_results))

# Plot predictions
preds = model.predict([nf_test, adj_test, ef_test])

plt.figure()
plt.scatter(preds, y_test, alpha=0.3)
plt.plot(range(-6, 6), range(-6, 6))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(log_dir + 'pred_v_true.png')
