from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

from spektral.datasets import delaunay
from spektral.layers import GraphAttention, GlobalAttentionPool
from spektral.utils import localpooling_filter
from spektral.utils.logging import init_logging

# Load data
adj, x, y = delaunay.generate_data(return_type='numpy', classes=[0, 5])

# Parameters
N = x.shape[-2]           # Number of nodes in the graphs
F = x.shape[-1]           # Original feature dimensionality
n_classes = y.shape[-1]   # Number of classes
l2_reg = 5e-4             # Regularization rate for l2
learning_rate = 1e-3      # Learning rate for Adam
epochs = 200              # Number of training epochs
batch_size = 32           # Batch size
es_patience = 10          # Patience fot early stopping
log_dir = init_logging()  # Create log directory and file

# Preprocessing
fltr = localpooling_filter(adj.copy())

# Train/test split
fltr_train, fltr_test, \
x_train, x_test,       \
y_train, y_test = train_test_split(fltr, x, y, test_size=0.1)

# Model definition
X_in = Input(shape=(N, F))
filter_in = Input((N, N))

gc1 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([X_in, filter_in])
gc2 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([gc1, filter_in])
pool = GlobalAttentionPool(128)(gc2)

output = Dense(n_classes, activation='softmax')(pool)

# Build model
model = Model(inputs=[X_in, filter_in], outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
model.summary()

# Callbacks
es_callback = EarlyStopping(monitor='val_acc', patience=es_patience)

# Train model
model.fit([x_train, fltr_train],
          y_train,
          batch_size=batch_size,
          validation_split=0.1,
          epochs=epochs,
          callbacks=[es_callback])

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([x_test, fltr_test],
                              y_test,
                              batch_size=batch_size)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
