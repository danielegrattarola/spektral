"""
This example implements the experiments on citation networks from the paper:

Simplifying Graph Convolutional Networks (https://arxiv.org/abs/1902.07153)
Felix Wu, Tianyi Zhang, Amauri Holanda de Souza Jr., Christopher Fifty, Tao Yu, Kilian Q. Weinberger
"""

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.datasets import citation
from spektral.layers import GraphConv
from spektral.utils.convolution import gcn_filter

# Load data
dataset = 'cora'
A, X, y, train_mask, val_mask, test_mask = citation.load_data(dataset)

# Parameters
K = 2                   # Degree of propagation
N = X.shape[0]          # Number of nodes in the graph
F = X.shape[1]          # Original size of node features
n_classes = y.shape[1]  # Number of classes
l2_reg = 5e-6           # L2 regularization rate
learning_rate = 0.2     # Learning rate
epochs = 20000          # Number of training epochs
es_patience = 200       # Patience for early stopping

# Preprocessing operations
fltr = gcn_filter(A).astype('f4')
X = X.toarray()

# Pre-compute propagation
for i in range(K - 1):
    fltr = fltr.dot(fltr)
fltr.sort_indices()

# Model definition
X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)
output = GraphConv(n_classes,
                   activation='softmax',
                   kernel_regularizer=l2(l2_reg),
                   use_bias=False)([X_in, fltr_in])

# Build model
model = Model(inputs=[X_in, fltr_in], outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

# Train model
validation_data = ([X, fltr], y, val_mask)
model.fit([X, fltr],
          y,
          sample_weight=train_mask,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,  # Shuffling data means shuffling the whole graph
          callbacks=[
              EarlyStopping(patience=es_patience,  restore_best_weights=True)
          ])

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([X, fltr],
                              y,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
