"""
This example implements the experiments on citation networks from the paper:

Simplifying Graph Convolutional Networks (https://arxiv.org/abs/1902.07153)
Felix Wu, Tianyi Zhang, Amauri Holanda de Souza Jr., Christopher Fifty, Tao Yu, Kilian Q. Weinberger
"""

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from spektral.datasets import citation
from spektral.layers import GraphConv
from spektral.utils.convolution import localpooling_filter

# Load data
dataset = 'cora'
adj, node_features, y_train, y_val, y_test, train_mask, val_mask, test_mask = citation.load_data(dataset)

# Parameters
K = 2                         # Degree of propagation
N = node_features.shape[0]    # Number of nodes in the graph
F = node_features.shape[1]    # Original feature dimensionality
n_classes = y_train.shape[1]  # Number of classes
l2_reg = 5e-6                 # Regularization rate for l2
learning_rate = 0.2           # Learning rate for SGD
epochs = 200                  # Number of training epochs
es_patience = 100             # Patience for early stopping

# Preprocessing operations
node_features = citation.preprocess_features(node_features)
fltr = localpooling_filter(adj)

# Pre-compute propagation
for i in range(K - 1):
    fltr = fltr.dot(fltr)

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

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_weighted_acc', patience=es_patience),
    ModelCheckpoint('best_model.h5', monitor='val_weighted_acc',
                    save_best_only=True, save_weights_only=True)
]

# Train model
validation_data = ([node_features, fltr], y_val, val_mask)
model.fit([node_features, fltr],
          y_train,
          sample_weight=train_mask,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,  # Shuffling data means shuffling the whole graph
          callbacks=callbacks)

# Load best model
model.load_weights('best_model.h5')

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([node_features, fltr],
                              y_test,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
