"""
This example implements the experiments on citation networks from the paper:

Graph Attention Networks (https://arxiv.org/abs/1710.10903)
Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
"""

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from spektral.datasets import citation
from spektral.layers import GraphAttention
from spektral.utils.misc import add_eye

# Load data
dataset = 'cora'
A, X, y, train_mask, val_mask, test_mask = citation.load_data(dataset)

# Parameters
gat_channels = 8        # Output size of first GraphAttention layer
n_attn_heads = 8        # Number of attention heads in first GAT layer
N = X.shape[0]          # Number of nodes in the graph
F = X.shape[1]          # Original feature dimensionality
n_classes = y.shape[1]  # Number of classes
dropout_rate = 0.25     # Dropout rate applied to the input of GAT layers
l2_reg = 5e-4           # Regularization rate for l2
learning_rate = 1e-2    # Learning rate for SGD
epochs = 20000          # Number of training epochs
es_patience = 200       # Patience fot early stopping

# Preprocessing operations
A = add_eye(A).toarray()  # Add self-loops

# Model definition
X_in = Input(shape=(F, ))
A_in = Input(shape=(N, ))

dropout_1 = Dropout(dropout_rate)(X_in)
graph_attention_1 = GraphAttention(gat_channels,
                                   attn_heads=n_attn_heads,
                                   concat_heads=True,
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg))([dropout_1, A_in])
dropout_2 = Dropout(dropout_rate)(graph_attention_1)
graph_attention_2 = GraphAttention(n_classes,
                                   attn_heads=1,
                                   concat_heads=False,
                                   dropout_rate=dropout_rate,
                                   activation='softmax',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg))([dropout_2, A_in])

# Build model
model = Model(inputs=[X_in, A_in], outputs=graph_attention_2)
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
validation_data = ([X, A], y, val_mask)
model.fit([X, A],
          y,
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
eval_results = model.evaluate([X, A],
                              y,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
