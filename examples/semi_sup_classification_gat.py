from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from spektral.datasets import citation
from spektral.layers import GraphAttention
from spektral.utils.logging import init_logging
from spektral.utils.misc import add_eye

# Load data
dataset = 'cora'
adj, node_features, y_train, y_val, y_test, train_mask, val_mask, test_mask = citation.load_data(dataset)

# Parameters
N = node_features.shape[0]    # Number of nodes in the graph
F = node_features.shape[1]    # Original feature dimensionality
n_classes = y_train.shape[1]  # Number of classes
gat_channels = 8              # Output size of first GraphAttention layer
n_attn_heads = 8              # Number of attention heads in first GAT layer
dropout_rate = 0.6            # Dropout rate applied to the input of GAT layers
l2_reg = 5e-4/2               # Regularization rate for l2
learning_rate = 5e-3          # Learning rate for SGD
epochs = 10000                # Number of epochs to train for
es_patience = 100             # Patience fot early stopping
log_dir = init_logging()      # Create log directory and file

# Preprocessing operations
node_features = citation.preprocess_features(node_features)
adj = add_eye(adj).toarray()  # Add self-loops

# Model definition
X_in = Input(shape=(F, ))
A_in = Input(shape=(N, ))

dropout_1 = Dropout(dropout_rate)(X_in)
graph_attention_1 = GraphAttention(gat_channels,
                                   attn_heads=n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg))([dropout_1, A_in])
dropout_2 = Dropout(dropout_rate)(graph_attention_1)
graph_attention_2 = GraphAttention(n_classes,
                                   attn_heads=1,
                                   attn_heads_reduction='average',
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
es_callback = EarlyStopping(monitor='val_weighted_acc', patience=es_patience)
tb_callback = TensorBoard(log_dir=log_dir, batch_size=N)
mc_callback = ModelCheckpoint(log_dir + 'best_model.h5',
                              monitor='val_weighted_acc',
                              save_best_only=True,
                              save_weights_only=True)

# Train model
validation_data = ([node_features, adj], y_val, val_mask)
model.fit([node_features, adj],
          y_train,
          sample_weight=train_mask,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,  # Shuffling data means shuffling the whole graph
          callbacks=[es_callback, tb_callback, mc_callback])

# Load best model
model.load_weights(log_dir + 'best_model.h5')

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([node_features, adj],
                              y_test,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
