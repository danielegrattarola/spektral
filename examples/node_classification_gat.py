"""
This example implements the experiments on citation networks from the paper:

Graph Attention Networks (https://arxiv.org/abs/1710.10903)
Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
"""

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.datasets import citation
from spektral.layers import GraphAttention
from spektral.utils.misc import add_eye

# Load data
dataset = "cora"
A, X, y, train_mask, val_mask, test_mask = citation.load_data(dataset)


# Parameters
channels = 8  # Number of channel in each head of the first GAT layer
n_attn_heads = 8  # Number of attention heads in first GAT layer
N = X.shape[0]  # Number of nodes in the graph
F = X.shape[1]  # Original feature dimensionality
n_classes = y.shape[1]  # Number of classes
dropout = 0.5  # Dropout rate applied to the features and adjacency matrix
l2_reg = 0  # Regularization rate for l2
learning_rate = 3e-3  # Learning rate for SGD
epochs = 1000  # Number of training epochs
es_patience = 50  # Patience for early stopping

# Preprocessing operations
A = add_eye(A).toarray()  # Add self-loops
X = X.todense()


# Model definition
X_in = Input(shape=(F,))
A_in = Input(shape=(N,), sparse=False)

net = X_in

net = Dropout(dropout)(net)
net = GraphAttention(
    channels,
    attn_heads=n_attn_heads,
    concat_heads=True,
    dropout_rate=dropout,
    activation="elu",
    kernel_regularizer=l2(l2_reg),
    attn_kernel_regularizer=l2(l2_reg),
    use_bias=False,
)([X_in, A_in])
net = Dropout(dropout)(net)
net = GraphAttention(
    n_classes,
    attn_heads=1,
    concat_heads=False,
    dropout_rate=dropout,
    activation="softmax",
    kernel_regularizer=l2(l2_reg),
    attn_kernel_regularizer=l2(l2_reg),
    use_bias=False,
)([net, A_in])

# Build model
model = Model(inputs=[X_in, A_in], outputs=net)
optimizer = Adam(lr=learning_rate)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", weighted_metrics=["acc"]
)
model.summary()

# Train model
validation_data = ([X, A], y, val_mask)
model.fit(
    [X, A],
    y,
    sample_weight=train_mask,
    epochs=epochs,
    batch_size=N,
    validation_data=validation_data,
    shuffle=False,  # Shuffling data means shuffling the whole graph
    callbacks=[EarlyStopping(patience=es_patience, restore_best_weights=True)],
)

# Evaluate model
print("Evaluating model.")
eval_results = model.evaluate([X, A], y, sample_weight=test_mask, batch_size=N)
print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))
