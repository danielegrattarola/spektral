"""
This script is an extension of the citation_gcn_fast.py script.
It shows how to train GAT (with the same experimental setting of the original
paper), using faster training and test functions.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.datasets import citation
from spektral.layers import GraphAttention
from spektral.layers import ops
from spektral.utils import tic, toc

# Load data
A, X, y, train_mask, val_mask, test_mask = citation.load_data('cora')
fltr = A.astype('f4')
fltr = ops.sp_matrix_to_sp_tensor(fltr)
X = X.toarray()

# Define model
X_in = Input(shape=(X.shape[1], ))
fltr_in = Input(shape=(X.shape[0], ), sparse=True)
X_1 = Dropout(0.6)(X_in)
X_1 = GraphAttention(8,
                     attn_heads=8,
                     concat_heads=True,
                     dropout_rate=0.6,
                     activation='elu',
                     kernel_regularizer=l2(5e-4),
                     attn_kernel_regularizer=l2(5e-4),
                     bias_regularizer=l2(5e-4))([X_1, fltr_in])
X_2 = Dropout(0.6)(X_1)
X_2 = GraphAttention(y.shape[1],
                     attn_heads=1,
                     concat_heads=True,
                     dropout_rate=0.6,
                     activation='softmax',
                     kernel_regularizer=l2(5e-4),
                     attn_kernel_regularizer=l2(5e-4),
                     bias_regularizer=l2(5e-4))([X_2, fltr_in])

# Build model
model = Model(inputs=[X_in, fltr_in], outputs=X_2)
optimizer = Adam(lr=5e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
loss_fn = model.loss_functions[0]


# Training step
@tf.function
def train():
    with tf.GradientTape() as tape:
        predictions = model([X, fltr], training=True)
        loss = loss_fn(y[train_mask], predictions[train_mask])
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


@tf.function
def test():
    predictions = model([X, fltr], training=False)
    losses = []
    accuracies = []
    for mask in [train_mask, val_mask, test_mask]:
        loss = loss_fn(y[mask], predictions[mask])
        loss += sum(model.losses)
        losses.append(loss)
        acc = tf.reduce_mean(categorical_accuracy(y[mask], predictions[mask]))
        accuracies.append(acc)
    return losses, accuracies


best_val_loss = 99999
best_test_acc = 0
current_patience = patience = 100
tic()
for epoch in range(1, 99999):
    train()
    l, a = test()
    print('Loss tr: {:.4f}, Acc tr: {:.4f}, '
          'Loss va: {:.4f}, Acc va: {:.4f}, '
          'Loss te: {:.4f}, Acc te: {:.4f}'
          .format(l[0], a[0], l[1], a[1], l[2], a[2]))
    if l[1] < best_val_loss:
        best_val_loss = l[1]
        best_test_acc = a[2]
        current_patience = patience
        print('Improved')
    else:
        current_patience -= 1
        if current_patience == 0:
            print('Best test acc: {}'.format(best_test_acc))
            break
toc('GAT ({} epochs)'.format(epoch))
