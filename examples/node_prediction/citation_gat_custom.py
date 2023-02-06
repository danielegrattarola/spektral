"""
This script is an extension of the citation_gcn_custom.py script.
It shows how to train GAT (with the same experimental setting of the original
paper), using faster training and test functions.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.datasets.citation import Cora
from spektral.layers import GATConv
from spektral.transforms import AdjToSpTensor, LayerPreprocess
from spektral.utils import tic, toc

tf.random.set_seed(0)

# Load data
dataset = Cora(normalize_x=True, transforms=[LayerPreprocess(GATConv), AdjToSpTensor()])
graph = dataset[0]
x, a, y = graph.x, graph.a, graph.y
mask_tr, mask_va, mask_te = dataset.mask_tr, dataset.mask_va, dataset.mask_te

l2_reg = 2.5e-4
# Define model
x_in = Input(shape=(dataset.n_node_features,))
a_in = Input(shape=(None,), sparse=True)
x_1 = Dropout(0.6)(x_in)
x_1 = GATConv(
    8,
    attn_heads=8,
    concat_heads=True,
    dropout_rate=0.6,
    activation="elu",
    kernel_regularizer=l2(l2_reg),
    attn_kernel_regularizer=l2(l2_reg),
    bias_regularizer=l2(l2_reg),
)([x_1, a_in])
x_2 = Dropout(0.6)(x_1)
x_2 = GATConv(
    dataset.n_labels,
    attn_heads=1,
    concat_heads=False,
    dropout_rate=0.6,
    activation="softmax",
    kernel_regularizer=l2(l2_reg),
    attn_kernel_regularizer=l2(l2_reg),
    bias_regularizer=l2(l2_reg),
)([x_2, a_in])

# Build model
model = Model(inputs=[x_in, a_in], outputs=x_2)
optimizer = Adam(learning_rate=5e-3)
loss_fn = CategoricalCrossentropy()


# Training step
@tf.function
def train():
    with tf.GradientTape() as tape:
        predictions = model([x, a], training=True)
        loss = loss_fn(y[mask_tr], predictions[mask_tr])
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


@tf.function
def evaluate():
    predictions = model([x, a], training=False)
    losses = []
    accuracies = []
    for mask in [mask_tr, mask_va, mask_te]:
        loss = loss_fn(y[mask], predictions[mask])
        loss += sum(model.losses)
        losses.append(loss)
        acc = tf.reduce_mean(categorical_accuracy(y[mask], predictions[mask]))
        accuracies.append(acc)
    return losses, accuracies


best_val_loss = 99999
best_test_acc = 0
current_patience = patience = 100
epochs = 999999
tic()
for epoch in range(1, epochs + 1):
    train()
    l, a = evaluate()
    print(
        "Loss tr: {:.4f}, Acc tr: {:.4f}, "
        "Loss va: {:.4f}, Acc va: {:.4f}, "
        "Loss te: {:.4f}, Acc te: {:.4f}".format(l[0], a[0], l[1], a[1], l[2], a[2])
    )
    if l[1] < best_val_loss:
        best_val_loss = l[1]
        best_test_acc = a[2]
        current_patience = patience
        print("Improved")
    else:
        current_patience -= 1
        if current_patience == 0:
            print("Test accuracy: {}".format(best_test_acc))
            break
toc("GAT ({} epochs)".format(epoch))
