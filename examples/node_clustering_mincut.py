"""
This example implements the experiments for node clustering on citation networks
from the paper:

Mincut pooling in Graph Neural Networks (https://arxiv.org/abs/1907.00481)
Filippo Maria Bianchi, Daniele Grattarola, Cesare Alippi

Note that the main training loop is written in Tensorflow, so that we can have
an in-depth look at what is going on in the unsupervised loss and study the
clustring found by the model.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.metrics.cluster import v_measure_score, homogeneity_score, completeness_score
from tqdm import tqdm

from spektral.datasets import citation
from spektral.layers.convolutional import GraphConvSkip
from spektral.layers.ops import sp_matrix_to_sp_tensor_value
from spektral.layers.pooling import MinCutPool
from spektral.utils.convolution import normalized_adjacency

np.random.seed(1)

iterations = 5000    # Training iterations
gnn_channels = 16    # Units in the GNN layer
mlp_channels = None  # Units in the MLP of MinCutPool Layer (if None, the MLP has no hidden layers)
gnn_activ = 'elu'    # Activation for the GNN layer
mlp_activ = None     # Activation for the hidden layers of MinCutPool
lr = 5e-4            # Learning rate

################################################################################
# LOAD DATASET
################################################################################
A, X, y, _, _, _ = citation.load_data('cora')
A_norm = normalized_adjacency(A)  # Normalize adjacency matrix
X = X.todense()
n_feat = X.shape[-1]
y = np.argmax(y, axis=-1)
n_clust = y.max() + 1

################################################################################
# MODEL
################################################################################
X_in = Input(tensor=tf.placeholder(tf.float32, shape=(None, n_feat), name='X_in'))
A_in = Input(tensor=tf.sparse_placeholder(tf.float32, shape=(None, None)), name='A_in', sparse=True)

X_1 = GraphConvSkip(gnn_channels,
                    kernel_initializer='he_normal',
                    activation=gnn_activ)([X_in, A_in])
pool1, adj1, S = MinCutPool(k=n_clust,
                            h=mlp_channels,
                            activation=mlp_activ)([X_1, A_in])

model = Model([X_in, A_in], [pool1, S])
model.compile('adam', None)

################################################################################
# TRAINING
################################################################################
# Setup
sess = K.get_session()

loss = model.total_loss        # The full unsupervised loss of MinCutPool
mincut_loss = model.losses[0]  # The minCUT loss of MinCutPool
ortho_loss = model.losses[1]   # The orthogonality loss of MinCutPool

opt = tf.train.AdamOptimizer(learning_rate=lr)
train_step = opt.minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Fit model
tr_feed_dict = {X_in: X,
                A_in: sp_matrix_to_sp_tensor_value(A_norm)}
loss_history = []
nmi_history = []
for _ in tqdm(range(iterations)):
    outs = sess.run([train_step, mincut_loss, ortho_loss, S], feed_dict=tr_feed_dict)
    loss_history.append((outs[1], outs[2], outs[1] + outs[2]))
    s = np.argmax(outs[3], axis=-1)
    nmi_history.append(v_measure_score(y, s))
loss_history = np.array(loss_history)

################################################################################
# RESULTS
################################################################################
S_ = sess.run([S], feed_dict=tr_feed_dict)[0]
s = np.argmax(S_, axis=-1)
hs = homogeneity_score(y, s)
cs = completeness_score(y, s)
nmis = v_measure_score(y, s)
print('Homogeneity: {:.3f}; Completeness: {:.3f}; NMI: {:.3f}'.format(hs, cs, nmis))

# Plots
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.plot(loss_history[:, 0], label='MinCUT loss')
plt.plot(loss_history[:, 1], label='Ortho. loss')
plt.plot(loss_history[:, 2], label='Total loss')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Iteration')

plt.subplot(122)
plt.plot(nmi_history, label='NMI')
plt.legend()
plt.ylabel('NMI')
plt.xlabel('Iteration')

plt.show()
