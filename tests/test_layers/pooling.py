import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
from keras import Input, Model
from keras import backend as K
from keras.backend import tf
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import kneighbors_graph

from spektral.layers import TopKPooling
from spektral.layers.ops import sp_matrix_to_sp_tensor_value

np.random.seed(None)
ITER = 20000
N = 100
F = 10
X, y = make_blobs(n_samples=N, centers=4, n_features=F, random_state=None)
X = X.astype(np.float32)

A = kneighbors_graph(X, n_neighbors=25, mode='distance').todense()
A = np.asarray(A)
A = np.maximum(A, A.T)
A /= A.max()  # normalize in [0,1]
A = sp.csr_matrix(A, dtype=np.float32)

###############################################################################
# LAYER IMPLEMENTATION
###############################################################################
X_in = Input(tensor=tf.placeholder(tf.float32, shape=(None, F), name='X_in'))
A_in = Input(tensor=tf.sparse_placeholder(tf.float32, shape=(None, None)), sparse=True)
S_in = Input(tensor=tf.placeholder(tf.int32, shape=(None,), name='segment_ids_in'))
target = Input(tensor=tf.placeholder(tf.float32, shape=(None, F), name='target'))

pool1, adj1, seg1 = TopKPooling(k=50)([X_in, A_in, S_in])
pool2, adj2, seg2 = TopKPooling(k=25)([pool1, adj1, seg1])
pool3, adj3, seg3 = TopKPooling(k=5)([pool2, adj2, seg2])

model = Model([X_in, A_in, S_in], pool3)
model.compile('adam', 'mse', target_tensors=[target])

# Setup
sess = K.get_session()
loss = model.total_loss
opt = tf.train.AdamOptimizer()
train_step = opt.minimize(loss)
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Run graph
tr_feed_dict = {X_in: X,
                A_in: sp_matrix_to_sp_tensor_value(A),
                S_in: y}  # We use y instead of segment ids only for plotting (we don't need S in this example anyway)
X_out = sess.run([model.output], feed_dict=tr_feed_dict)[0]
S_out = sess.run([seg3], feed_dict=tr_feed_dict)[0]

# Fit layer
tr_feed_dict = {X_in: X,
                A_in: sp_matrix_to_sp_tensor_value(A),
                S_in: y,
                target: np.random.normal(size=(5, F)),
                'top_k_pooling_3_sample_weights:0': np.ones(1)}
outs = sess.run([train_step, loss], feed_dict=tr_feed_dict)


# Plots
print('Nodes left: ', X_out.shape[0])
print('I: {} - II: {} - III: {} - IV: {}'
      ''.format(np.sum(S_out == 0), np.sum(S_out == 1),
                np.sum(S_out == 2), np.sum(S_out == 3)))

# Plot graphs
plt.figure()
plt.subplot(231)
G = nx.from_numpy_array(A.todense())
nx.draw_networkx(G, with_labels=False, node_size=20, edge_color='lightgray', node_color=y / y.max(), linewidths=1)

plt.subplot(234)
lab = sess.run([seg1], feed_dict=tr_feed_dict)[0]
adj_ = sess.run([adj1], feed_dict=tr_feed_dict)[0]
A_pool = sp.csr_matrix((adj_.values, (adj_.indices[:, 0], adj_.indices[:, 1])), shape=adj_.dense_shape)
G = nx.from_scipy_sparse_matrix(A_pool)
nx.draw_networkx(G, with_labels=False, node_size=20, edge_color='lightgray', node_color=lab / lab.max(), linewidths=1)

plt.subplot(235)
lab = sess.run([seg2], feed_dict=tr_feed_dict)[0]
adj_ = sess.run([adj2], feed_dict=tr_feed_dict)[0]
A_pool = sp.csr_matrix((adj_.values, (adj_.indices[:, 0], adj_.indices[:, 1])), shape=adj_.dense_shape)
G = nx.from_scipy_sparse_matrix(A_pool)
nx.draw_networkx(G, with_labels=False, node_size=20, edge_color='lightgray', node_color=lab / lab.max(), linewidths=1)

plt.subplot(236)
lab = sess.run([seg3], feed_dict=tr_feed_dict)[0]
adj_ = sess.run([adj3], feed_dict=tr_feed_dict)[0]
A_pool = sp.csr_matrix((adj_.values, (adj_.indices[:, 0], adj_.indices[:, 1])), shape=adj_.dense_shape)
G = nx.from_scipy_sparse_matrix(A_pool)
nx.draw_networkx(G, with_labels=False, node_size=20, edge_color='lightgray', node_color=lab / lab.max(), linewidths=1)
plt.show()
