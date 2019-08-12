"""
This example shows how to perform regression of molecular properties with the
QM9 database, using a simple GNN in graph batch mode (note that in this example
we ignore edge attributes).
Note that the main training loop is written in TensorFlow, because we need to
avoid the restriction imposed by Keras that the input and the output have the
same first dimension. This is the most efficient way of training a GNN in
graph batch mode.
"""

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from spektral.datasets import qm9
from spektral.layers import GraphConv, GlobalAvgPool
from spektral.layers.ops import sp_matrix_to_sp_tensor_value
from spektral.utils import Batch, batch_iterator
from spektral.utils import label_to_one_hot

np.random.seed(0)
SW_KEY = 'dense_2_sample_weights:0'  # Keras automatically creates a placeholder for sample weights, which must be fed

# Load data
A, X, _, y = qm9.load_data(return_type='numpy',
                           nf_keys='atomic_num',
                           ef_keys='type',
                           self_loops=True,
                           auto_pad=False,
                           amount=1000)  # Set to None to train on whole dataset
y = y[['cv']].values  # Heat capacity at 298.15K

# Preprocessing
uniq_X = np.unique([v for x in X for v in np.unique(x)])
X = [label_to_one_hot(x, uniq_X) for x in X]
y = StandardScaler().fit_transform(y).reshape(-1, y.shape[-1])

# Parameters
F = X[0].shape[-1]    # Dimension of node features
n_out = y.shape[-1]   # Dimension of the target
learning_rate = 1e-3  # Learning rate
epochs = 25           # Number of training epochs
batch_size = 64       # Batch size

# Train/test split
A_train, A_test, \
X_train, X_test, \
y_train, y_test = train_test_split(A, X, y, test_size=0.1)

# Model definition
X_in = Input(batch_shape=(None, F))
A_in = Input(batch_shape=(None, None), sparse=True)
I_in = Input(batch_shape=(None, ), dtype='int64')
target = Input(tensor=tf.placeholder(tf.float32, shape=(None, n_out), name='target'))

gc1 = GraphConv(64, activation='relu')([X_in, A_in])
gc2 = GraphConv(64, activation='relu')([gc1, A_in])
pool = GlobalAvgPool()([gc2, I_in])
dense1 = Dense(64, activation='relu')(pool)
output = Dense(n_out)(dense1)

# Build model
model = Model(inputs=[X_in, A_in, I_in], outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='mse', target_tensors=target)
model.summary()

# Training setup
sess = K.get_session()
loss = model.total_loss
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_step = opt.minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

batches_train = batch_iterator([A_train, X_train, y_train], batch_size=batch_size, epochs=epochs)
model_loss = 0
batch_index = 0
batches_in_epoch = np.ceil(len(A_train) / batch_size)

# Training loop
for b in batches_train:
    batch = Batch(b[0], b[1])
    X_, A_, I_ = batch.get('XAI')
    y_ = b[2]
    tr_feed_dict = {X_in: X_,
                    A_in: sp_matrix_to_sp_tensor_value(A_),
                    I_in: I_,
                    target: y_,
                    SW_KEY: np.ones((1,))}
    outs = sess.run([train_step, loss], feed_dict=tr_feed_dict)
    model_loss += outs[-1]

    batch_index += 1
    if batch_index == batches_in_epoch:
        print('Loss: {}'.format(model_loss / batches_in_epoch))
        model_loss = 0
        batch_index = 0

# Test setup
batches_test = batch_iterator([A_test, X_test, y_test], batch_size=batch_size)
model_loss = 0
batches_in_epoch = np.ceil(len(A_test) / batch_size)

# Test loop
for b in batches_test:
    batch = Batch(b[0], b[1])
    X_, A_, I_ = batch.get('XAI')
    y_ = b[2]
    tr_feed_dict = {X_in: X_,
                    A_in: sp_matrix_to_sp_tensor_value(A_),
                    I_in: I_,
                    target: y_,
                    SW_KEY: np.ones((1,))}
    model_loss += sess.run([loss], feed_dict=tr_feed_dict)[0]
print('---------------------------------------------')
print('Test loss: {}'.format(model_loss / batches_in_epoch))
