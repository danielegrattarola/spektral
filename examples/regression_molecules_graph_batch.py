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
from spektral.utils import Batch, batch_iterator
from spektral.utils import label_to_one_hot

# Load data
A, X, _, y = qm9.load_data(return_type='numpy',
                           nf_keys='atomic_num',
                           ef_keys='type',
                           self_loops=True,
                           auto_pad=False,
                           amount=1000)
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
I_in = Input(batch_shape=(None, 1), dtype='int64')

# The inputs will have an arbitrary dimension, while the targets consist of
# batch_size values.
# However, Keras expects the inputs to have the same dimension as the output.
# This is a hack in Tensorflow to bypass the requirements of Keras.
# We use a dynamically initialized tf.Dataset to feed the target values to the
# model at training time.
target_ph = tf.placeholder(tf.float32, shape=(None, 1))
target_data = tf.data.Dataset.from_tensor_slices(target_ph)
target_data = target_data.batch(batch_size)
target_iter = target_data.make_initializable_iterator()
target = target_iter.get_next()

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
batches_train = batch_iterator([A_train, X_train, y_train], batch_size=batch_size, epochs=epochs)
loss = 0
batch_index = 0
batches_in_epoch = np.ceil(len(A_train) / batch_size)

# Training loop
for b in batches_train:
    batch = Batch(b[0], b[1])
    y_ = b[2]
    sess.run(target_iter.initializer, feed_dict={target_ph: y_})
    loss += model.train_on_batch(list(batch.get('XAI')), None)

    batch_index += 1
    if batch_index == batches_in_epoch:
        print('Loss: {}'.format(loss / batches_in_epoch))
        loss = 0
        batch_index = 0

# Test setup
batches_test = batch_iterator([A_test, X_test, y_test], batch_size=batch_size)
loss = 0
batches_in_epoch = np.ceil(len(A_test) / batch_size)

# Test loop
for b in batches_test:
    batch = Batch(b[0], b[1])
    y_ = b[2]
    sess.run(target_iter.initializer, feed_dict={target_ph: y_})
    loss += model.test_on_batch(list(batch.get('XAI')), None)
print('---------------------------------------------')
print('Test loss: {}'.format(loss / batches_in_epoch))
