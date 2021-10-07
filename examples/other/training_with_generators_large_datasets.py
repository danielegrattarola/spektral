# Example for loading large image datasets and using GCN Layers 
import os
import glob
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.layers import GCNConv, GlobalSumPool
from spektral.utils.sparse import sp_matrix_to_sp_tensor

# We will be using the dataset used in the competiton: 
# Plant Seedlings Classification: https://www.kaggle.com/c/plant-seedlings-classification
# Any other dataset can also be used, this dataset is structured in terms of classes in the
# train folder in which case flow_from_directory is a better option. Still this is an example
# of getting things done with a custom generator which loads images in 


train_path = '../input/plant-seedlings-classification/train'

# Get all the filenames with complete path and labels, from train folder into 2 lists
filenames, labels = [], []
label_dict = {'Black-grass':0, 'Charlock': 1, 'Cleavers': 2, 'Common Chickweed': 3, 'Common wheat': 4, 'Fat Hen': 5, 'Loose Silky-bent': 6, 'Maize': 7, 'Scentless Mayweed': 8, 'Shepherds Purse':9, 'Small-flowered Cranesbill':10, 'Sugar beet':11}
for files in glob.glob(train_path + '/*/*.*'):
    filenames.append(files)
    labels.append(label_dict[files.split('/')[-2]])

# Train Valid Split in the train dataset
filenames_shuffled, labels_shuffled = shuffle(filenames, labels)
filenames_shuffled_numpy = np.array(filenames_shuffled)
X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
    filenames_shuffled_numpy, labels_shuffled, test_size=0.2, random_state=1)

# Convert Labels to arrays
y_train = np.array(y_train)
y_val = np.array(y_val)

# Get the Adjacency Matrix, as in Mixed mode we have to call a function within read() of 
# Custom Dataset 
IMAGE_SIZE = 64
k = 8
# Creates Grid coordinates
def grid_coordinates(side):
    M = side ** 2
    x = np.linspace(0, 1, side, dtype=np.float32)
    y = np.linspace(0, 1, side, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), np.float32)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z
# Creates KNN Graph
def get_adj_from_data(X, k, **kwargs):
    A = kneighbors_graph(X, k, **kwargs).toarray()
    A = sp.csr_matrix(np.maximum(A, A.T))
    return A
# For Adjacency Matrix
def psc_grid_graph(k):
    X = grid_coordinates(IMAGE_SIZE)
    A = get_adj_from_data(X, k, mode="connectivity", metric="euclidean", include_self=False)
    return A

# Create filter for GCN and convert to sparse tensor
adj_matrix = psc_grid_graph(8)
adj_matrix = GCNConv.preprocess(adj_matrix)
adj_matrix = sp_matrix_to_sp_tensor(adj_matrix)

# Custom Image Generator using keras Sequence to get 'batch_size' images in each batch

class Image_Generator(tensorflow.keras.utils.Sequence) :
    
    def __init__(self, image_filenames, labels, batch_size) :
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
       
    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx) :
    	# Just in case we needed steps_per_epochs, or else we can use __len__() as well
        self.steps_per_epochs = self.__len__()
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        # Read in gray scale and Normalize
        batch_x = np.array([cv2.resize(cv2.imread(file_name,0), (IMAGE_SIZE, IMAGE_SIZE)) for file_name in batch_x])/255.0
        # Resize the batch in following format to use it for GCN Layer
        batch_x = batch_x.reshape(-1,IMAGE_SIZE**2,1)
        batch_y = np.array(self.labels[idx * self.batch_size : (idx+1) * self.batch_size])
        # Return a tuple containing inputs = (x, adjacency_matrix) and target for GCN
        return (batch_x, adj_matrix), batch_y

# If you try inheritance from 2 classes, Sequence from Keras.utils and Dataset from Spektral
# The scenario is a bit tough, and some functions are messed up. Instead we can directly use
# Keras Sequence to get generators. We already got the adjacency matrix as well seperately.

# Create Training and Validation Generators
training_batch_generator = Image_Generator(X_train_filenames, y_train, batch_size=32)
validation_batch_generator = Image_Generator(X_val_filenames, y_val, batch_size=32)

# Parameters
l2_reg = 5e-4         # Regularization rate for l2
learning_rate = 1e-3  # Learning rate for SGD
batch_size = 32       # Batch size
epochs = 1000        # Number of training epochs
es_patience = 20     # Patience fot early stopping

# Build Model
class Net(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GCNConv(32, activation="elu", kernel_regularizer=l2(l2_reg))
        self.conv2 = GCNConv(32, activation="elu", kernel_regularizer=l2(l2_reg))
        self.flatten = GlobalSumPool()
        #self.flatten = Flatten()
        self.fc1 = Dense(512, activation="relu")
        self.fc2 = Dense(12, activation="softmax")  

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)
        return output

model = Net()
optimizer = Adam(learning_rate=learning_rate)
loss_fn = SparseCategoricalCrossentropy()

# Create model
model = Net()
optimizer = Adam()
loss_fn = SparseCategoricalCrossentropy()

# Training Function
@tf.function
def train_on_batch(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
        acc = tf.reduce_mean(sparse_categorical_accuracy(target, predictions))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, acc

# Evluation Function
def evaluate(generator):
    step = 0
    results = []
    for batch in generator:
        step += 1
        inputs, target = batch
        predictions = model(inputs, training=False)
        loss = loss_fn(target, predictions)
        acc = tf.reduce_mean(sparse_categorical_accuracy(target, predictions))
        results.append((loss, acc, len(target)))  # Keep track of batch size
        if step == generator.__len__():
            results = np.array(results)
            return np.average(results[:, :-1], 0, weights=results[:, -1])


# Setup Training
best_val_loss = 99999
current_patience = es_patience
step = 0
results_tr = []
## If you want to visualize training and validation loss/accuracy, uncomment below section
# l_tr, l_va = [], []
# acc_tr, acc_va = [], []

# Run the following code for 'epochs' number of times as in dataloader its already taken care of
# But we are using a generator
for epoch in range(epochs):
    for batch in training_batch_generator:
        step += 1
    # Training step
        inputs, target = batch
        loss, acc = train_on_batch(inputs, target)
        results_tr.append((loss, acc, len(target)))
        if step == training_batch_generator.__len__():
            results_va = evaluate(validation_batch_generator)
            if results_va[0] < best_val_loss:
                best_val_loss = results_va[0]
                current_patience = es_patience
            else:
                current_patience -= 1
                if current_patience == 0:
                    print("Early stopping")
                    break

        # Print results
            results_tr = np.array(results_tr)
            results_tr = np.average(results_tr[:, :-1], 0, weights=results_tr[:, -1])
            print('Epoch: {}'.format(epoch+1))
            print("Train loss: {:.4f}, acc: {:.4f} | " "Valid loss: {:.4f}, acc: {:.4f} | ".format(*results_tr, *results_va))
            ## Uncomment below section to visualize results
            #l_tr.append(results_tr[0])
            #l_va.append(results_va[0])
            #acc_tr.append(results_tr[1])
            #acc_va.append(results_va[1])
            # Reset epoch
            results_tr = []
            step = 0