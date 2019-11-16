import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time

import pandas as pd
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from spektral.datasets import citation, graphsage
from spektral.layers import GraphConv, GraphConvSkip, GraphSageConv, ARMAConv, GraphAttention, APPNP, GINConv
from spektral.utils.convolution import localpooling_filter, rescale_laplacian, normalized_laplacian

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--neighbourhood', type=int, default=1)
args = parser.parse_args()

dataset = args.dataset
neighbourhood = args.neighbourhood

dropout_rate = 0.5      # Dropout rate applied to the input of GCN layers
l2_reg = 5e-4           # Regularization rate for l2
learning_rate = 5e-3    # Learning rate for SGD
epochs = 20000          # Number of training epochs
es_patience = 50        # Patience for early stopping
runs = 10

base_kwargs = {
    'channels': 32,
    'activation': 'relu',
    'kernel_regularizer': l2(l2_reg),
}

CONFIG = [
    {
        'layer': GraphConv,
        'n_layers': neighbourhood,
        'kwargs': {},
        'fltr': lambda A: localpooling_filter(A),
        'sparse': True
    },
    {
        'layer': GraphConvSkip,
        'n_layers': neighbourhood,
        'kwargs': {},
        'fltr': lambda A: localpooling_filter(A),
        'sparse': True
    },
    {
        'layer': ARMAConv,
        'n_layers': 1,
        'kwargs': {
            'T': neighbourhood,
            'K': 1,
            'recurrent': True,
            'dropout_rate': dropout_rate
        },
        'fltr': lambda A: rescale_laplacian(normalized_laplacian(A), lmax=2),
        'sparse': True
    },
    {
        'layer': GraphAttention,
        'n_layers': neighbourhood,
        'kwargs': {
            'dropout_rate': dropout_rate
        },
        'fltr': lambda A: A,
        'sparse': False
    },
    {
        'layer': GraphSageConv,
        'n_layers': neighbourhood,
        'kwargs': {},
        'fltr': lambda A: A,
        'sparse': True
    },
    {
        'layer': APPNP,
        'n_layers': 1,
        'kwargs': {
            'mlp_channels': 32,
            'alpha': 0.1,
            'H': 1,
            'K': neighbourhood,
            'dropout_rate': dropout_rate
        },
        'fltr': lambda A: localpooling_filter(A),
        'sparse': True
    },
{
        'layer': GINConv,
        'n_layers': neighbourhood,
        'kwargs': {
            'mlp_channels': 32
        },
        'fltr': lambda A: A,
        'sparse': True
    }
]

results = {}
weights = []
for c in CONFIG:
    acc = []
    times = []
    for i in range(runs):
        if dataset is 'ppi':
            A, X, y, train_mask, val_mask, test_mask = graphsage.load_data(
                dataset_name=dataset
            )
        else:
            A, X, y, train_mask, val_mask, test_mask = citation.load_data(
                dataset, random_split=True
            )

        # Parameters
        N = X.shape[0]          # Number of nodes in the graph
        F = X.shape[1]          # Original feature dimensionality
        n_classes = y.shape[1]  # Number of classes

        # Preprocessing operations
        fltr = c['fltr'](A)

        # Model definition
        X_in = Input(shape=(F, ))
        fltr_in = Input((N, ), sparse=c['sparse'])

        gc_1 = Dropout(dropout_rate)(X_in)
        for _ in range(c['n_layers']):
            gc_1 = c['layer'](**dict(base_kwargs, **c['kwargs']))([gc_1, fltr_in])
        gc_2 = Dropout(dropout_rate)(gc_1)
        gc_2 = GraphConv(n_classes, activation='softmax')([gc_2, fltr_in])

        # Build model
        model = Model(inputs=[X_in, fltr_in], outputs=gc_2)
        optimizer = Adam(lr=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      weighted_metrics=['acc'])
        if i == 0:
            weights.append((
                c['layer'].__name__, sum([i.size for i in model.get_weights()])
            ))

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_weighted_acc',
                          patience=es_patience,
                          restore_best_weights=True),
        ]

        # Train model
        validation_data = ([X, fltr], y, val_mask)
        timer = -time.time()
        model.fit([X, fltr],
                  y,
                  sample_weight=train_mask,
                  epochs=epochs,
                  batch_size=N,
                  validation_data=validation_data,
                  shuffle=False,
                  callbacks=callbacks,
                  verbose=0)
        timer += time.time()
        times.append(timer)

        # Evaluate model
        eval_results = model.evaluate([X, fltr],
                                      y,
                                      sample_weight=test_mask,
                                      batch_size=N,
                                      verbose=0)
        acc.append(eval_results[1])
        print('{} - Test loss: {}, Test accuracy: {}'
              .format(c['layer'].__name__, *eval_results))
        K.clear_session()

    results[c['layer'].__name__ + ' acc'] = acc
    results[c['layer'].__name__ + ' time'] = times
    pd.DataFrame(results).to_csv('{}_{}_results.csv'.format(dataset, neighbourhood), index=False)

pd.DataFrame(weights).to_csv('{}_{}_weights.csv'.format(dataset, neighbourhood), index=False)
print(weights)
