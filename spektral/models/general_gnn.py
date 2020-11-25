from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Concatenate, Add, BatchNormalization, Dropout, Activation
from tensorflow.keras.layers import PReLU

from spektral.data import DisjointLoader
from spektral.datasets import TUDataset
from spektral.layers import GeneralConv
from spektral.layers.pooling import global_pool


def get_act(identifier):
    if identifier == 'prelu':
        return PReLU()
    else:
        return Activation(identifier)


class GeneralGNN(Model):
    r"""
    This model implements the GNN architecture described in

    > [Design Space for Graph Neural Networks](https://arxiv.org/abs/2011.08843)<br>
    > Jiaxuan You, Rex Ying, Jure Leskovec<br>
    > NeurIPS 2020

    The default parameters of the model are selected according to the best
    results obtained in the paper, and should provide a good performance on
    many node-level and graph-level tasks, without modifications.
    The defaults are as follows:

    - 256 hidden channels
    - 4 message passing layers
    - 2 pre-processing layers
    - 2 post-processing layers
    - Skip connections with concatenation
    - Batch normalization
    - No dropout
    - PReLU activations
    - Sum aggregation in the message-passing layers
    - Global sum pooling (not from the paper)

    The GNN uses the [`GeneralConv` layer](/layers/convolution/#generalconv)
    for message passing, and has a pre- and a post-processing MLP for the node
    features.
    Message-passing layers also have optional skip connections, which can be
    implemented as sum or concatenation.

    The dense layers of the pre-processing and post-processing MLPs compute the
    following update of the node features:

    $$
        \h_i = \mathrm{Act} \left( \mathrm{Dropout} \left( \mathrm{BN}
        \left( \x_i \W + \b \right) \right) \right)
    $$

    Message-passing layers compute:

    $$
        \h_i = \mathrm{Agg} \left( \left\{ \mathrm{Act} \left( \mathrm{Dropout}
        \left( \mathrm{BN} \left( \x_j \W + \b \right) \right) \right),
        j \in \mathcal{N}(i) \right\} \right)
    $$

    **Arguments**

    - `output`: int, the number of output units;
    - `activation`: the activation function of the output layer.
    - `hidden`: int, the number of hidden units for all layers except the output
    one;
    - `message_passing`: int, the nummber of message-passing layers;
    - `pre_process`: int, the number of layers in the pre-processing MLP;
    - `post_process`: int, the number of layers in the post-processing MLP;
    - `connectivity`: the type of skip connection. Can be: None, 'sum' or 'cat';
    - `batch_norm`: bool, whether to use batch normalization;
    - `dropout`: float, dropout rate;
    - `aggregate`: string or callable, an aggregation function. Supported
    aggregations: 'sum', 'mean', 'max', 'min', 'prod'.
    - `hidden_activation`: activation function in the hidden layers. The PReLU
    activation can be used by passing `hidden_activation='prelu'`.
    - `pool`: string or None, the global pooling function. If None, no global
    pooling is applied (e.g., for node-level learning). Supported pooling methods:
    'sum', 'avg', 'max', 'attn', 'attn_sum', 'sort'
    (see `spektral.layers.pooling.global_pool`).
    """
    def __init__(self, output, activation=None, hidden=256, message_passing=4,
                 pre_process=2, post_process=2, connectivity='cat',
                 batch_norm=True, dropout=0.0, aggregate='sum',
                 hidden_activation='prelu', pool='sum'):
        super().__init__()

        # Connectivity function
        if connectivity is None:
            self.connectivity = None
        elif connectivity == 'sum':
            self.connectivity = Add()
        elif connectivity == 'cat':
            self.connectivity = Concatenate()
        else:
            raise ValueError('Unknown connectivity: {}. Available: None, sum, cat.')

        # Global pooling
        if pool is not None:
            self.pool = global_pool.get(pool)()
        else:
            self.pool = None

        # Neural blocks
        self.pre = MLP(hidden, hidden, pre_process, batch_norm, dropout,
                       hidden_activation, hidden_activation)
        self.gnn = [GeneralConv(hidden, batch_norm, dropout, aggregate, hidden_activation)
                    for _ in range(message_passing)]
        self.post = MLP(output, hidden, post_process, batch_norm, dropout, hidden_activation, activation)

    def call(self, inputs):
        if len(inputs) == 2:
            x, a = inputs
            i = None
        else:
            x, a, i = inputs

        # Pre-process
        out = self.pre(x)
        # Message passing
        for layer in self.gnn:
            z = layer([out, a])
            if self.connectivity is not None:
                out = self.connectivity([z, out])
            else:
                out = z
        # Global pooling
        if self.pool is not None:
            out = self.pool([out] + [i] if i is not None else [])
        # Post-process
        out = self.post(out)

        return out


class MLP(Model):
    def __init__(self, output, hidden=256, layers=2, batch_norm=True,
                 dropout=0.0, activation='prelu', final_activation=None):
        super().__init__()
        self.batch_norm = batch_norm
        self.dropout_rate = dropout

        self.mlp = Sequential()
        for i in range(layers):
            # Linear
            self.mlp.add(Dense(hidden if i < layers - 1 else output))
            # Batch norm
            if self.batch_norm:
                self.mlp.add(BatchNormalization())
            # Dropout
            self.mlp.add(Dropout(self.dropout_rate))
            # Activation
            self.mlp.add(get_act(activation if i < layers - 1 else final_activation))

    def call(self, inputs):
        return self.mlp(inputs)


if __name__ == '__main__':
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.optimizers import Adam
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Best config
    batch_size = 32
    learning_rate = 0.01
    epochs = 400

    # Read data
    data = TUDataset('PROTEINS')

    # Train/test split
    np.random.shuffle(data)
    split = int(0.8 * len(data))
    data_tr, data_te = data[:split], data[split:]

    # Data loader
    loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
    loader_te = DisjointLoader(data_te, batch_size=batch_size)

    # Create model
    model = GeneralGNN(data.n_labels, activation='softmax')
    optimizer = Adam(learning_rate)
    model.compile('adam', 'categorical_crossentropy', metrics=['categorical_accuracy'])

    # Evaluation function
    def eval(loader):
        step = 0
        results = []
        for batch in loader:
            step += 1
            l, a = model.test_on_batch(*batch)
            results.append((l, a))
            if step == loader.steps_per_epoch:
                return np.mean(results, 0)

    # Training loop
    epoch = step = 0
    results = []
    for batch in loader_tr:
        step += 1
        l, a = model.train_on_batch(*batch)
        results.append((l, a))
        if step == loader_tr.steps_per_epoch:
            step = 0
            epoch += 1
            results_te = eval(loader_te)
            print('Epoch {} - Train loss: {:.3f} - Train acc: {:.3f} - '
                  'Test loss: {:.3f} - Test acc: {:.3f}'
                  .format(epoch, *np.mean(results, 0), *results_te))

    results_te = eval(loader_te)
    print('Final results - Loss: {:.3f} - Acc: {:.3f}'.format(*results_te))
