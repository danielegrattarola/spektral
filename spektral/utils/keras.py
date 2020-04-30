from tensorflow.keras import initializers, regularizers, constraints, activations

LAYER_KWARGS = {'activation', 'use_bias'}
KERAS_KWARGS = {'activity_regularizer', 'autocast', 'batch_input_shape',
                'batch_size', 'input_shape', 'weights'}


def is_layer_kwarg(key):
    return (key.endswith('_initializer')
            or key.endswith('_regularizer')
            or key.endswith('_constraint')
            or key in LAYER_KWARGS) and not key in KERAS_KWARGS


def is_keras_kwarg(key):
    return key in KERAS_KWARGS


def deserialize_kwarg(key, attr):
    if key.endswith('_initializer'):
        return initializers.get(attr)
    if key.endswith('_regularizer'):
        return regularizers.get(attr)
    if key.endswith('_constraint'):
        return constraints.get(attr)
    if key == 'activation':
        return activations.get(attr)


def serialize_kwarg(key, attr):
    if key.endswith('_initializer'):
        return initializers.serialize(attr)
    if key.endswith('_regularizer'):
        return regularizers.serialize(attr)
    if key.endswith('_constraint'):
        return constraints.serialize(attr)
    if key == 'activation':
        return activations.serialize(attr)
    if key == 'use_bias':
        return attr