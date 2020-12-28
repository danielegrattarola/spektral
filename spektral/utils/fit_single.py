import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
if tf.version.VERSION < "2.3":
    from tensorflow.python.keras.callbacks import CallbackList
else:
    CallbackList = callbacks.CallbackList


def _as_tensor(x):
    if isinstance(x, np.ndarray):
        x = tf.convert_to_tensor(x)
    return x


if tf.version.VERSION < "2.3":
    def _unpack_x_y_sample_weight(data):
        # tf.keras.utils.unpack_x_y_sample_weight in tf >= 2.3
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None
        return x, y, sample_weight

    def _get_logs(model, y, preds, sample_weight):
        loss = model.loss(y, preds, sample_weight)
        if model.losses:
            loss = tf.add_n([loss, *model.losses])
        logs = {'loss': loss}
        for m in model.metrics:
            m.update_state(y, preds, sample_weight)
            logs[m.name] = m.result()
        return logs

    def _build_train_step(model, data):
        data = tf.nest.map_structure(_as_tensor, data)

        @tf.function
        def train_fn():
            x, y, sample_weight = _unpack_x_y_sample_weight(data)
            with tf.GradientTape() as tape:
                preds = model(x, training=True)
                logs = _get_logs(model, y, preds, sample_weight)
            var_list = model.trainable_variables
            grads = tape.gradient(logs['loss'], var_list)

            model.optimizer.apply_gradients(zip(grads, var_list))
            return logs

        return train_fn

    def _build_test_step(model, data):
        data = tf.nest.map_structure(_as_tensor, data)

        @tf.function
        def test_fn():
            model.reset_metrics()
            x, y, sample_weight = _unpack_x_y_sample_weight(data)
            preds = model(x, training=False)
            return _get_logs(model, y, preds, sample_weight)

        return test_fn

else:
    def _build_train_step(model, data):
        data = tf.nest.map_structure(_as_tensor, data)
        @tf.function
        def train_fn():
            return model.train_step(data)

        return train_fn


    def _build_test_step(model, data):
        data = tf.nest.map_structure(_as_tensor, data)

        @tf.function
        def test_fn():
            model.reset_metrics()
            return model.test_step(data)

        return test_fn


class EpochProgbarLogger(callbacks.Callback):
    """Progress bar that updates at the end of each epoch."""
    def __init__(self):
        super().__init__()
        self.progbar = None
        self.epochs = None
        self.stateful_metrics = None
        self.last_seen = None

    def set_params(self, params):
        self.epochs = params["epochs"]
        self.stateful_metrics = params["metric_names"]

    def on_train_begin(self, logs=None):
        self.progbar = tf.keras.utils.Progbar(
            target=self.epochs,
            stateful_metrics=self.stateful_metrics,
            unit_name="epoch",
        )

    def on_epoch_end(self, epoch: int, logs=None):
        self.last_seen = epoch + 1
        self.progbar.update(epoch + 1, list(logs.items()))

    def on_train_end(self, logs=None):
        if self.last_seen < self.progbar.target:
            if tf.version.VERSION < "2.3":
                sys.stdout.write('\n')
            else:
                self.progbar.update(self.last_seen, finalize=True)


def fit_single(
        model,
        train_data,
        validation_data=None,
        epochs=1,
        initial_epoch=0,
        steps_per_epoch=1,
        callbacks=(),
        verbose=True,
    ):
    """
    Optimized keras.Model.fit for training on a single graph.
    :param model: keras model.
    :param train_data: (inputs, labels, sample_weight) or dataset with a
    single element for training.
    :param validation_data: (inputs, labels, sample_weight) or dataset with a
    single element for validation.
    :param epochs: int, maximum number of epochs to train for. One epoch is defined
    as `steps_per_epoch` training steps.
    :param initial_epoch: int, starting epoch.
    :steps_per_epoch: int, number of training steps defining an epoch.
    :callbacks: Iterable of tf.keras.callbacks.Callbacks.
    :verbose: flag resulting in verbose outputs.
    :return history: tf.keras.callbacks.History object.
    """
    if isinstance(train_data, tf.data.Dataset):
        train_data = tf.data.experimental.get_single_element(train_data)
    if isinstance(validation_data, tf.data.Dataset):
        validation_data = tf.data.experimental.get_single_element(validation_data)
    do_validation = validation_data is not None

    metric_names = ["loss"]
    for metrics in (model.compiled_metrics._metrics,
                   model.compiled_metrics._weighted_metrics):
        if metrics is not None:
            assert all(isinstance(
                m, (str, tf.keras.metrics.Metric)) for m in metrics)
            metric_names.extend(
                (m.name if isinstance(m, tf.keras.metrics.Metric) else m
                for m in metrics))
    if do_validation:
        metric_names = [*metric_names, *(f"val_{n}" for n in metric_names)]
    params = dict(
        epochs=epochs,
        verbose=verbose,
        steps=steps_per_epoch,
        metric_names=metric_names,
        do_validation=do_validation,
    )
    callbacks = list(callbacks)
    if verbose:
        callbacks.append(EpochProgbarLogger())
    if tf.version.VERSION < "2.3":
        history = tf.keras.callbacks.History()
        callbacks.append(history)
        cb = CallbackList(callbacks)
        cb.set_model(model)
        cb.set_params(params)
    else:
        cb = CallbackList(
            callbacks,
            add_history=True,
            add_progbar=False,
            model=model,
            **params,
        )
        history = model.history
    del callbacks
    train_step = _build_train_step(model, train_data)
    if validation_data is None:
        validation_step = None
    else:
        validation_step = _build_test_step(model, validation_data)

    model.stop_training = False
    cb.on_train_begin(logs=None)
    # _maybe_load_initial_epoch_from_ckpt behaviour is influenced by
    # callbacks.experimental.BackupAndRestore
    kwargs = {}
    if tf.version.VERSION < "2.3":
        kwargs['mode'] = 'train'
    initial_epoch = model._maybe_load_initial_epoch_from_ckpt(initial_epoch, **kwargs)

    logs = None
    for epoch in range(initial_epoch, epochs):
        model.reset_metrics()
        cb.on_epoch_begin(epoch, logs=None)
        for batch in range(steps_per_epoch):
            cb.on_train_batch_begin(batch)
            logs = train_step()
            cb.on_train_batch_end(batch, logs=logs)
            if model.stop_training:
                break
        # validation
        if validation_step is not None:
            val_logs = validation_step()
            logs.update({f"val_{k}": v for k, v in val_logs.items()})
        cb.on_epoch_end(epoch, logs)
        if model.stop_training:
            break
        # raise Exception()  # HACK
    cb.on_train_end(logs)
    return history
