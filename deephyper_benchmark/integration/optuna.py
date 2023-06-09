import warnings

import optuna
import tensorflow as tf


class KerasPruningCallback(tf.keras.callbacks.Callback):
    """Keras callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/main/
    keras/keras_integration.py>`__
    if you want to add a pruning callback which observes validation accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` and
            ``val_accuracy``. Please refer to `keras.Callback reference
            <https://keras.io/callbacks/#callback>`_ for further details.
        interval:
            Check if trial should be pruned every n-th epoch. By default ``interval=1`` and
            pruning is performed after every epoch. Increase ``interval`` to run several
            epochs faster before applying pruning.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str, interval: int = 1) -> None:
        super().__init__()

        self._trial = trial
        self._monitor = monitor
        self._interval = interval

    def on_epoch_end(self, epoch: int, logs = None) -> None:
        self.epoch_stopped = epoch
        if (epoch + 1) % self._interval != 0:
            return

        logs = logs or {}
        current_score = logs.get(self._monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self._monitor)
            )
            warnings.warn(message)
            return
        self._trial.report(float(current_score), step=epoch)
        if self._trial.should_prune():
            self.message = "Trial was pruned at epoch {}.".format(epoch)
            self.model.stop_training = True