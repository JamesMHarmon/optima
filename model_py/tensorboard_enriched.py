import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from metric import Metrics

class TensorBoardEnriched(Callback):
    def __init__(self, log_dir, step_ratio=1.0):
        super().__init__()
        self._log_dir = log_dir
        self._step_ratio = step_ratio
        self._step = None
        self._log_every_n_steps = 10
        self._epoch_begin_logs = None
        self._metrics = Metrics()

    def on_epoch_begin(self, epoch, logs={}):
        self.writer = tf.summary.create_file_writer(logdir=self._log_dir)
        self._epoch_begin_logs = logs

    def on_train_batch_begin(self, step, logs={}):
        self._step = step
        
        if self._epoch_begin_logs is not None:
            self._write_logs(self._epoch_begin_logs)
            self._epoch_begin_logs = None
        
        self._write_logs(logs)

    def on_train_batch_end(self, step, logs={}):
        metrics = self._metrics

        for name, value in logs.items():
            metrics.update(name, value)

        if step % self._log_every_n_steps == 0:
            metric_logs = metrics.aggregate()
            self._write_logs(metric_logs)
            self._write_learning_rate()
            metrics.reset()

    def on_train_end(self, logs=None):
        self._write_learning_rate()
        
    def on_epoch_end(self, epoch, logs={}):
        self._write_logs(logs)
        self._write_weights()
        self.writer.flush()
        self.writer.close()
        
    def _write_logs(self, logs):
        for name, value in logs.items():
            self._write_value(name, value)
    
    def _write_learning_rate(self):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        self._write_value("info/learning rate", val=lr)
        
    def _write_weights(self):
        with self.writer.as_default(step=int(self._step * self._step_ratio)):
            for w in self.model.weights:
                tf.summary.histogram(w.name, w)

    def _write_value(self, tag, val):
        with self.writer.as_default(step=int(self._step * self._step_ratio)):
            tf.summary.scalar(tag, data=val)
            self.writer.flush()