import logging as log
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class FitLogger(Callback):
    def __init__(self, log_steps=10):
        super().__init__()
        self._log_steps = log_steps

    def on_epoch_begin(self, epoch, logs={}):
        log.info(f'Starting epoch {epoch}')
        sys.stdout.flush()

    def on_train_batch_end(self, step, logs={}):
        if step % self._log_steps == 0:
            log.info('Step: {:,}, LR: {:.2f}, VL: {:2f}, PL: {:2f}, ML: {:2f}'.format(
                step,
                float(tf.keras.backend.get_value(self.model.optimizer.lr)),
                logs.get('loss/value_head loss', float('nan')),
                logs.get('loss/policy_head loss', float('nan')),
                logs.get('loss/moves_left_head loss', float('nan'))))

            sys.stdout.flush()

    def on_epoch_end(self, epoch, logs={}):
        log.info(f'Epoch {epoch} completed')

