import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from get_gradient_norm import get_gradient_norm, get_gradient_norm_func

class TensorBoardEnriched(keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__() 
        self.writer = tf.summary.FileWriter(logdir=log_dir)

    def on_train_begin(self, logs=None):
        self._gradient_norm_func = get_gradient_norm_func(self.model)

    def on_epoch_begin(self, epoch, logs=None):
        self._grad_norms = []

    def on_epoch_end(self, epoch, logs={}):
        self._write_logs(logs, epoch)
        self._write_learning_rate(epoch)
        self._write_reg_term(epoch)
        self._write_gradient_norm(epoch)
        
        self.writer.flush()

    def on_batch_begin(self, step, logs=None):
        if step % 10 == 0:
            num_outputs = (len(self.validation_data) - 2) // 2
            batch_size = self.params['batch_size']
            from_idx = batch_size * (step // 10)
            to_idx = batch_size * (step // 10 + 1)
            X_test = self.validation_data[0][from_idx:to_idx]
            y_tests = self.validation_data[1:num_outputs + 1]
            y_test = [data[from_idx:to_idx] for data in y_tests]

            if len(X_test) == batch_size:
                grad_norm = get_gradient_norm(self.model, self._gradient_norm_func, X_test, y_test)
                self._grad_norms.append(grad_norm)
        
    def _write_logs(self, logs, epoch):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
    
    def _write_learning_rate(self, epoch):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        self._write_value("Learning Rate", epoch=epoch, val=lr)
        
    def _write_reg_term(self, epoch):
        reg_term_op = tf.reduce_sum(self.model.losses)
        reg_term_val = K.get_session().run(reg_term_op)
        self._write_value("Regularization Term", epoch=epoch, val=reg_term_val)
        
    def _write_gradient_norm(self, epoch):
        print("Number of Norms: ", len(self._grad_norms))
        if len(self._grad_norms) > 0:
            grad_norm_avg = sum(self._grad_norms) / len(self._grad_norms)
            grad_norm_max = max(self._grad_norms)
            self._write_value("Gradient Norm Average", epoch=epoch, val=grad_norm_avg)
            self._write_value("Gradient Norm Max", epoch=epoch, val=grad_norm_max)
        
    def _write_value(self, tag, epoch, val):
        print(tag + ':', val)
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
        self.writer.add_summary(summary, epoch)
    
    def on_train_end(self, _):
        self.writer.close()