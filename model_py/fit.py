import math
import tensorflow as tf
import sys

class Fit:
    def __init__(self, **kwargs):
        self._model=kwargs['model']
        self._generator=kwargs['data_generator']
        self._batch_size=kwargs['batch_size']
        self._initial_epoch=kwargs['initial_epoch']
        self._initial_step=kwargs['initial_step']
        self._accuracy_metrics=kwargs['accuracy_metrics']
        self._train_size=kwargs['train_size']
        self._clip_norm=kwargs['clip_norm']
        
        callbacks = kwargs['callbacks']
        callbacks = tf.keras.callbacks.CallbackList(callbacks=callbacks, model=self._model)
        self._callbacks=callbacks
        
    def fit(self):
        callbacks = self._callbacks
        epoch = self._initial_epoch

        num_steps = self._generator.__len__()
        num_train_steps = math.floor(num_steps * self._train_size)
        num_val_steps = num_steps - num_train_steps
        starting_step = self._initial_step
        ending_step = self._initial_step + num_train_steps - 1
        starting_val_step = ending_step - num_val_steps + 1

        epoch_logs = self._get_epoch_logs()
        callbacks.on_epoch_begin(epoch, epoch_logs)
        callbacks.on_train_begin()

        for step_num in range(starting_step, ending_step + 1):
            data = self._generator.__getitem__(step_num - starting_step)
            callbacks.on_train_batch_begin(step_num)
            batch_logs = self._step(data, training=True)
            callbacks.on_train_batch_end(step_num, batch_logs)

        callbacks.on_train_end({})
        callbacks.on_test_begin()

        for step_num in range(starting_val_step, ending_step + 1):
            # The entry that we get from the generator is higher than the number of steps
            # This is because we want the steps to align with the end of the training steps to align logs
            data = self._generator.__getitem__(step_num + num_val_steps - starting_step)
            callbacks.on_test_batch_begin(step_num)
            batch_logs = self._step(data, training=False)
            callbacks.on_test_batch_end(step_num, batch_logs)

        callbacks.on_test_end({})

        epoch_logs = self._get_epoch_logs()
        callbacks.on_epoch_end(epoch, epoch_logs)

        return num_steps
    
    @tf.function
    def _step(self, data, **kwargs):
        training = kwargs['training']
        model = self._model
        logs = {}

        with tf.GradientTape() as tape:
            predictions = model(data['X'], training=training)

            total_loss = sum(model.losses)
            logs['loss/model loss'] = total_loss

            for name, prediction in zip(model.output_names, predictions):
                loss_fn = model.loss[name]
                target = data[name]
                weight = model.compiled_loss._loss_weights[name]

                loss = tf.reduce_mean(loss_fn(target, prediction))
                logs[f'loss/{name} loss'] = loss

                total_loss += loss * weight

                if name in self._accuracy_metrics:
                    metric_fn = self._accuracy_metrics[name]
                    logs[f'accuracy/{name}'] = tf.reduce_mean(metric_fn(target, prediction))
                    
            for loss_layer in model.losses:
                logs[f'loss_layer/{loss_layer.name}'] = loss_layer
        
        logs['loss/total loss'] = tf.reduce_mean(total_loss)

        if training:
            gradients = tape.gradient(total_loss, model.trainable_variables)

            clipped_grads, global_norm = tf.clip_by_global_norm(gradients, self._clip_norm)
            clipped_global_norm = tf.linalg.global_norm(clipped_grads)

            logs['norm/global norm clipped'] = clipped_global_norm
            logs['norm/global norm'] = global_norm

            model.optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
            
        return logs

    def _get_epoch_logs(self):
        logs = {
            'info/num steps': self._generator.__len__(),
            'info/epoch': self._initial_epoch,
            'info/batch_size': self._batch_size,
            'norm/clip_norm_value': self._clip_norm
        }
        
        for name in self._model.output_names:
            logs[f'info/weight/{name}'] = self._model.compiled_loss._loss_weights[name]

        return logs