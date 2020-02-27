from keras.callbacks import Callback
from tensorflow.keras import backend as K

class WarmupLearningRateScheduler(Callback):
    def __init__(self, lr, warmup_steps, steps_per_epoch):
        super(WarmupLearningRateScheduler, self).__init__()
        self.target_lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = warmup_steps

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_begin(self, batch, logs=None):
        step_num = self.epoch * self.steps_per_epoch + batch + 1

        if step_num <= self.warmup_steps:
            lr = (step_num / self.warmup_steps) * lr
            K.set_value(self.model.optimizer.lr, lr)
            print('\nStep %05d: WarmupLearningRateScheduler setting learning rate to %s.' % (step_num, lr))

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)