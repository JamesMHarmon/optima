from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

class WarmupLearningRateScheduler(Callback):
    def __init__(self, lr, warmup_steps):
        super(WarmupLearningRateScheduler, self).__init__()
        self.target_lr = lr
        self.warmup_steps = warmup_steps

    def on_train_batch_begin(self, step_num, logs=None):
        if step_num <= self.warmup_steps:
            lr = (step_num / self.warmup_steps) * self.target_lr
            K.set_value(self.model.optimizer.lr, lr)
            print('\nStep %05d: WarmupLearningRateScheduler setting learning rate to %s.' % (step_num, lr))
        elif step_num == self.warmup_steps:
            K.set_value(self.model.optimizer.lr, self.target_lr)
            print('\nStep %05d: WarmupLearningRateScheduler setting learning rate to %s.' % (step_num, lr))

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)