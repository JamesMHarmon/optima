from tensorflow.keras.utils import Sequence
import logging as log
import numpy as np
import math
import sys
import time
import tensorflow as tf

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, *, replay_buffer, epoch, batch_size=2048, games_per_epoch=32_000, window_size=500_000, avg_num_samples_per_pos=0.5, window_warmup=0.5, shapes=None):
        self._replay_buffer = replay_buffer 
        self._epoch = int(epoch)
        self._batch_size = int(batch_size)
        self._games_per_epoch = int(games_per_epoch)
        self._window_size = int(window_size)
        self._avg_num_samples_per_pos = float(avg_num_samples_per_pos)
        self._window_warmup = float(window_warmup)
        self._num_games = 0
        self._shape = shapes

        avg_num_samples_per_game = replay_buffer.avg_num_samples_per_game(look_back=games_per_epoch)
        if avg_num_samples_per_game < 1:
            avg_num_samples_per_game = 10

        self._steps = int(games_per_epoch * avg_num_samples_per_game * avg_num_samples_per_pos) // batch_size 

    def to_tensor(self, sample, shape):
        shape = (self._batch_size, ) + shape
        sample = sample.reshape(shape)
        return tf.convert_to_tensor(sample, dtype=tf.float32)
        

    def __len__(self):
        return self._steps

    def __getitem__(self, index):
        end_idx = int(math.floor(((self._epoch - 1) * self._games_per_epoch) + ((index + 1) / self._steps) * self._games_per_epoch))
        start_idx = int(max(0, end_idx - self._curr_epochs_window_size()))

        if self._num_games <= end_idx:
            while True:
                start = time.time()
                self._num_games = self._replay_buffer.games()
                end = time.time()
                log.info('Indexing of games in {:.2f}'.format(end - start))

                if not self._num_games <= end_idx:
                    break

                log.info('Current number of games is {:,}, waiting for game {:,}.'.format(self._num_games, end_idx + 1))
                sys.stdout.flush()
                time.sleep(60)

        sample = self._replay_buffer.sample(self._batch_size, start_idx=start_idx, end_idx=end_idx)

        input = self.to_tensor(sample['inputs'], self._shape['inputs'])
        targets = {
            'policy_head': self.to_tensor(sample['policy'], self._shape['policy']),
            'value_head': self.to_tensor(sample['value'], self._shape['value'])
        }

        if 'moves_left' in self._shape:
            targets['moves_left_head'] = self.to_tensor(sample['moves_left'], self._shape['moves_left'])

        if 'victory_margin' in self._shape:
            targets['victory_margin_head'] = self.to_tensor(sample['victory_margin'], self._shape['victory_margin'])

        return input, targets

    def _curr_epochs_window_size(self):
        if self._epoch == 1:
            self._games_per_epoch

        return int(min(self._window_size, self._epoch * self._games_per_epoch * self._window_warmup))
