from tensorflow.keras.utils import Sequence
import numpy as np
import math
import sys
import time
import tensorflow as tf

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, *, replay_buffer, epoch, batch_size=2048, games_per_epoch=32_000, window_size=500_000, avg_num_samples_per_pos=0.5, window_warmup=0.5):
        self._replay_buffer = replay_buffer 
        self._epoch = int(epoch)
        self._batch_size = int(batch_size)
        self._games_per_epoch = int(games_per_epoch)
        self._window_size = int(window_size)
        self._avg_num_samples_per_pos = float(avg_num_samples_per_pos)
        self._window_warmup = float(window_warmup)
        self._num_games = 0

        avg_num_samples_per_game = replay_buffer.avg_num_samples_per_game(look_back=games_per_epoch)
        if avg_num_samples_per_game < 1:
            avg_num_samples_per_game = 10

        self._steps = (games_per_epoch * avg_num_samples_per_game * avg_num_samples_per_pos) // batch_size 


    def __len__(self):
        return self._steps

    def __getitem__(self, index):
        adj_window_size = min(self._window_size, self._epoch * self._games_per_epoch * self._window_warmup)
        end_idx = math.floor(((self._epoch - 1) * self._games_per_epoch) + ((index + 1) / self._steps) * self._games_per_epoch)
        start_idx = max(0, end_idx - adj_window_size)

        if self._num_games <= end_idx:
            while True:
                self._num_games = self._replay_buffer.games()

                if not self._num_games <= end_idx:
                    break

                print('Current number of games is {:,}, waiting for game {:,}.'.format(self._num_games, end_idx + 1))
                sys.stdout.flush()
                time.sleep(30)

        sample = self._replay_buffer.sample(self._batch_size, start_idx=start_idx, end_idx=end_idx)

        return (
            tf.convert_to_tensor(sample['X'], dtype=tf.float32),
            {
                'policy_head': tf.convert_to_tensor(sample['yp'], dtype=tf.float32),
                'value_head': tf.convert_to_tensor(sample['yv'], dtype=tf.float32),
                'moves_left_head': tf.convert_to_tensor(sample['ym'], dtype=tf.float32)
            }
        )
