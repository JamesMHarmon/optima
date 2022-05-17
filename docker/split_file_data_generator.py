from tensorflow.keras.utils import Sequence
import numpy as np

class SplitFileDataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, **kwargs):
        self._batch_size = int(kwargs['batch_size'])
        self._files = kwargs['files']
        self._total_samples = int(kwargs['total_samples'])
        self._input_size = kwargs['input_size']
        self._output_size = kwargs['output_size']
        self._moves_left_size = kwargs['moves_left_size']
        self._value_size = kwargs['value_size']
        self._input_h = kwargs['input_h']
        self._input_w = kwargs['input_w']
        self._input_c = kwargs['input_c']
        self._current_file_idx = -1
        self._current_length = 0
        self._starting_file_idx = 0
        
        assert self._total_samples % self._batch_size == 0, f"Number of entries should be divisible by batch size, got: {self._total_samples}"
        
    def __len__(self):
        return int(self._total_samples / self._batch_size)
        
    def __getitem__(self, index):            
        start = index * self._batch_size
        end = start + self._batch_size
        
        assert end <= self._total_samples, f"Out of bounds. Requested index {index} of {self._batch_size} batch_size with total length of {self._total_samples}."
                
        if self._current_file_idx == -1 or self._starting_file_idx + self._current_length < end:
            self._starting_file_idx += self._current_length
            self._current_file_idx += 1
            self._load_file(self._current_file_idx)
            self._current_length = self._data['X'].shape[0]
            
            assert self._current_length % self._batch_size == 0, f"Number of entries in file should be divisible by batch size, got: {self._current_length}"

        start -= self._starting_file_idx
        end -= self._starting_file_idx
        
        return {
            'X': self._data['X'][start:end],
            'policy_head': self._data['yp'][start:end],
            'value_head': self._data['yv'][start:end],
            'moves_left_head': self._data['ym'][start:end]
        }
        
    def _load_file(self, file_idx):
        self._data = None

        input_size, output_size, moves_left_size, value_size = self._input_size, self._output_size, self._moves_left_size, self._value_size
        input_h, input_w, input_c = self._input_h, self._input_w, self._input_c
        path = self._files[file_idx]
        
        print('Loading file: ', path)

        dataset = np.load(path).reshape(-1, input_size + output_size + moves_left_size + value_size)
        X = dataset[:,0:input_size].reshape(dataset.shape[0],input_h,input_w,input_c)
        start_index = input_size
        yp = dataset[:,start_index:start_index + output_size]
        start_index += output_size
        yv = dataset[:,start_index]
        start_index += value_size
        ym = dataset[:,start_index:]

        self._data = { 'X': X, 'yp': yp, 'yv': yv, 'ym': ym }