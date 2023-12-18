import numpy as np


class Sampler:
    def __init__(self, data: np.ndarray, label: np.ndarray, batch_size=32, shuffle=False):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.index = np.arange(label.shape[0])
        if shuffle:
            np.random.shuffle(self.index)

        self.current_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx < self.label.shape[0]:
            start = self.current_idx
            self.current_idx += self.batch_size
            index = self.index[start:min(self.current_idx, self.label.shape[0])]
            return (self.data[index], self.label[index])
        else:
            raise StopIteration
