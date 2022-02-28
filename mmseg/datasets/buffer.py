import ctypes
import multiprocessing as mp
from typing import Callable

import numpy as np


class Counter(object):

    def __init__(self, length: int):
        self.val = mp.Value('i', 0)
        self.length = length

    def increment(self):
        with self.val.get_lock():
            self.val.value = (self.val.value + 1) % self.length

    @property
    def value(self):
        return self.val.value


class FixedBuffer:
    """Abstraction that holds a numpy array and uses it as circular buffer.
    """

    def __init__(self, num_classes: int, max_length: int = 128, reduction: Callable = np.sum):
        self.shared_base = mp.Array(ctypes.c_float, max_length * num_classes)
        self.shared_array = np.ctypeslib.as_array(self.shared_base.get_obj()).reshape(max_length, num_classes)
        self.indexer = Counter(length=max_length)
        self.num_classes = num_classes
        self.max_length = max_length
        self.reduction = reduction
        self.index = 0

    def append(self, data: np.ndarray):
        with self.shared_base.get_lock():  # synchronize access
            arr = np.ctypeslib.as_array(self.shared_base.get_obj()).reshape(self.max_length,
                                                                            self.num_classes)  # no data copying
            arr[self.indexer.value] = data
            self.indexer.increment()

    def get_counts(self):
        with self.shared_base.get_lock():  # synchronize access
            arr = np.ctypeslib.as_array(self.shared_base.get_obj()).reshape(self.max_length,
                                                                            self.num_classes)  # no data copying
            avg = self.reduction(arr, axis=0)
        return avg
