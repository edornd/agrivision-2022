import json
import os.path as osp
from abc import abstractmethod
from typing import Callable

import mmcv
import numpy as np
import torch

from mmseg.datasets.custom import CustomDataset


def get_class_probs(data_root: str):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {k: v for k, v in sorted(overall_class_stats.items(), key=lambda item: item[1])}
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq, dim=-1)
    return list(overall_class_stats.keys()), freq.numpy()


class FixedBuffer:
    """Abstraction that holds a numpy array and uses it as circular buffer.
    """

    def __init__(self, num_classes: int, max_length: int = 128, reduction: Callable = np.sum):
        self.buffer = np.zeros((max_length, num_classes))
        self.num_classes = num_classes
        self.max_length = max_length
        self.reduction = reduction
        self.index = 0

    def append(self, data: np.ndarray):
        self.buffer[self.index] = data
        self.index = (self.index + 1) % self.max_length

    def get_counts(self):
        return self.reduction(self.buffer, axis=0)


class SamplingDataset(CustomDataset):

    def __init__(self, sampling: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.sampling_enabled = sampling is not None
        if self.sampling_enabled:
            self.min_crop_ratio = sampling['min_crop_ratio']
            self.min_pixels = sampling['min_pixels']
            # memory buffer holding a fixed amount of class-wise pixel counts
            buf_len = sampling["window_size"]
            self.count_buffer = FixedBuffer(num_classes=len(self.CLASSES), max_length=buf_len)
            self.conf_buffer = FixedBuffer(num_classes=len(self.CLASSES), max_length=buf_len, reduction=np.mean)

            self.sampling_classes, self.sampling_probs = get_class_probs(self.data_root)
            mmcv.print_log(f'Classes: {self.sampling_classes}', 'mmseg')
            mmcv.print_log(f'ClassProb: {self.sampling_probs}', 'mmseg')

            with open(osp.join(self.data_root, 'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items() if int(k) in self.sampling_classes
            }
            self.samples_with_class = {}
            for c in self.sampling_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}
            for i, dic in enumerate(self.img_infos):
                file = dic['ann']['seg_map']
                self.file_to_idx[file] = i

    def get_rare_class_sample(self):
        # select a class with lowest class count in the current window
        # the indexing and +1 exclude the background (no need to oversample that)
        windowed_counts = self.count_buffer.get_counts()
        # UNCOMMENT THE FOLLOWING LINE TO CHECK PIXELS COUNTS
        # mmcv.print_log(f'counts: {windowed_counts}', 'mmseg')
        c = np.argmin(windowed_counts[1:]) + 1
        f = np.random.choice(self.samples_with_class[c])
        idx = self.file_to_idx[f]
        sample = self.prepare_batch(idx)
        if self.min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(sample['gt_semantic_seg'].data == (c - 1))
                if n_class > self.min_pixels * self.min_crop_ratio:
                    break
                sample = self.prepare_batch(idx)
        return sample

    def update_statistics(self, class_counts: np.ndarray, class_confidence: np.ndarray):
        self.count_buffer.append(class_counts)
        self.conf_buffer.append(class_confidence)

    @abstractmethod
    def prepare_batch(self, idx: int):
        raise NotImplementedError()

    def prepare_train_img(self, idx: int):
        """Yet another wrapper to switch between RCS and standard random sampling.
        """
        if self.sampling_enabled:
            return self.get_rare_class_sample()
        return self.prepare_batch(idx)
