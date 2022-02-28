import json
import os.path as osp
from abc import abstractmethod

import mmcv
import numpy as np
import torch

from mmseg.datasets.buffer import FixedBuffer
from mmseg.datasets.custom import CustomDataset


def get_class_weights(data_root: str, temperature: float):
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
    overall_class_stats = {k: v for k, v in sorted(overall_class_stats.items(), key=lambda item: item[0])}
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = torch.softmax((1 - freq) / temperature, dim=-1)
    return list(overall_class_stats.keys()), freq.numpy()


def softmax(x: np.ndarray):
    """Quick softmax function for numpy arrays, instead of converting to torch
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class SamplingDataset(CustomDataset):

    def __init__(self, sampling: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.sampling_enabled = sampling is not None
        if self.sampling_enabled:
            self.min_crop_ratio = sampling['min_crop_ratio']
            self.min_pixels = sampling['min_pixels']
            # memory buffer holding a fixed amount of class-wise pixel counts
            buf_len = sampling["window_size"]
            self.conf_buffer = FixedBuffer(num_classes=len(self.CLASSES), max_length=buf_len, reduction=np.mean)

            self.class_list, self.class_weights = get_class_weights(self.data_root, sampling["temp"])
            mmcv.print_log(f'Classes            : {self.class_list}', 'mmseg')
            mmcv.print_log(f'Normalized weights.: {self.class_weights}', 'mmseg')

            with open(osp.join(self.data_root, 'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items() if int(k) in self.class_list
            }
            self.samples_with_class = {}
            for c in self.class_list:
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
        average_class_confidence = self.conf_buffer.get_counts()
        weighted_class_confidence = self.class_weights * (1 - average_class_confidence)
        weighted_class_confidence = softmax(weighted_class_confidence)

        # UNCOMMENT THE FOLLOWING LINE(S) TO CHECK
        c = np.random.choice(self.class_list, p=weighted_class_confidence)
        # mmcv.print_log(f'weights:      {weighted_class_confidence}', 'mmseg')
        # mmcv.print_log(f'class chosen: {c}', 'mmseg')

        f = np.random.choice(self.samples_with_class[c])
        idx = self.file_to_idx[f]
        sample = self.prepare_batch(idx)
        if self.min_crop_ratio > 0:
            for _ in range(10):
                n_class = torch.sum(sample['gt_semantic_seg'].data == (c - 1))
                if n_class > self.min_pixels * self.min_crop_ratio:
                    break
                sample = self.prepare_batch(idx)
        return sample

    def update_statistics(self, class_confidence: np.ndarray):
        # mmcv.print_log(f'batch:   {class_confidence}', 'mmseg')
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