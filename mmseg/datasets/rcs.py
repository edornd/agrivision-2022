import json
import os.path as osp
from abc import abstractmethod

import mmcv
import numpy as np
import torch

from mmseg.datasets.custom import CustomDataset


def get_rcs_class_probs(data_root: str, temperature: float):
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
    freq = torch.softmax(freq / temperature, dim=-1)
    return list(overall_class_stats.keys()), freq.numpy()


class RCSDataset(CustomDataset):

    def __init__(self, rare_class_sampling: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.rcs_enabled = rare_class_sampling is not None
        if self.rcs_enabled:
            self.rcs_class_temp = rare_class_sampling['class_temp']
            self.rcs_min_crop_ratio = rare_class_sampling['min_crop_ratio']
            self.rcs_min_pixels = rare_class_sampling['min_pixels']

            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(self.data_root, self.rcs_class_temp)
            mmcv.print_log(f'RCS Classes: {self.rcs_classes}', 'mmseg')
            mmcv.print_log(f'RCS ClassProb: {self.rcs_classprob}', 'mmseg')

            with open(osp.join(self.data_root, 'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items() if int(k) in self.rcs_classes
            }
            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}
            for i, dic in enumerate(self.img_infos):
                file = dic['ann']['seg_map']
                self.file_to_idx[file] = i

    def get_rare_class_sample(self):
        # select a class with probability given by the stats
        # then select a sample containing such class
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f = np.random.choice(self.samples_with_class[c])
        idx = self.file_to_idx[f]
        sample = self.prepare_batch(idx)
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(sample['gt_semantic_seg'].data == (c - 1))
                # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                sample = self.prepare_batch(idx)
        return sample

    @abstractmethod
    def prepare_batch(self, idx: int):
        raise NotImplementedError()

    def prepare_train_img(self, idx: int):
        """Yet another wrapper to switch between RCS and standard random sampling.
        """
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        return self.prepare_batch(idx)
