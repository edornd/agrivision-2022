import os
from typing import List

import mmcv
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from mmseg.models.builder import MODELS
from mmseg.models.utils.augmentation import AugParams, RepeatableTransform
from mmseg.models.utils.visualization import subplotimg
from mmseg.models.wrapper import SegmentorWrapper
from mmseg.utils.misc import denorm, get_mean_std


@MODELS.register_module()
class CustomModel(SegmentorWrapper):

    def __init__(self,
                 model_config: dict,
                 max_iters: int,
                 resume_iters: int,
                 num_channels: int = 3,
                 aug: dict = None,
                 **kwargs):
        super().__init__(model_config, max_iters, resume_iters, num_channels, **kwargs)
        aug = aug or {}
        self.aug_params = AugParams(**aug)
        self.augment = self.aug_params.factor > 0
        self.aug_transform = RepeatableTransform(self.aug_params) if self.augment else None
        mmcv.print_log(f"Using augmentation invariance: {self.augment}")
        mmcv.print_log("Augmentation parameters:")
        for k, v in self.aug_params.dict().items():
            mmcv.print_log(f"{k:<20s}: {str(v)}")
        # create loss module
        if self.aug_params.loss == "mse":
            self.aug_loss = nn.MSELoss()
        elif self.aug_params.loss == "l1":
            self.aug_loss = nn.L1Loss()
        else:
            raise NotImplementedError(f"Unknown loss: '{self.aug.loss}'")

    def compute_aug_loss(self, features: torch.Tensor, aug_params: dict, split_at: int):
        # first part of the batch are the original images, the last the augmented
        f_x = features[:split_at]
        f_tx = features[split_at:]
        t_fx = self.aug_transform(f_x, params=aug_params, custom_shape=f_x.shape, color_transform=False)
        # compute the loss, smoothed out by the temperature value (decreasing to 1)
        # temp = self.temperature.get_value()
        loss_inv = self.aug_loss(t_fx, f_tx) * self.aug_params.factor
        return {"decode.loss_aug": loss_inv}

    def forward_train(self, img: torch.Tensor, img_metas: List[dict], gt_semantic_seg: torch.Tensor):
        # mmcv.print_log(f"iter: {self.local_iter}")
        # mmcv.print_log(f"input shape: {img.shape}")
        # mmcv.print_log(f"label shape: {gt_semantic_seg.shape}")
        batch_size = img.shape[0]
        device = img.device
        # prepare mean and std for the augmentations
        means, stds = get_mean_std(img_metas=img_metas,
                                   device=device,
                                   batch_size=batch_size,
                                   num_channels=self.num_channels)
        if self.aug_transform is not None:
            self.aug_transform.set_stats(means, stds)

        # INVARIANCE TRANSFORM - source, step 1
        # # compute x and T(x), then concatenate in a single batch
        if self.augment:
            aug_params = self.aug_transform.compute_params(img.shape, device=device)
            img_aug, gt_aug = self.aug_transform(img, gt_semantic_seg, params=aug_params)
            img = torch.cat((img, img_aug), dim=0)
            gt_semantic_seg = torch.cat((gt_semantic_seg, gt_aug), dim=0)

        # Forward on the (possibly augmented) batch with standard segmentation
        losses = super().forward_train(img, img_metas, gt_semantic_seg, seg_weight=None, return_feat=self.augment)
        if self.augment:
            features = losses.pop("decode.features")
            losses_aug: dict = self.compute_aug_loss(features=features, aug_params=aug_params, split_at=batch_size)
            losses.update(losses_aug)
            # debug plots when required
            if self.aug_params.debug_augs and self.local_iter % self.aug_params.debug_interval == 0:
                self._plot_aug_debug(img[:batch_size],
                                     img[batch_size:],
                                     gt_semantic_seg[:batch_size],
                                     gt_semantic_seg[batch_size:],
                                     features[:batch_size],
                                     features[batch_size:],
                                     means=means,
                                     stds=stds,
                                     batch_size=batch_size)

        # increment iteration
        self.local_iter += 1
        return losses

    def _plot_aug_debug(self,
                        src_img: torch.Tensor,
                        aug_img: torch.Tensor,
                        src_labels: torch.Tensor,
                        aug_labels: torch.Tensor,
                        src_features: torch.Tensor,
                        aug_features: torch.Tensor,
                        means: torch.Tensor,
                        stds: torch.Tensor,
                        batch_size: int,
                        prefix=""):
        # create dir in the cwd to store plots
        out_dir = os.path.join(self.work_dir, 'aug_debug')
        os.makedirs(out_dir, exist_ok=True)
        # denormalize images to restore 0-1 range
        vis_src_img = torch.clamp(denorm(src_img, means[:batch_size], stds[:batch_size]), 0, 1)
        vis_aug_img = torch.clamp(denorm(aug_img, means[:batch_size], stds[:batch_size]), 0, 1)
        # iterate batch and save plots
        for i in range(batch_size):
            index = np.random.randint(0, src_features.shape[1])
            rows, cols = 2, 3
            fig, axs = plt.subplots(rows,
                                    cols,
                                    figsize=(3 * cols, 3 * rows),
                                    gridspec_kw={
                                        'hspace': 0.1,
                                        'wspace': 0,
                                        'top': 0.95,
                                        'bottom': 0,
                                        'right': 1,
                                        'left': 0
                                    })
            subplotimg(axs[0][0], vis_src_img[i, :3], 'Source image')
            subplotimg(axs[1][0], vis_aug_img[i, :3], 'Aug. image')
            subplotimg(axs[0][1], src_labels[i], 'Source Label', cmap='agrivision')
            subplotimg(axs[1][1], aug_labels[i], 'Aug. Label', cmap='agrivision')
            subplotimg(axs[0][2], src_features[i, index], 'f(x)', cmap="viridis")
            subplotimg(axs[1][2], aug_features[i, index], 'f(T(x))', cmap="viridis")

            for ax in axs.flat:
                ax.axis('off')
            plt.savefig(os.path.join(out_dir, f'{prefix}_{(self.local_iter + 1):06d}_{i}.png'))
            plt.close(fig)
