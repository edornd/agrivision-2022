from mmseg.models.builder import MODELS
from mmseg.models.wrapper import SegmentorWrapper


@MODELS.register_module()
class CustomModel(SegmentorWrapper):

    def forward_train(self, img, img_metas, gt_semantic_seg, return_feat=False):
        # print(f"iter: {self.local_iter}")
        # print(f"input shape: {img.shape}")
        # print(f"label shape: {gt_semantic_seg.shape}")
        losses = self.model.forward_train(img, img_metas, gt_semantic_seg, seg_weight=None, return_feat=return_feat)
        self.local_iter += 1
        return losses
