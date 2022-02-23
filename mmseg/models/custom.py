from mmseg.models.builder import MODELS
from mmseg.models.wrapper import SegmentorWrapper


@MODELS.register_module()
class CustomModel(SegmentorWrapper):

    def forward_train(self, img, img_metas, gt_semantic_seg, return_feat=False):
        return super().forward_train(img, img_metas, gt_semantic_seg, return_feat)
