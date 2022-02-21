import os.path as osp

import mmcv
from mmcv.utils import print_log

from mmseg.utils import get_root_logger
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class AgricultureVisionDataset(CustomDataset):

    CLASSES = ("background", "double_plant", "drydown", "endrow", "nutrient_deficiency", "planter_skip", "water",
               "waterway", "weed_cluster")
    PALETTE = [[64, 64, 64], [23, 190, 207], [32, 119, 180], [148, 103, 189], [43, 160, 44], [127, 127, 127],
               [214, 39, 40], [140, 86, 75], [255, 127, 14]]

    def __init__(self, **kwargs):
        self.nir_dir = "nir"
        self.rgb_dir = "rgb"
        super().__init__(img_suffix=".jpg", seg_map_suffix=".png", **kwargs)

    def load_annotations(self, img_dir: str, img_suffix: str, ann_dir: str, seg_map_suffix: str, split: str):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """
        img_infos = []
        rgb_dir = osp.join(img_dir, self.rgb_dir)
        nir_dir = osp.join(img_dir, self.nir_dir)
        rgb_generator = mmcv.scandir(rgb_dir, img_suffix, recursive=True)
        nir_generator = mmcv.scandir(nir_dir, img_suffix, recursive=True)
        for rgb, nir in zip(rgb_generator, nir_generator):
            assert osp.basename(rgb) == osp.basename(nir)
            img_info = dict(filename=(rgb, nir))
            if ann_dir is not None:
                seg_map = rgb.replace(img_suffix, seg_map_suffix)
                img_info['ann'] = dict(seg_map=seg_map)
            img_infos.append(img_info)
        img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos
