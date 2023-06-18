import os

import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

"""
In the loaded numpy array, only 0-6 integer labels are allowed, and they represent the annotations in the following way:
0 - background
1 - double_plant
2 - drydown
3 - endrow
4 - nutrient_deficiency
5 - planter_skip
6 - water
7 - waterway
8 - weed_cluster
9 - storm_damage (not evaluated)
"""

palette = {
    0: (0, 0, 0),  # background
    1: (23, 190, 207),  # double_plant
    2: (32, 119, 180),  # drydown
    3: (148, 103, 189),  # endrow
    4: (43, 160, 44),  # nutrient_deficiency
    5: (127, 127, 127),  # planter_skip
    6: (214, 39, 40),  # water
    7: (140, 86, 75),  # waterway
    8: (255, 127, 14),  # weed cluster
}

labels_folder = {
    "double_plant": 1,
    "drydown": 2,
    "endrow": 3,
    "nutrient_deficiency": 4,
    "planter_skip": 5,
    "water": 6,
    "waterway": 7,
    "weed_cluster": 8,
}

excluded_folder = {
    "storm_damage": 9,
}

land_classes = [
    "background",
    "double_plant",
    "drydown",
    "endrow",
    "nutrient_deficiency",
    "planter_skip",
    "water",
    "waterway",
    "weed_cluster",
]

IMG = "images"  # RGB or IRRG, rgb/nir
GT = "gt"
IDS = "IDs"


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def img_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


def is_image(filename):
    return any(filename.endswith(ext) for ext in [".png", ".jpg"])


def prepare_gt(
    root_folder: str,
    out_path: str = "gt",
    ignore_index: int = 255,
    check_only: bool = False,
):
    if not os.path.exists(os.path.join(root_folder, out_path)) or not check_only:
        print(f"----------Creating groundtruth data for: {root_folder}---------------")
        check_mkdir(os.path.join(root_folder, out_path))
        basenames = [
            img_basename(f) for f in os.listdir(os.path.join(root_folder, "images/rgb"))
        ]
        assert len(basenames) > 0, f"Image count not quite right: {len(basenames)}"

        def process(fname: str):
            gtz = np.zeros((512, 512), dtype=int)
            # merge single label files
            for label_name, label_index in labels_folder.items():
                image_filename = f"{fname}.png"
                label = cv2.imread(
                    os.path.join(root_folder, "labels", label_name, image_filename), -1
                )
                mask = np.array(label / 255, dtype=int) * label_index
                gtz[gtz < 1] = mask[gtz < 1]
            # mask excluded parts with ignore index
            for subdir in ["boundaries", "masks"]:
                mask = np.array(
                    cv2.imread(os.path.join(root_folder, subdir, image_filename), -1)
                    / 255,
                    dtype=int,
                )
                gtz[mask == 0] = ignore_index
            # excluded classes
            for label_name, label_index in excluded_folder.items():
                mask = cv2.imread(
                    os.path.join(root_folder, "labels", label_name, image_filename), -1
                )
                mask = np.array(mask / 255, dtype=int) * label_index
                gtz[mask == label_index] = ignore_index
            cv2.imwrite(os.path.join(root_folder, out_path, image_filename), gtz)

        # parallelize processing on multiple jobs
        Parallel(n_jobs=16)(delayed(process)(name) for name in tqdm(basenames))
    else:
        print(f"----------Checking groundtruth data for: {root_folder}---------------")
        basenames = [
            img_basename(f) for f in os.listdir(os.path.join(root_folder, "images/rgb"))
        ]
        for name in tqdm(basenames):
            image_filename = f"{name}.png"
            gt_path = os.path.join(root_folder, out_path, image_filename)
            assert os.path.exists(gt_path), f"Missing ground truth for '{name}'"
            image = np.array(cv2.imread(gt_path, -1))
            assert image.shape == (512, 512)


def get_training_list(root_folder=None, count_label=True):
    dict_list = {}
    basename = [
        img_basename(f) for f in os.listdir(os.path.join(root_folder, "images/nir"))
    ]
    if count_label:
        for key in labels_folder.keys():
            no_zero_files = []
            for fname in basename:
                gt = np.array(
                    cv2.imread(
                        os.path.join(root_folder, "labels", key, fname + ".png"), -1
                    )
                )
                if np.count_nonzero(gt):
                    no_zero_files.append(fname)
                else:
                    continue
            dict_list[key] = no_zero_files
    return dict_list, basename


def split_train_val_test_sets(data_root, name="agrivision", bands=["NIR", "RGB"]):
    dataset_root = data_root
    train_root = os.path.join(dataset_root, "train")
    val_root = os.path.join(dataset_root, "val")
    test_root = os.path.join(dataset_root, "test")
    data_folder = {
        "agrivision": {
            "ROOT": dataset_root,
            "RGB": "images/rgb/{}.jpg",
            "NIR": "images/nir/{}.jpg",
            "SHAPE": (512, 512),
            "GT": "gt/{}.png",
        },
    }

    train_id, t_list = get_training_list(root_folder=train_root, count_label=False)
    val_id, v_list = get_training_list(root_folder=val_root, count_label=False)
    test_id, test_list = get_training_list(root_folder=test_root, count_label=False)

    img_folders = [
        os.path.join(data_folder[name]["ROOT"], "train", data_folder[name][band])
        for band in bands
    ]
    gt_folder = os.path.join(
        data_folder[name]["ROOT"], "train", data_folder[name]["GT"]
    )

    val_folders = [
        os.path.join(data_folder[name]["ROOT"], "val", data_folder[name][band])
        for band in bands
    ]
    val_gt_folder = os.path.join(
        data_folder[name]["ROOT"], "val", data_folder[name]["GT"]
    )

    test_folders = [
        os.path.join(data_folder[name]["ROOT"], "test", data_folder[name][band])
        for band in bands
    ]

    train_dict = {
        IDS: train_id,
        IMG: [[img_folder.format(id) for img_folder in img_folders] for id in t_list],
        GT: [gt_folder.format(id) for id in t_list],
        "all_files": t_list,
    }

    val_dict = {
        IDS: val_id,
        IMG: [[val_folder.format(id) for val_folder in val_folders] for id in v_list],
        GT: [val_gt_folder.format(id) for id in v_list],
        "all_files": v_list,
    }

    test_dict = {
        IDS: test_id,
        IMG: [
            [test_folder.format(id) for test_folder in test_folders] for id in test_list
        ],
        "all_files": test_list,
    }

    return train_dict, val_dict, test_dict
