# Agriculture Vision 2022

Code for [Augmentation Invariance and Adaptive Sampling in Semantic Segmentation of Agricultural Aerial Images](https://arxiv.org/abs/2204.07969)  (CVPRW 2022)

## Installation

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
pip install -e .
```

## Bring me to the important bits

Well, there are mostly two contributions in this work: augmentation invariance, and adaptive sampling.
1. The augmentation invariance regularization is computed in [custom.py](https://github.com/edornd/agrivision-2022/blob/5d3a72a4310de94df81ad753f2a2f1691a96f1dc/mmseg/models/custom.py#L147)
2. The adaptive sampling is part of the dataset, in [sampling.py](https://github.com/edornd/agrivision-2022/blob/5d3a72a4310de94df81ad753f2a2f1691a96f1dc/mmseg/datasets/sampling.py#L43)

## Training

### Single GPU
```bash
 CUDA_VISIBLE_DEVICES=$DEVICES tools/train.py [CONFIG_FILE] [OPTIONS]
```

### Multi GPU
```bash
 source scripts/dist_train.sh [GPUS, e.g. 0,1] [CONFIG_FILE] [OPTIONS]
```

## Testing
Evaluation only:
```bash
 CUDA_VISIBLE_DEVICES=$DEVICES python tools/test.py [CONFIG] --eval mIoU
```

Evaluate generating images:
```bash
 # evaluate while plotting masks
 CUDA_VISIBLE_DEVICES=$DEVICES python tools/test.py [CONFIG] --eval mIoU --show --opacity=1
 # evaluate while plotting RGB images
 CUDA_VISIBLE_DEVICES=$DEVICES python tools/test.py [CONFIG] --eval mIoU --show --opacity=0
 # evaluate while plotting IRRG images
 CUDA_VISIBLE_DEVICES=$DEVICES python tools/test.py [CONFIG] --eval mIoU --show --opacity=0 --channels irrg
```
