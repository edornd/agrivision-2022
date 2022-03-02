# Agriculture Vision 2022

Repository for CVPRW 2022

## Installation

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
pip install -e .
```

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
```bash
 CUDA_VISIBLE_DEVICES=$DEVICES python tools/test.py [CONFIG] [CHECKPOINT] --eval mIoU
```