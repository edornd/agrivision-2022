# Agriculture Vision 2022

Repository for CVPRW 2022

## Installation

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.5
```

## Training

```bash
 CUDA_VISIBLE_DEVICES=[DEVICES] source tools/dist_train.sh [CONFIG_FILE] [NUM_GPUS] [OPTIONS]
```