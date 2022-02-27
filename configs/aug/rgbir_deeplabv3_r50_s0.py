_base_ = [
    '../_base_/default_runtime.py',
    # Network Architecture
    '../_base_/models/deeplabv3_r50-d8.py',
    # Dataset
    '../_base_/datasets/agrivision_rgbir.py',
    # Customization
    '../_base_/custom/aug_flip_rot90_jitter_010.py',
    # Training schedule
    '../_base_/schedules/schedule_160k.py'
]
# Random Seed
seed = 0
group = "aug"
data = dict(samples_per_gpu=4, workers_per_gpu=4)
