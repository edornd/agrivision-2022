_base_ = [
    '../_base_/default_runtime.py',
    # Network Architecture
    '../_base_/models/ocrnet_hr18.py',
    # Dataset
    '../_base_/datasets/agrivision_rgbir.py',
    # Customization
    '../_base_/custom/base.py',
    # Training schedule
    '../_base_/schedules/schedule_160k.py'
]
# Random Seed
seed = 0
group = "rcs"
data = dict(samples_per_gpu=4,
            workers_per_gpu=2,
            train=dict(sampling=dict(
                min_pixels=3000,
                min_crop_ratio=0.5,
                window_size=128,
            )))
