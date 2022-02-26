_base_ = [
    '../_base_/default_runtime.py',
    # Network Architecture
    '../_base_/models/fpn_r50.py',
    # Dataset
    '../_base_/datasets/agrivision_rgbir.py',
    # Customization
    '../_base_/custom/aug.py',
    # Training schedule
    '../_base_/schedules/schedule_160k.py'
]
# Random Seed
seed = 0
group = "aug"
data = dict(samples_per_gpu=2, workers_per_gpu=2)
