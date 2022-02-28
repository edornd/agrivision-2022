_base_ = [
    '../_base_/default_runtime.py',
    # Network Architecture
    '../_base_/models/segformer_mit-b5.py',
    # Dataset
    '../_base_/datasets/agrivision_rgbir.py',
    # Customization
    '../_base_/custom/aug_flip_rot90_jitter_persp_025.py',
    # Training schedule
    '../_base_/schedules/schedule_160k.py'
]
# Random Seed
seed = 0
group = "segformer"

# optimizer
optimizer = dict(_delete_=True,
                 type='AdamW',
                 lr=0.00006,
                 betas=(0.9, 0.999),
                 weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={
                     'pos_block': dict(decay_mult=0.),
                     'norm': dict(decay_mult=0.),
                     'head': dict(lr_mult=10.)
                 }))

lr_config = dict(_delete_=True,
                 policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0,
                 min_lr=1e-7,
                 by_epoch=False)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    # dynamic sampling active if sampling is present
    train=dict(sampling=dict(
        min_pixels=3000,
        min_crop_ratio=0.5,
        temp=1.0,
        window_size=128,
    )))
# important: for dynamic sampling also set return_confidence=True
model = dict(decode_head=dict(return_confidence=True))