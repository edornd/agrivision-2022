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
                 min_lr=0.0,
                 by_epoch=False)

data = dict(samples_per_gpu=2, workers_per_gpu=2)
checkpoint_config = dict(by_epoch=False, interval=10_000)
evaluation = dict(interval=10_000, metric='mIoU', pre_eval=True)
