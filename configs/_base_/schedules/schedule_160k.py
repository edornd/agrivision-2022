# optimizer
optimizer = dict(type='AdamW', lr=1e-2, betas=(0.9, 0.999), weight_decay=0.01)
optimizer_config = dict()

lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0,
                 min_lr=1e-6,
                 by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160_000)
checkpoint_config = dict(by_epoch=False, interval=10_000)
evaluation = dict(interval=10_000, metric='mIoU', pre_eval=True)
