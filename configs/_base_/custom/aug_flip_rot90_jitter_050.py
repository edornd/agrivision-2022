custom = dict(
    type="CustomModel",
    aug=dict(
        factor=0.50,
        hflip_prob=0.5,
        vflip_prob=0.5,
        random_degrees=360,  # freedom in terms of degrees
        random_step=90,  # min degree interval (1 means full freedom)
        jitter_prob=0.5,
        jitter_strength=0.10,
        perspective_prob=0.0,
        perspective_dist=0.0,
        debug_augs=True,
        debug_interval=2000))
