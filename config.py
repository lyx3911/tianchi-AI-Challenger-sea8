args_resnet = {
    'epochs': 200,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.01,
        'momentum': 0.9,
        # 'betas': (0.9, 0.999), 
        # 'eps': 1e-08,
        'weight_decay': 1e-4
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 200
    },
    # 'scheduler_name': None,
    'batch_size': 256,
}
args_densenet = {
    'epochs': 200,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.01,
        # 'betas': (0.9, 0.999), 
        # 'eps': 1e-08,
        'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 200
    },
    # 'scheduler_name': None,
    'batch_size': 256,
}