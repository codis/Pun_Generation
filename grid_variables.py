grid_params = {
    'lstm': [
        [16],
        [32, 16],
        [16, 16],
        [64, 32],
        [128, 64],
        [256, 128]
    ],
    'merge_layer': ['concat', 'add'],
    'dense': {
        'size':[
            [32],
            [32, 16],
            [64, 32],
            [16, 8],
            [16],
            [8],
            []
        ],
        'dropout':[
            0,
            0.1,
            0.2,
            0.3,
            0.4
        ],
        'act':[
            'elu',
            'relu',
            'selu'
        ]
    },
    'optimizer':[
        'adam',
        'adagrad',
        'rmsprop',
    ],
    'lr':[
        0.01,
        0.05,
        0.001,
        0.005,
        0.0001,
        0.0005,
    ]
    }