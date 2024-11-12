# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='CASViT',
        arch='xs'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=220,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))