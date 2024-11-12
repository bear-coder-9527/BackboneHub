# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='CASViT',
        arch='s'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))