"""Fixed protocol. Do not edit once a production dataset has been prepared."""
PROTOCOL = {
    'protocol_version': '1.0',
    'n_classes': 28,
    'samples_per_class': 30,
    'selection_seed': 42,
    'preferred_classes': [
        '3987',
        '7697',
        '7686',
        '3314',
        '20178',
        '12884',
        '6099',
        '8455',
        '20184',
        '7702'
    ],
    # Same-design denominations; retain 7686/7697 because the reverses differ.
    'merge_groups': [['3314', '3987'], ['20178', '20184']],
    'required_classes': ['3314', '3987', '20178', '20184', '7686', '7697'],
    'seeds': [42, 123, 2026],
    'folds': 5,
    'validation_fraction_of_all': 0.1,
    'max_epochs': 100,
    'patience': 12,
    'min_delta': 0.001,
    'batch_size': 32,
    'head_lr': 0.001,
    'backbone_lr': 0.0001,
    'image_size': 224,
    'sift_size': 256,
    'visual_words': 100,
    'sift_C': [1, 10],
    'embedding_dim': 128,
    'margin': 1.0,
    'siamese_pairs_per_epoch': 600,
    'rotation_degrees': 15,
    'brightness': 0.15,
    'contrast': 0.15,
    'dropout': 0.3,
    'alpha': 0.05,
    'weights': {
        'resnet50': 'IMAGENET1K_V2',
        'siamese_resnet18': 'IMAGENET1K_V1',
        'efficientnet_b0': 'IMAGENET1K_V1'
    },
    'sift_kernel': 'rbf',
    'sift_gamma': 'scale',
    'sift_kmeans_batch_size': 2048,
    'sift_kmeans_n_init': 1,
    'selection_metric': 'macro_f1',
    'optimizer': 'Adam',
    'inference_warmup_batches': 1,
}

METHODS = ('sift_svm', 'resnet50', 'siamese_resnet18', 'efficientnet_b0')
EXPERIMENTS = ('original', 'merged')
