def grid_params():
    return {
        'scoring': [
            'f1_micro',
            'f1_macro',
            'roc_auc',
            'average_precision',
            'precision_micro',
            'precision_macro',
            'recall_micro',
            'recall_macro',
        ],
        'verbose': 100,
        'n_jobs': 1,
        'iid': True,
        'refit': False,
    }
