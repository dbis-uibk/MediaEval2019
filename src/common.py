from sklearn.metrics import (average_precision_score, f1_score, make_scorer,
                             precision_score, recall_score, roc_auc_score)

DEFAULT_PATH = '/storage/nas3/datasets/music/mediaeval2019'


def grid_params():
    return {
        'scoring': {
            'f1_micro':
                make_scorer(f1_score, average='micro'),
            'f1_macro':
                make_scorer(f1_score, average='macro'),
            'roc_auc':
                make_scorer(roc_auc_score, average='macro'),
            'average_precision':
                make_scorer(average_precision_score, average='macro'),
            'precision_micro':
                make_scorer(precision_score, average='micro'),
            'precision_macro':
                make_scorer(precision_score, average='macro'),
            'recall_micro':
                make_scorer(recall_score, average='micro'),
            'recall_macro':
                make_scorer(recall_score, average='macro'),
        },
        'verbose': 100,
        'n_jobs': -1,
        'iid': True,
        'refit': False,
    }
