from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

DEFAULT_PATH = '/storage/nas3/datasets/music/mediaeval2019'


def grid_params():
    return {
        'scoring': {
            'f1_micro':
                make_scorer(f1_score, average='micro'),
            'f1_macro':
                make_scorer(f1_score, average='macro'),
            'roc_auc':
                make_scorer(roc_auc_score, average='macro', needs_proba=True),
            'average_precision':
                make_scorer(average_precision_score,
                            average='macro',
                            needs_proba=True),
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


def fixed_split_params():
    return {
        'scoring': {
            'f1_micro':
                make_scorer(f1_score, average='micro'),
            'f1_macro':
                make_scorer(f1_score, average='macro'),
            'roc_auc':
                make_scorer(roc_auc_score, average='macro', needs_proba=True),
            'average_precision':
                make_scorer(average_precision_score,
                            average='macro',
                            needs_proba=True),
            'precision_micro':
                make_scorer(precision_score, average='micro'),
            'precision_macro':
                make_scorer(precision_score, average='macro'),
            'recall_micro':
                make_scorer(recall_score, average='micro'),
            'recall_macro':
                make_scorer(recall_score, average='macro'),
            'average_precision_all':
                make_scorer(average_precision_score,
                            average=None,
                            needs_proba=True),
            'roc_auc_all':
                make_scorer(roc_auc_score, average=None, needs_proba=True),
            'confusion_matrix':
                make_scorer(multilabel_confusion_matrix),
        },
    }
