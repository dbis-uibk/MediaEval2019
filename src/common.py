from os.path import basename
from os.path import splitext

from dbispipeline import store
import numpy as np
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


def store_prediction(model, dataloader, file_name_prefix=None):
    if not file_name_prefix:
        file_name_prefix = type(model).__name__
    elif file_name_prefix[-1] != '_':
        file_name_prefix += '_'

    if store['config_path']:
        file_name_prefix += splitext(basename(store['config_path']))[0]

    x_test, _ = dataloader.load_test()
    y_pred = model.predict(x_test)
    np.save(file_name_prefix + '_decisions.npy', y_pred.astype(bool))
    y_pred = model.predict_proba(x_test)
    np.save(file_name_prefix + '_predictions.npy', y_pred.astype(np.float64))
