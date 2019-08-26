import common
from dbispipeline.evaluators import FixedSplitEvaluator
import dbispipeline.result_handlers as result_handlers
from dbispipeline.utils import prefix_path
from loaders.melspectrograms import MelSpectrogramsLoader
from models.crnn import CRNNModel
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

WINDOW_SIZE = 1366

dataloader = MelSpectrogramsLoader(
    data_path=prefix_path("melspec_data", common.DEFAULT_PATH),
    training_path=prefix_path("autotagging_moodtheme-train.tsv",
                              common.DEFAULT_PATH),
    test_path=prefix_path("autotagging_moodtheme-test.tsv",
                          common.DEFAULT_PATH),
    validate_path=prefix_path("autotagging_moodtheme-validation.tsv",
                              common.DEFAULT_PATH),
    window_size=WINDOW_SIZE,
)

pipeline = Pipeline([
    ("model", CRNNModel(epochs=2, dataloader=dataloader)),
])

evaluator = FixedSplitEvaluator(
    scoring={
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
        'average_precision_all':
            make_scorer(average_precision_score, average=None),
        'roc_auc_all':
            make_scorer(roc_auc_score, average=None),
        'confusion_matrix':
            make_scorer(multilabel_confusion_matrix),
    })

result_handlers = [
    result_handlers.print_gridsearch_results,
]
