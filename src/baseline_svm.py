from loaders.librosa_features import LibRosaLoader
import dbispipeline.result_handlers as result_handlers
from dbispipeline.evaluators import FixedSplitGridEvaluator
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier

dataloader = LibRosaLoader(
    "/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-train-librosa.pickle",
    "/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-test-librosa.pickle"
)

pipeline = Pipeline([("scaler", StandardScaler()),
                     ("model", MultiOutputClassifier(SVC(probability=True)))])

evaluator = FixedSplitGridEvaluator(
    {
        # these parameters will all be tested by gridsearch.
        "model__estimator__C": [0.1, 1.0, 10.0]
    },
    {
        'scoring': ['f1_micro', 'f1_macro', 'roc_auc', 'average_precision'],
        'verbose': 100,
        'n_jobs': 1,
        'iid': True,
        'refit': False,
    })

result_handlers = [
    result_handlers.print_gridsearch_results,
]
