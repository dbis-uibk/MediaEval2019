from loaders.librosa_features import LibRosaLoader
import dbispipeline.result_handlers as result_handlers
from dbispipeline.evaluators import FixedSplitGridEvaluator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier

dataloader = LibRosaLoader(
    "/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-train-librosa.pickle", 
    "/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-test-librosa.pickle"
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", ExtraTreesClassifier())
])

evaluator = FixedSplitGridEvaluator(
    {
        # these parameters will all be tested by gridsearch.
        "classifier__n_estimators": [5, 10, 25, 100],       
        "classifier__class_weight": [None, "balanced"]
    },
    {
        'scoring': ['f1_micro', 'f1_macro', 'roc_auc'],
        'verbose': 100,
        'n_jobs': 1,
        'iid': True,
    }
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]

