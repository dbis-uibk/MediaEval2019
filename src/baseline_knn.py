from loaders.librosa_features import LibRosaLoader
import dbispipeline.result_handlers as result_handlers
from dbispipeline.evaluators import FixedSplitGridEvaluator
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

dataloader = LibRosaLoader(
    "/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-train-librosa.pickle", 
    "/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-test-librosa.pickle"
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier())
])

evaluator = FixedSplitGridEvaluator(
    {
        # these parameters will all be tested by gridsearch.
        "classifier__n_neighbors": [1, 3, 5, 10],
    },
    {
        'scoring': ['f1_micro', 'f1_macro', 'roc_auc'],
        'verbose': 100,
        'n_jobs': 1,
        'iid': True,
        'refit': True
    }
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]

