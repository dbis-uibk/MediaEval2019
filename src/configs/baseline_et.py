import common

from loaders.librosa_features import LibRosaLoader

from dbispipeline.evaluators import FixedSplitGridEvaluator
import dbispipeline.result_handlers as result_handlers

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier

MEDIAEVAL_PATH = "/storage/nas3/datasets/music/mediaeval2019"

dataloader = LibRosaLoader(
    MEDIAEVAL_PATH + "/autotagging_moodtheme-train-librosa.pickle",
    MEDIAEVAL_PATH + "/autotagging_moodtheme-test-librosa.pickle",
)

pipeline = Pipeline([("scaler", StandardScaler()),
                     ("model", ExtraTreesClassifier())])

evaluator = FixedSplitGridEvaluator(
    params={
        "model__n_estimators": [5, 10, 25, 100],
        "model__class_weight": [None, "balanced"]
    },
    grid_params=common.grid_params(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
