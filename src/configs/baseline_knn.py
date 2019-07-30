import common

from loaders.librosa_features import LibRosaLoader

from dbispipeline.evaluators import FixedSplitGridEvaluator
import dbispipeline.result_handlers as result_handlers

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

MEDIAEVAL_PATH = "/storage/nas3/datasets/music/mediaeval2019"

dataloader = LibRosaLoader(
    MEDIAEVAL_PATH + "/autotagging_moodtheme-train-librosa.pickle",
    MEDIAEVAL_PATH + "/autotagging_moodtheme-test-librosa.pickle",
)

pipeline = Pipeline([("scaler", StandardScaler()),
                     ("model", KNeighborsClassifier())])

evaluator = FixedSplitGridEvaluator(
    params={
        "model__n_neighbors": [1, 3, 5, 10],
    },
    grid_params=common.grid_params(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
