import common

from loaders.librosa_features import LibRosaLoader

from dbispipeline.evaluators import FixedSplitGridEvaluator
import dbispipeline.result_handlers as result_handlers

from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

MEDIAEVAL_PATH = "/storage/nas3/datasets/music/mediaeval2019"

dataloader = LibRosaLoader(
    MEDIAEVAL_PATH + "/autotagging_moodtheme-train-librosa.pickle",
    MEDIAEVAL_PATH + "/autotagging_moodtheme-test-librosa.pickle",
)

pipeline = Pipeline([("scaler", StandardScaler()),
                     ("model", NearestCentroid())])

evaluator = FixedSplitGridEvaluator(
    params={
        "model__metric": ['eucledian', 'cosine', 'haversine'],
        "model__shrink_threshold": [None, .1, .2],
    },
    grid_params=common.grid_params(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
