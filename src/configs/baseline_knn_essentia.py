import common

from loaders.acousticbrainz import AcousticBrainzLoader

from dbispipeline.evaluators import FixedSplitGridEvaluator
import dbispipeline.result_handlers as result_handlers

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

dataloader = AcousticBrainzLoader(
    training_path="/storage/nas3/datasets/music/mediaeval2019/accousticbrainz-train.pickle",
    test_path="/storage/nas3/datasets/music/mediaeval2019/accousticbrainz-test.pickle",
    validation_path="/storage/nas3/datasets/music/mediaeval2019/accousticbrainz-validation.pickle"
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
