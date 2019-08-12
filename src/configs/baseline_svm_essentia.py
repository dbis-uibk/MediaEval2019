import common
from dbispipeline.evaluators import FixedSplitGridEvaluator
import dbispipeline.result_handlers as result_handlers
from dbispipeline.utils import prefix_path
from loaders.acousticbrainz import AcousticBrainzLoader
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

dataloader = AcousticBrainzLoader(
    training_path=prefix_path("accousticbrainz-train.pickle",
                              common.DEFAULT_PATH),
    test_path=prefix_path("accousticbrainz-test.pickle", common.DEFAULT_PATH),
    validation_path=prefix_path("accousticbrainz-validation.pickle",
                                common.DEFAULT_PATH),
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MultiOutputClassifier(SVC(probability=True))),
])

evaluator = FixedSplitGridEvaluator(
    params={"model__estimator__C": [0.1, 1.0, 10.0]},
    grid_params=common.grid_params(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
