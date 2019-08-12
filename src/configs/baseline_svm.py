import common
from dbispipeline.evaluators import FixedSplitGridEvaluator
import dbispipeline.result_handlers as result_handlers
from dbispipeline.utils import prefix_path
from loaders.librosa_features import LibRosaLoader
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

dataloader = LibRosaLoader(
    prefix_path("autotagging_moodtheme-train-librosa.pickle",
                common.DEFAULT_PATH),
    prefix_path("autotagging_moodtheme-test-librosa.pickle",
                common.DEFAULT_PATH),
)

pipeline = Pipeline([("scaler", StandardScaler()),
                     ("model", MultiOutputClassifier(SVC(probability=True)))])

evaluator = FixedSplitGridEvaluator(
    params={"model__estimator__C": [0.1, 1.0, 10.0]},
    grid_params=common.grid_params(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
