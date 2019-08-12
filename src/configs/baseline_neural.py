import common
from dbispipeline.evaluators import FixedSplitGridEvaluator
import dbispipeline.result_handlers as result_handlers
from dbispipeline.utils import prefix_path
from loaders.librosa_features import LibRosaLoader
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

dataloader = LibRosaLoader(
    prefix_path("autotagging_moodtheme-train-librosa.pickle",
                common.DEFAULT_PATH),
    prefix_path("autotagging_moodtheme-test-librosa.pickle",
                common.DEFAULT_PATH),
)

pipeline = Pipeline([("scaler", StandardScaler()), ("model", MLPClassifier())])

evaluator = FixedSplitGridEvaluator(
    params={},
    grid_params=common.grid_params(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
