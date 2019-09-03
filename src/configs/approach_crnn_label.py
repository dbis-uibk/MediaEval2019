import common
from dbispipeline.evaluators import FixedSplitEvaluator
import dbispipeline.result_handlers as result_handlers
from dbispipeline.utils import prefix_path
from loaders.melspectrograms import MelSpectrogramsLoader
from models.crnn import CRNNModel
from sklearn.pipeline import Pipeline

WINDOW_SIZE = 1366

dataloader = MelSpectrogramsLoader(
    data_path=prefix_path("melspec_data", common.DEFAULT_PATH),
    training_path=prefix_path("autotagging_moodtheme-train.tsv",
                              common.DEFAULT_PATH),
    test_path=prefix_path("autotagging_moodtheme-test.tsv",
                          common.DEFAULT_PATH),
    validate_path=prefix_path("autotagging_moodtheme-validation.tsv",
                              common.DEFAULT_PATH),
    window_size=WINDOW_SIZE,
)

pipeline = Pipeline([
    ("model", CRNNModel(epochs=8, dataloader=dataloader)),
])

evaluator = FixedSplitEvaluator(**common.fixed_split_params())

result_handlers = [
    result_handlers.print_gridsearch_results,
]
