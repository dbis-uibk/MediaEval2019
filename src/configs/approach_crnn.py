import common
from dbispipeline.evaluators import FixedSplitGridEvaluator
import dbispipeline.result_handlers as result_handlers
from dbispipeline.utils import prefix_path
from loaders.melspectrograms import MelSpectrogramsLoader
from models.crnn import CRNNModel
from sklearn.pipeline import Pipeline

dataloader = MelSpectrogramsLoader(
    data_path=prefix_path("melspec_data", common.DEFAULT_PATH),
    training_path=prefix_path("autotagging_moodtheme-train.tsv",
                              common.DEFAULT_PATH),
    test_path=prefix_path("autotagging_moodtheme-test.tsv",
                          common.DEFAULT_PATH),
    validate_path=prefix_path("autotagging_moodtheme-validation.tsv",
                              common.DEFAULT_PATH),
)

pipeline = Pipeline([("model", CRNNModel(dataloader=dataloader))])

grid_params = common.grid_params()
grid_params['n_jobs'] = 1

evaluator = FixedSplitGridEvaluator(
    params={
        "model__epochs": [2, 4, 8, 16, 32],
        "model__output_dropout": [None],
    },
    grid_params=grid_params,
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
