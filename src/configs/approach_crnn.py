import common

import dbispipeline.result_handlers as result_handlers
from dbispipeline.evaluators import FixedSplitGridEvaluator
from loaders.melspectrograms import MelSpectrogramsLoader
from models.crnn import CRNNModel
from sklearn.pipeline import Pipeline

dataloader = MelSpectrogramsLoader(
    data_path="/storage/nas3/datasets/music/mediaeval2019/melspec_data",
    training_path=
    "/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-train.tsv",  # noqa E501
    test_path=
    "/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-test.tsv",  # noqa E501
    validate_path=
    "/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-validation.tsv",  # noqa E501
)

pipeline = Pipeline([("model", CRNNModel(dataloader=dataloader))])

grid_params = common.grid_params()
grid_params['n_jobs'] = 1

evaluator = FixedSplitGridEvaluator(
    params={
        "model__epochs": [2],
    },
    grid_params=grid_params,
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
