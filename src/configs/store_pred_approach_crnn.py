import common
from dbispipeline.evaluators import FixedSplitEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
import dbispipeline.result_handlers as result_handlers
from dbispipeline.utils import prefix_path
from loaders.melspectrograms import MelSpectrogramsLoader
from models.crnn import CRNNModel
import numpy as np
from sklearn.pipeline import Pipeline

WINDOW_SIZE = 1366


def store_prediction(model, dataloader, file_name_prefix):
    x_test, _ = dataloader.load_test()
    y_pred = model.predict(x_test)
    np.save(file_name_prefix + '_decisions.npy', y_pred.astype(bool))
    y_pred = model.predict_proba(x_test)
    np.save(file_name_prefix + '_predictions.npy', y_pred.astype(np.float64))


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
    ("model", CRNNModel(epochs=1, dataloader=dataloader)),
])

evaluator = ModelCallbackWrapper(
    FixedSplitEvaluator(**common.fixed_split_params()),
    lambda model: store_prediction(model, dataloader, 'crnn'),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
