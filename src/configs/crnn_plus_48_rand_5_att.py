import common
from dbispipeline.evaluators import FixedSplitEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
import dbispipeline.result_handlers as result_handlers
from dbispipeline.utils import prefix_path
from loaders.combined import CombinedLoader
from models.crnn import CRNNPlusModel
from sklearn.pipeline import Pipeline

dataloader = CombinedLoader(
    mel_data_path=prefix_path("melspec_data", common.DEFAULT_PATH),
    mel_training_path=prefix_path("autotagging_moodtheme-train.tsv",
                                  common.DEFAULT_PATH),
    mel_test_path=prefix_path("autotagging_moodtheme-test.tsv",
                              common.DEFAULT_PATH),
    mel_validate_path=prefix_path("autotagging_moodtheme-validation.tsv",
                                  common.DEFAULT_PATH),
    ess_training_path=prefix_path("accousticbrainz-train.pickle",
                                  common.DEFAULT_PATH),
    ess_test_path=prefix_path("accousticbrainz-test.pickle",
                              common.DEFAULT_PATH),
    ess_validate_path=prefix_path("accousticbrainz-validation.pickle",
                                  common.DEFAULT_PATH),
    window='random',
    num_windows=5,
)

pipeline = Pipeline([
    ("model", CRNNPlusModel(epochs=48, dataloader=dataloader, attention=True)),
])

evaluator = ModelCallbackWrapper(
    FixedSplitEvaluator(**common.fixed_split_params()),
    lambda model: common.store_prediction(model, dataloader),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
