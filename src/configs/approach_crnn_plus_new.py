import common
from dbispipeline.evaluators import FixedSplitGridEvaluator
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

pipeline = Pipeline([("model", CRNNPlusModel(dataloader=dataloader))])

grid_params = common.grid_params()
grid_params['n_jobs'] = 1

evaluator = FixedSplitGridEvaluator(
    params={
        "model__epochs": [8, 16, 32],
        "model__output_dropout": [0.3],
        "model__concat_bn": [True],
    },
    grid_params=grid_params,
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
