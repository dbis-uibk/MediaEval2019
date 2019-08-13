import common
from dbispipeline.evaluators import FixedSplitGridEvaluator
import dbispipeline.result_handlers as result_handlers
from dbispipeline.utils import prefix_path
from loaders.acousticbrainz import AcousticBrainzLoader
from loaders.melspectrograms import MelSpectrogramsLoader
from models.crnn_plus import CRNNPlusModel
from sklearn.pipeline import Pipeline

ab_loader = AcousticBrainzLoader(
    training_path=prefix_path("accousticbrainz-train.pickle",
                              common.DEFAULT_PATH),
    test_path=prefix_path("accousticbrainz-test.pickle", common.DEFAULT_PATH),
    validation_path=prefix_path("accousticbrainz-validation.pickle",
                                common.DEFAULT_PATH),
)

dataloader = MelSpectrogramsLoader(
    data_path=prefix_path("melspec_data", common.DEFAULT_PATH),
    training_path=prefix_path("autotagging_moodtheme-train.tsv",
                              common.DEFAULT_PATH),
    test_path=prefix_path("autotagging_moodtheme-test.tsv",
                          common.DEFAULT_PATH),
    validate_path=prefix_path("autotagging_moodtheme-validation.tsv",
                              common.DEFAULT_PATH),
)

pipeline = Pipeline([("model",
                      CRNNPlusModel(dataloader=dataloader,
                                    essentia_loader=ab_loader))])

grid_params = common.grid_params()
grid_params['n_jobs'] = 1

evaluator = FixedSplitGridEvaluator(
    params={
        "model__epochs": [2, 4, 8, 16, 32],
    },
    grid_params=grid_params,
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
