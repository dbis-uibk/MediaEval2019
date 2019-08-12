import common

import dbispipeline.result_handlers as result_handlers
from dbispipeline.evaluators import FixedSplitGridEvaluator
from loaders.melspectrograms import MelSpectrogramsLoader
from loaders.acousticbrainz import AcousticBrainzLoader
from models.crnn_plus import CRNNPlusModel
from sklearn.pipeline import Pipeline

ab_loader = AcousticBrainzLoader(
    training_path=
    "/storage/nas3/datasets/music/mediaeval2019/accousticbrainz-train.pickle",
    test_path=
    "/storage/nas3/datasets/music/mediaeval2019/accousticbrainz-test.pickle",
    validation_path=
    "/storage/nas3/datasets/music/mediaeval2019/accousticbrainz-validation.pickle"
)

dataloader = MelSpectrogramsLoader(
    data_path="/storage/nas3/datasets/music/mediaeval2019/melspec_data",
    training_path=
    "/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-train.tsv",  # noqa E501
    test_path=
    "/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-test.tsv",  # noqa E501
    validate_path=
    "/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-validation.tsv",  # noqa E501
)

pipeline = Pipeline([("model",
                      CRNNPlusModel(dataloader=dataloader,
                                    essentia_loader=ab_loader))])

grid_params = common.grid_params()
grid_params['n_jobs'] = 1

evaluator = FixedSplitGridEvaluator(
    params={
        "model__epochs": [3],
    },
    grid_params=grid_params,
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
