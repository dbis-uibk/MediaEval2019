from loaders.melspectrograms import MelSpectrogramsLoader
from models.crnn import CRNNModel
import dbispipeline.result_handlers as result_handlers
from dbispipeline.evaluators import FixedSplitGridEvaluator
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

dataloader = MelSpectrogramsLoader(
    "/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-train.tsv",
    "/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-test.tsv",
    "/storage/nas3/datasets/music/mediaeval2019/melspec_data")

pipeline = Pipeline([("model", CRNNModel())])

evaluator = FixedSplitGridEvaluator(
    {
        # these parameters will all be tested by gridsearch.
        "model__num_filters": [50],
        "model__epochs": [2],
    },
    {
        'scoring': ['f1_micro', 'f1_macro'],
        'verbose': 100,
        'n_jobs': 1,
        'iid': True,
        'refit': False,
    })

result_handlers = [
    result_handlers.print_gridsearch_results,
]
