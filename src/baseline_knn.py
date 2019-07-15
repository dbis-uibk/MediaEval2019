from loaders.melspectrograms import MelSpectrogramsLoader
import dbispipeline.result_handlers as result_handlers
from dbispipeline.evaluators import FixedSplitGridEvaluator
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

dataloader = MelSpectrogramsLoader(
    "/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-train.tsv", 
    "/storage/nas3/datasets/music/mediaeval2019/autotagging_moodtheme-test.tsv", 
    "/storage/nas3/datasets/music/mediaeval2019/melspec_data"
)

pipeline = Pipeline([
    ("classifier", KNeighborsClassifier())
])

evaluator = FixedSplitGridEvaluator(
    {
        # these parameters will all be tested by gridsearch.
        "classifier__n_neighbors": [1, 3, 5, 10, 100],
    },
    {
        'scoring': 'f1_micro',
        'verbose': 1,
        'n_jobs': 1,
        'cv': 2,
        'iid': True,
    }
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]

