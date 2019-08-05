import json
from os import path

from dbispipeline.base import TrainTestLoader

import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

from . import utils


class AcousticBrainzLoader(TrainTestLoader):
    """
    Loads the AcousticBrainz features provided by the task organizers and the
    labels for both the training and test set.
    """

    def __init__(self, training_path, test_path, data_path):
        self.training_path = training_path
        self.test_path = test_path
        self.data_path = data_path
        self.mlb = MultiLabelBinarizer()
        self.mlb_fit = True

    def load_train(self):
        """Returns the train data."""
        return self._load_set(self.training_path)

    def load_test(self):
        """Returns the test data."""
        return self._load_set(self.test_path)

    @property
    def configuration(self):
        """Returns a dict-like representation of the configuration of this loader.

        This is for storing its state in the database.
        """
        return {
            'training_path': self.training_path,
            'test_path': self.test_path,
            'data_path': self.data_path,
        }

    def _load_set(self, set_path):
        sample_set = utils.load_set_info(set_path)[['PATH', 'TAGS']]
        X = self._load_data(sample_set)

        # TODO: Remove workaround
        if self.mlb_fit:
            y = self.mlb.fit_transform(sample_set['TAGS'])
            self.mlb_fit = False
        else:
            y = self.mlb.transform(sample_set['TAGS'])

        return X, y

    def _load_data(self, sample_set):
        samples = pd.DataFrame()

        for sample in sample_set['PATH']:
            sample_path = sample.replace('.mp3', '.json')
            data = json.load(path.join(self.data_path, sample_path))
            data = pd.io.json.json_normalize(data)
            samples.append(data, sort=False, ignore_index=True)

        return samples
