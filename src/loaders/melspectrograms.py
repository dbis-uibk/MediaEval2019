from os import path

from dbispipeline.base import TrainTestLoader

import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer

from . import utils


class MelSpectrogramsLoader(TrainTestLoader):
    """
    Loads the mel-spectrograms provided by the task organizers and the labels
    for both the training and test set.
    """

    def __init__(self, training_path, test_path, data_path,
                 center_sample=True):
        self.training_path = training_path
        self.test_path = test_path
        self.data_path = data_path
        # FIXME: this flag is unused
        self.center_sample = center_sample
        self.mlb = MultiLabelBinarizer()
        self.mlb_fit = True

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
        X = []

        for sample in sample_set['PATH']:
            sample_path = sample.replace('.mp3', '.npy')
            X_temp = np.load(path.join(self.data_path, sample_path))
            start_idx = int(X_temp.shape[1] / 2 - 683)
            X_temp = X_temp[:, start_idx:(start_idx + 1366)]

            X.append(X_temp)

        return np.array(X)

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
            'center_sample': self.center_sample,
        }
