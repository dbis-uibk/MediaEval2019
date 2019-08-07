import json
from os import path

from dbispipeline.base import TrainValidateTestLoader
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from . import utils


class AcousticBrainzLoader(TrainValidateTestLoader):
    """
    Loads the AcousticBrainz features provided by the task organizers and the
    labels for both the training and test set.
    """

    def __init__(self, training_path, test_path, validation_path):
        self.training_path = training_path
        self.test_path = test_path
        self.validation_path = validation_path

        self.mlb = MultiLabelBinarizer()
        self.mlb_fitted = False

    def load_train(self):
        """Returns the train data."""
        return self._load_set(self.training_path)

    def load_test(self):
        """Returns the test data."""
        return self._load_set(self.test_path)

    def load_validate(self):
        """Returns the validation data."""
        return self._load_set(self.validation_path)

    @property
    def configuration(self):
        """Returns a dict-like representation of the configuration of this loader.

        This is for storing its state in the database.
        """
        return {
            'training_path': self.training_path,
            'test_path': self.test_path,
            'validation_path': self.validation_path,
        }

    def _load_set(self, set_path):
        data = pickle.load(open(set_path, "rb"))

        X = data.values[:, 2:]
        y = data.values[:, 1]

        # TODO: Remove workaround
        if not self.mlb_fitted:
            y = self.mlb.fit_transform(y)
            self.mlb_fitted = True
        else:
            y = self.mlb.transform(y)

        return X, y

