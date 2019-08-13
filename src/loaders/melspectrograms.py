from os import path

from dbispipeline.base import TrainValidateTestLoader
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from . import utils


class MelSpectrogramsLoader(TrainValidateTestLoader):
    """
    Loads the mel-spectrograms provided by the task organizers and the labels
    for both the training and test set.
    """

    def __init__(self,
                 data_path,
                 training_path,
                 test_path,
                 validate_path=None,
                 window='center',
                 window_size=1366,
                 num_windows=1):
        self.training_path = training_path
        self.test_path = test_path
        self.validate_path = validate_path
        self.data_path = data_path
        self.window = window
        self.window_size = window_size
        self.num_windows = num_windows
        self.mlb = MultiLabelBinarizer()
        self.mlb_fit = True

    def _load_set(self, set_path):
        sample_set = utils.load_set_info(set_path)[['PATH', 'TAGS']]
        X, y = self._load_data(sample_set)

        # TODO: Remove workaround
        if self.mlb_fit:
            y = self.mlb.fit_transform(y)
            self.mlb_fit = False
        else:
            y = self.mlb.transform(y)

        return X, y

    def _load_data(self, sample_set):
        X = []
        y = []

        for _, sample in sample_set.iterrows():
            sample_path = sample['PATH'].replace('.mp3', '.npy')
            sample_data = np.load(path.join(self.data_path, sample_path))

            sample_data = utils.get_windows(sample=sample_data,
                                            window=self.window,
                                            window_size=self.window_size,
                                            num_windows=self.num_windows)
            X.extend(sample_data)
            y.extend([sample['TAGS']] * self.num_windows)

        return np.array(X), y

    def load_train(self):
        """Returns the train data."""
        return self._load_set(self.training_path)

    def load_validate(self):
        """Returns the validate data."""
        if self.validate_path:
            return self._load_set(self.validate_path)
        else:
            raise NotImplementedError()

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
            'validate_path': self.validate_path,
            'test_path': self.test_path,
            'data_path': self.data_path,
            'window': self.window,
            'window_size': self.window_size,
            'num_windows': self.num_windows,
        }
