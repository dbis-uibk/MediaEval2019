from dbispipeline.base import TrainTestLoader
from os import path
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


class MelSpectrogramsLoader(TrainTestLoader):
    """
    Loads the mel-spectrograms provided by the task organizers and the labels 
    for both the training and test set.
    """

    def __init__(self, training_path, test_path, data_path, center_sample=True):
        self.training_path = training_path
        self.test_path = test_path
        self.data_path = data_path
        self.center_sample = center_sample
        self.mlb = None

    def _load_set(self, set_path):
        X = []
        y = []

        # Process every song in the given set.
        with open(set_path, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                fields = line.split("\t")

                npy_path = fields[3].replace(".mp3", ".npy")
                tags = [t.replace("\n", "") for t in fields[5:]]
                
                X_temp = np.load(path.join(self.data_path, npy_path))
                start_idx = int(X_temp.shape[1] / 2 - 683)
                X_temp = X_temp[:, start_idx:(start_idx + 1366)]

                X.append(X_temp)
                y.append(tags)

        # Binarize labels.
        if self.mlb is None:
            self.mlb = MultiLabelBinarizer()
            y = self.mlb.fit_transform(y)
        else:
            y = self.mlb.transform(y)

        return np.array(X), np.array(y)

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
        return {}
