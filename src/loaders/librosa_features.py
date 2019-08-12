import pickle

from dbispipeline.base import TrainTestLoader
from sklearn.preprocessing import MultiLabelBinarizer


class LibRosaLoader(TrainTestLoader):
    """
    Loads librosa features and labels for both the training and test set.
    """

    def __init__(self, training_path, test_path):
        self.training_path = training_path
        self.test_path = test_path
        self.mlb = None

    def _load_set(self, set_path):
        X = []
        y = []

        # Load pickled data frame.
        frame = pickle.load(open(set_path, "rb"))
        X = frame[frame.columns[:-1]].values
        y = frame["tags"].values

        # Binarize labels.
        if self.mlb is None:
            self.mlb = MultiLabelBinarizer()
            y = self.mlb.fit_transform(y)
        else:
            y = self.mlb.transform(y)

        return X, y

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
