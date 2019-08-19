import pickle

from dbispipeline.base import TrainValidateTestLoader
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


class AcousticBrainzLoader(TrainValidateTestLoader):
    """
    Loads the AcousticBrainz features provided by the task organizers and the
    labels for both the training and test set.
    """

    def __init__(self,
                 training_path,
                 test_path,
                 validation_path,
                 num_windows=1):
        self.training_path = training_path
        self.test_path = test_path
        self.validation_path = validation_path
        self.num_windows = num_windows

        self.mlb = MultiLabelBinarizer()
        self.mlb_fitted = False

        self.columns = [
            "#ID",
            "#tags",
            "lowlevel.average_loudness",
            "highlevel.danceability.all.danceable",
            "highlevel.genre_tzanetakis.all.blu",
            "highlevel.genre_tzanetakis.all.cla",
            "highlevel.genre_tzanetakis.all.cou",
            "highlevel.genre_tzanetakis.all.dis",
            "highlevel.genre_tzanetakis.all.hip",
            "highlevel.genre_tzanetakis.all.jaz",
            "highlevel.genre_tzanetakis.all.met",
            "highlevel.genre_tzanetakis.all.pop",
            "highlevel.genre_tzanetakis.all.reg",
            "highlevel.genre_tzanetakis.all.roc",
            "highlevel.mood_acoustic.all.acoustic",
            "highlevel.mood_aggressive.all.aggressive",
            "highlevel.mood_electronic.all.electronic",
            "highlevel.mood_happy.all.happy",
            "highlevel.mood_party.all.party",
            "highlevel.mood_relaxed.all.relaxed",
            "highlevel.mood_sad.all.sad",
            "highlevel.timbre.all.bright",
            "highlevel.timbre.all.dark",
            "highlevel.tonal_atonal.all.tonal",
        ]

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
        # Unpickle data.
        data = pickle.load(open(set_path, "rb"))
        data = data[self.columns]

        # Split features and labels.
        X = data.drop(columns=['#ID', '#tags']).to_numpy()
        y = data['#tags'].to_numpy()

        # Duplicate data num_windows times.
        X = np.repeat(X, repeats=self.num_windows, axis=0)
        y = np.repeat(y, repeats=self.num_windows, axis=0)

        # TODO: Remove workaround
        if not self.mlb_fitted:
            y = self.mlb.fit_transform(y)
            self.mlb_fitted = True
        else:
            y = self.mlb.transform(y)

        return X, y
