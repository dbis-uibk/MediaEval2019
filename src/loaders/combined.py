import numpy as np
from dbispipeline.base import TrainValidateTestLoader

from .acousticbrainz import AcousticBrainzLoader
from .melspectrograms import MelSpectrogramsLoader


class CombinedLoader(TrainValidateTestLoader):

    def __init__(self, mel_data_path, mel_training_path, mel_validate_path,
                 mel_test_path, ess_training_path, ess_validate_path,
                 ess_test_path):
        self.mel_loader = MelSpectrogramsLoader(
            data_path=mel_data_path,
            training_path=mel_training_path,
            validate_path=mel_validate_path,
            test_path=mel_test_path)
        self.ess_loader = AcousticBrainzLoader(
            training_path=ess_training_path,
            validation_path=ess_validate_path,
            test_path=ess_test_path)

    def load_train(self):
        """Returns the train data."""
        X_mel, y = self.mel_loader.load_train()
        X_ess, _ = self.ess_loader.load_train()
        X = np.array(list(zip(X_mel, X_ess.to_numpy())))
        return X, y

    def load_validate(self):
        """Returns the validate data."""
        X_mel, y = self.mel_loader.load_validate()
        X_ess, _ = self.ess_loader.load_validate()
        X = np.array(list(zip(X_mel, X_ess.to_numpy())))
        return X, y

    def load_test(self):
        """Returns the test data."""
        X_mel, y = self.mel_loader.load_test()
        X_ess, _ = self.ess_loader.load_test()
        X = np.array(list(zip(X_mel, X_ess.to_numpy())))
        return X, y

    @property
    def configuration(self):
        """Returns a dict-like representation of the configuration of this loader.

        This is for storing its state in the database.
        """
        return {
            'melspectrogram_loader': self.mel_loader.configuration,
            'essentia_loader': self.ess_loader.configuration,
        }
