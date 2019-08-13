import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_curve
from sklearn.preprocessing import normalize
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, ELU, GRU, Input, MaxPooling2D,
                                     Reshape, ZeroPadding2D)
from tensorflow.keras.models import Model

from .utils import cached_model_predict


class CRNNModel(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 batch_size=64,
                 epochs=100,
                 padding='same',
                 dataloader=None,
                 output_dropout=0.3):
        self.batch_size = batch_size
        self.epochs = epochs
        self.padding = padding
        self.dataloader = dataloader
        self.output_dropout = output_dropout

    def fit(self, X, y):
        input_shape = (96, 1366, 1)
        output_shape = y.shape[1]
        self._create_model(input_shape, output_shape)

        X = X.reshape(X.shape[0], *input_shape)
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)
        cached_model_predict.clear_cache()

        if self.dataloader:
            try:
                validation_data = self.dataloader.load_validate()
            except (NotImplementedError, AttributeError):
                validation_data = None

            if validation_data:
                self.validate(*validation_data)
            else:
                self.threshold(np.full(output_shape, .5))

    def validate(self, X, y):
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        y_pred = self.model.predict(X)
        threshold = []
        for label_idx in range(y_pred.shape[1]):
            fpr, tpr, thresholds = roc_curve(y[..., label_idx],
                                             y_pred[..., label_idx])
            try:
                idx = find_elbow(tpr, fpr)
            except ValueError as ex:
                print(ex)
                idx = -1

            if idx >= 0:
                threshold.append(thresholds[idx])
            else:
                threshold.append(0.5)

        self.threshold = np.array(threshold)

    def _create_model(self, input_shape, output_shape):
        channel_axis = 3

        melgram_input = Input(shape=input_shape, dtype="float32")

        # Input block
        hidden = ZeroPadding2D(padding=(0, 37))(melgram_input)

        # Conv block 1
        hidden = Conv2D(64, (3, 3), padding=self.padding, name='conv1')(hidden)
        hidden = BatchNormalization(axis=channel_axis, name='bn1')(hidden)
        hidden = ELU()(hidden)
        hidden = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                              name='pool1')(hidden)
        hidden = Dropout(0.1, name='dropout1')(hidden)

        # Conv block 2
        hidden = Conv2D(128, (3, 3), padding=self.padding,
                        name='conv2')(hidden)
        hidden = BatchNormalization(axis=channel_axis, name='bn2')(hidden)
        hidden = ELU()(hidden)
        hidden = MaxPooling2D(pool_size=(3, 3), strides=(3, 3),
                              name='pool2')(hidden)
        hidden = Dropout(0.1, name='dropout2')(hidden)

        # Conv block 3
        hidden = Conv2D(128, (3, 3), padding=self.padding,
                        name='conv3')(hidden)
        hidden = BatchNormalization(axis=channel_axis, name='bn3')(hidden)
        hidden = ELU()(hidden)
        hidden = MaxPooling2D(pool_size=(4, 4), strides=(4, 4),
                              name='pool3')(hidden)
        hidden = Dropout(0.1, name='dropout3')(hidden)

        # Conv block 4
        hidden = Conv2D(128, (3, 3), padding=self.padding,
                        name='conv4')(hidden)
        hidden = BatchNormalization(axis=channel_axis, name='bn4')(hidden)
        hidden = ELU()(hidden)
        hidden = MaxPooling2D(pool_size=(4, 4), strides=(4, 4),
                              name='pool4')(hidden)
        hidden = Dropout(0.1, name='dropout4')(hidden)

        # reshaping
        hidden = Reshape((15, 128))(hidden)

        # GRU block 1, 2, output
        hidden = GRU(32, return_sequences=True, name='gru1')(hidden)
        hidden = GRU(32, return_sequences=False, name='gru2')(hidden)
        if self.output_dropout:
            hidden = Dropout(self.output_dropout)(hidden)
        output = Dense(output_shape, activation='sigmoid',
                       name='output')(hidden)

        self.model = Model(inputs=melgram_input, outputs=output)
        self.model.compile(optimizer="adam",
                           loss="binary_crossentropy",
                           metrics=['accuracy'])
        self.model.summary()

    def predict(self, X):
        predictions = self.predict_proba(X)
        labels = np.greater(predictions, self.threshold)

        return labels

    def predict_proba(self, X):
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        return cached_model_predict(self.model, X)


def find_elbow(x_values, y_values):
    origin = (x_values[0], y_values[0])
    baseline_vec = np.subtract((x_values[-1], y_values[-1]), origin)
    baseline_vec = (-baseline_vec[1], baseline_vec[0])  # rotate 90 degree
    baseline_vec = normalize(np.array(baseline_vec).reshape(1, -1))[0]

    idx = -1
    max_distance = 0
    for i, point in enumerate(zip(x_values, y_values)):
        point_vec = np.subtract(point, origin)
        distance = abs(np.dot(point_vec, baseline_vec))
        max_distance = max(max_distance, distance)

        if max_distance == distance:
            idx = i

    return idx
