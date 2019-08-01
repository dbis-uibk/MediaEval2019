import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, ELU, GRU, Input, MaxPooling2D,
                                     Reshape, ZeroPadding2D)
from tensorflow.keras.models import Model


class CRNNModel(BaseEstimator, ClassifierMixin):

    def __init__(self, batch_size=64, epochs=100):
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y):
        input_shape = (96, 1366, 1)
        output_shape = y.shape[1]
        self._create_model(input_shape, output_shape)

        X = X.reshape(X.shape[0], *input_shape)
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)

    def _create_model(self, input_shape, output_shape):
        channel_axis = 3
        freq_axis = 1

        melgram_input = Input(shape=input_shape, dtype="float32")

        # Input block
        hidden = ZeroPadding2D(padding=(0, 37))(melgram_input)
        hidden = BatchNormalization(axis=freq_axis, name='bn_0_freq')(hidden)

        # Conv block 1
        hidden = Conv2D(64, (3, 3), padding='same', name='conv1')(hidden)
        hidden = BatchNormalization(axis=channel_axis, name='bn1')(hidden)
        hidden = ELU()(hidden)
        hidden = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                              name='pool1')(hidden)
        hidden = Dropout(0.1, name='dropout1')(hidden)

        # Conv block 2
        hidden = Conv2D(128, (3, 3), padding='same', name='conv2')(hidden)
        hidden = BatchNormalization(axis=channel_axis, name='bn2')(hidden)
        hidden = ELU()(hidden)
        hidden = MaxPooling2D(pool_size=(3, 3), strides=(3, 3),
                              name='pool2')(hidden)
        hidden = Dropout(0.1, name='dropout2')(hidden)

        # Conv block 3
        hidden = Conv2D(128, (3, 3), padding='same', name='conv3')(hidden)
        hidden = BatchNormalization(axis=channel_axis, name='bn3')(hidden)
        hidden = ELU()(hidden)
        hidden = MaxPooling2D(pool_size=(4, 4), strides=(4, 4),
                              name='pool3')(hidden)
        hidden = Dropout(0.1, name='dropout3')(hidden)

        # Conv block 4
        hidden = Conv2D(128, (3, 3), padding='same', name='conv4')(hidden)
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
        hidden = Dropout(0.3)(hidden)
        output = Dense(output_shape, activation='sigmoid',
                       name='output')(hidden)

        self.model = Model(inputs=melgram_input, outputs=output)
        self.model.compile(optimizer="adam",
                           loss="binary_crossentropy",
                           metrics=['accuracy'])
        self.model.summary()

    def predict(self, X):
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        predictions = self.model.predict(X)
        labels = np.zeros(predictions.shape)
        labels[predictions > 0.5] = 1

        return labels

    def predict_proba(self, X):
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        return self.model.predict(X)
