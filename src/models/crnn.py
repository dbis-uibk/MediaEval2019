from sklearn.base import BaseEstimator, ClassifierMixin
from keras.models import Model
from keras.layers import Input, GRU, Dense, Conv2D, MaxPooling2D, Reshape
import numpy as np


class CRNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self, batch_size=64, epochs=100, num_filters=100):
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_filters = num_filters


    def fit(self, X, y):
        ########## Reshape data ##########
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        ########## Build the model ##########
        model_input = Input(shape=(96, 1366, 1), dtype="float32")

        cnn1 = Conv2D(self.num_filters, (3, 3))(model_input)
        pool1 = MaxPooling2D((2, 2))(cnn1)

        cnn2 = Conv2D(self.num_filters, (3, 3))(pool1)
        pool2 = MaxPooling2D((3, 3))(cnn2)

        cnn3 = Conv2D(self.num_filters, (3, 3))(pool2)
        pool3 = MaxPooling2D((4, 4))(cnn3)

        cnn4 = Conv2D(self.num_filters, (3, 3))(pool3)
        rsh1 = Reshape((54, self.num_filters))(cnn4)

        rnn1 = GRU(self.num_filters, return_sequences=True)(rsh1)
        rnn2 = GRU(self.num_filters)(rnn1)

        output = Dense(y.shape[1], activation="softmax")(rnn2)

        self.model = Model(inputs=model_input, outputs=output)
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        self.model.summary()

        ########## Train the model ##########
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)


    def predict(self, X):
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        predictions = self.model.predict(X)
        labels = np.zeros(predictions.shape)
        labels[predictions > 0.5] = 1

        return labels

    def predict_proba(self, X):
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        return self.model.predict(X)
