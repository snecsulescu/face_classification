import datetime
import itertools
import logging.config
import os
import pickle
import shutil
from collections import Counter

from keras.layers import Dense, BatchNormalization, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler

from config import Config

logging.config.fileConfig(Config.LOGGING_FILE)
logger = logging.getLogger()


class SequentialModel:
    def __init__(self):
        self.scaler = None
        self.model = None

    def preprocessing(self, x, y):
        """
        The embeddings must be scaled before traing the Sequential model with them, and the y transformed into an array
        with one-hot vectors on each line. The y preprocessing can be eliminated if the last layer of teh model contains
        only one neuron.
        :param x:
        :param y:
        :return:
        """
        x = self.scaler.transform(x)
        y = to_categorical(y)
        return x, y

    def _model_structure(self, input_dim):
        self.model = Sequential()

        self.model.add(Dense(128, input_dim=input_dim))
        self.model.add(BatchNormalization())
        self.model.add(Activation('tanh'))
        self.model.add(Dense(64))
        self.model.add(BatchNormalization())
        self.model.add(Activation('tanh'))
        self.model.add(Dense(32))
        self.model.add(BatchNormalization())
        self.model.add(Activation('tanh'))
        self.model.add(Dense(2))
        self.model.add(Activation('softmax'))

    def train(self, input_size, x_train, y_train, x_test, y_test):
        self.data = [x_train, y_train]
        self.scaler = StandardScaler()
        self.scaler.fit(x_train)

        x_train, y_train, = self.preprocessing(x_train, y_train)
        x_test, y_test, = self.preprocessing(x_test, y_test)

        self._model_structure(input_size)

        optimizer = Adam(lr=0.001)

        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

    def evaluate(self, x, y):
        """
        Computes the accuracy of the model
        :param x:
        :param y:
        :return:
        """
        logger.info("Start evaluating the model ...")
        x, y = self.preprocessing(x, y)

        scores = self.model.evaluate(x, y)
        logger.info('Accuracy: {score:.4f}'
                    .format(score=scores[1]))
        return scores[1]

    def save(self, output_dir):
        """
        Serializes the scaler, the model and the source file on the disk.

        :param output_dir:
        :return:
        """
        with open(os.path.join(output_dir, Config.SCALER_FILE), 'wb') as fout:
            pickle.dump(self.scaler, fout)
        self.model.save(os.path.join(output_dir, Config.SEQUENCIAL_MODEL_FILE))
        shutil.copy(os.path.realpath(__file__), output_dir)

        logger.info("Model saved in {}".format(output_dir))
        return output_dir

    def load(self, input_dir):
        """
        Loads the model from the disk.
        :param input_dir:
        :return:
        """
        with open(os.path.join(input_dir, Config.SCALER_FILE), 'rb') as fin:
            self.scaler = pickle.load(fin)

        self.model = load_model(os.path.join(input_dir, Config.SEQUENCIAL_MODEL_FILE))

    def _predict(self, x):
        """Predict the classes for an array of distances."""
        x = self.scaler.transform(x)
        y_pred = self.model.predict_classes(x)

        return y_pred

    def predict_class(self, image, dataset, threshold=None):
        """ Predicts the class for an image.
        It first computes the distances from the image to each image in the data set and predicts if they are similar
        or not. it counts which is the most significative class similar with the image.
        """
        x, y = dataset.compute_distance_from_embedding(image)

        y_pred = self._predict(x)
        results = Counter(itertools.compress(y, y_pred))
        for cls, counts in results.items():
            results[cls] = counts / dataset[cls].length

        try:
            argmax = max(results, key=results.get)
            return argmax if argmax and (not threshold or results[argmax] >= threshold) else Config.UNKNOWN_TAG
        except:
            return Config.UNKNOWN_TAG
