import logging.config
import os
import pickle
import shutil

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from config import Config

logging.config.fileConfig(Config.LOGGING_FILE)
logger = logging.getLogger()


def grid_search(clf, params, x, y):
    grid = GridSearchCV(clf, param_grid=params, cv=3, scoring='accuracy')

    grid.fit(x, y)

    logger.info('Grid search: best score = {score}, best params = {params}, estimator = {estim}'.format(
        score=grid.best_score_, params=grid.best_params_, estim=grid.best_estimator_))

    return grid.best_params_


def is_similar(embedding, images=None, threshold=None):
    """
    Given an embedding and a set of images, computes if the embedding is similar with the images.
    :param embedding:
    :param images:
    :param threshold: which is the maximal euclidian distance for two vectors to be considered similar.
    :return:
    """
    total_dist = 0
    counter = 0
    for image in images:
        dist = np.linalg.norm(image - embedding)
        total_dist += dist
        counter += 1

    dist = total_dist / counter
    if dist > threshold:
        return False
    else:
        return True


class KnnModel:

    def __init__(self):
        self.model = None

    def train(self, k, x_train, y_train, x_test, y_test):
        logger.info("Start training the model")
        self.model = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree')
        self.model.fit(x_train, y_train)

        self.evaluate(x_train, y_train)
        self.evaluate(x_test, y_test)

    def predict_class(self, image, trainset=None, threshold=None):
        """
        Predicts the class of an unlabeled image. If the trainset and threshold arguments are provided,
        it also verifies that the image is similar with the images from training set that belong to the predicted class.
        :param image:
        :param trainset:
        :param threshold:
        :return:
        """
        pred_cls = self.model.predict([image.embedding])[0]
        pred_cls = pred_cls if is_similar(image, trainset[pred_cls], threshold) else Config.UNKNOWN_TAG
        return pred_cls

    def tune(self, x, y):
        """
        Searched for the best k for the model.
        :param x_train:
        :param y_train:
        :return:
        """
        self.model = KNeighborsClassifier(n_neighbors=5)

        k_range = list(range(3, 20))
        params = {'n_neighbors': k_range}

        params = grid_search(self.model, params, x, y)
        best_k = params['n_neighbors']

        logger.info('Best K ={}'.format(best_k))
        return best_k

    def save(self, output_dir):
        """
        Serializes the model and the source file on the disk.
        :param output_dir:
        :return:
        """
        with open(os.path.join(output_dir, Config.KNN_MODEL_FILE), 'wb') as fout:
            pickle.dump(self, fout)

        shutil.copy(os.path.realpath(__file__), output_dir)

        logger.info("Model saved in {}".format(output_dir))

    @staticmethod
    def load(input_dir):
        """
        Loads the model from the disk.
        :param input_dir:
        :return:
        """
        logger.info("Load the model from {}".format(input_dir))
        with open(os.path.join(input_dir, Config.KNN_MODEL_FILE), 'rb') as fin:
            return pickle.load(fin)

    def evaluate(self, x, y):
        """Computed the accuracy of the model."""
        y_pred = self.model.predict(x)
        logger.info('Accuracy: {score:.4f}'
                    .format(score=accuracy_score(y, y_pred)))

        return accuracy_score(y, y_pred)
