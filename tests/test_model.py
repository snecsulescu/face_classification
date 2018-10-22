import datetime
import logging
import os
import unittest

import numpy as np

from config import Config
from src.data.dataset import FaceDataset
from src.model.knn_model import KnnModel
from src.model.sequencial_model import SequentialModel
from src.problem2 import score

logging.config.fileConfig(Config.LOGGING_FILE)
logger = logging.getLogger()


class TestModel(unittest.TestCase):
    """
    Test the sequential model and the knn model.
    """

    def setUp(self):
        data1 = FaceDataset()
        data1.read_dataset('tests/fixtures/faces/same_faces')

        data2 = FaceDataset()
        data2.read_dataset('tests/fixtures/faces/newdata')

        data1.save(os.path.join('tests/fixtures/faces/same_faces/testset.pkl'))
        data2.save(os.path.join('tests/fixtures/faces/newdata/testset.pkl'))

    def test_sequencial_model(self):
        dataset = FaceDataset.load('tests/fixtures/faces/same_faces/testset.pkl')
        X, y = dataset.get_distance_vectors()

        modeldir = os.path.join(Config.SAVED_MODELS, datetime.datetime.now().strftime('%Y-%m-%d'))

        kmodel = SequentialModel()
        kmodel.load(modeldir)

        score = kmodel.evaluate(X, y)
        self.assertGreater(score, 0.8)

    def test_sequencial_model_newdata(self):
        dataset = FaceDataset.load('tests/fixtures/faces/newdata/testset.pkl')
        dataset = dataset.create_unknown_class()

        modeldir = os.path.join(Config.SAVED_MODELS, datetime.datetime.now().strftime('%Y-%m-%d'))
        trainset = FaceDataset.load(os.path.join(modeldir, Config.TRAINSET_FILE))

        seqmodel = SequentialModel()
        seqmodel.load(modeldir)

        results = score(seqmodel, dataset, trainset)
        self.assertGreater(results, 0.8)

    def test_knn_model(self):
        """
        Tests the data loading from a json file.
        :return:
        """
        dataset = FaceDataset.load('tests/fixtures/faces/same_faces/testset.pkl')
        modeldir = os.path.join(Config.SAVED_MODELS, datetime.datetime.now().strftime('%Y-%m-%d'))

        knnmodel = KnnModel.load(modeldir)

        X, y = dataset.get_embeddings()
        score = knnmodel.evaluate(X, y)
        self.assertGreater(score, 0.8)

    def test_knn_model_newdata(self):
        """

        :return:
        """
        dataset = FaceDataset.load('tests/fixtures/faces/newdata/testset.pkl')
        dataset = dataset.create_unknown_class()

        modeldir = os.path.join(Config.SAVED_MODELS, datetime.datetime.now().strftime('%Y-%m-%d'))
        trainset = FaceDataset.load(os.path.join(modeldir, Config.TRAINSET_FILE))

        knnmodel = KnnModel.load(modeldir)

        result = score(knnmodel, dataset, trainset)
        self.assertGreater(result, 0.8)

    def test_0(self):
        modeldir = os.path.join(Config.SAVED_MODELS, datetime.datetime.now().strftime('%Y-%m-%d'))

        kmodel = SequentialModel()
        kmodel.load(modeldir)

        X, y = np.zeros((1, 512)), np.array([1])
        score = kmodel.evaluate(X, y)
        self.assertGreater(score, 0.8)


if __name__ == '__main__':
    unittest.main()
