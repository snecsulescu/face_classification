import logging
import os
import unittest

import numpy as np

from config import Config
from src.data import dataset
from src.data.dataset import FaceDataset, Balance, FaceImageClass

logging.config.fileConfig(Config.LOGGING_FILE)
logger = logging.getLogger()


class TestModel(unittest.TestCase):
    """
    Tests for the dataset.
    """
    def setUp(self):
        data1 = FaceDataset()
        data1.read_dataset('tests/fixtures/faces/same_faces')

        data2 = FaceDataset()
        data2.read_dataset('tests/fixtures/faces/newdata')

        data1.append(data2)
        data1.save(os.path.join('tests/fixtures/faces/testset.pkl'))

    def test_read_dataset(self):
        data = FaceDataset()
        data.read_dataset('tests/fixtures/faces/same_faces')
        self.assertEqual(data.classes_length, 150)
        self.assertEqual(data['George_W_Bush'].length, 26)

    def test_save(self):
        data = FaceDataset()
        data.read_dataset('tests/fixtures/faces/newdata')

        data.save('tests/fixtures/faces/newdata/newdata.pkl')
        self.assertTrue(os.path.exists('tests/fixtures/faces/newdata/newdata.pkl'))

    def test_load(self):
        data = FaceDataset()
        data.read_dataset('tests/fixtures/faces/newdata')
        data.save('tests/fixtures/faces/newdata/newdata.pkl')
        new_data = FaceDataset.load('tests/fixtures/faces/newdata/newdata.pkl')
        self.assertEqual(data.classes_length, new_data.classes_length)
        for cls in data.get_classes():
            self.assertTrue(cls in new_data.get_classes())
            self.assertEqual(data[cls].length, new_data[cls].length)

    def test_balance_upsample(self):
        data = FaceDataset.load('tests/fixtures/faces/testset.pkl')
        X, y = data.get_distance_vectors()
        X_bal, y_bal = dataset.balance_classes(X, y, Balance.UPSAMPLE)
        self.assertEqual(y_bal[y_bal == 0].shape, y_bal[y_bal == 1].shape)
        self.assertEqual(y_bal.shape[0], y[y == 0].shape[0] * 2)

    def test_balance_downsample(self):
        data = FaceDataset.load('tests/fixtures/faces/testset.pkl')
        X, y = data.get_distance_vectors()
        X_bal, y_bal = dataset.balance_classes(X, y, Balance.DOWNSAMPLE)
        self.assertEqual(y_bal[y_bal == 0].shape, y_bal[y_bal == 1].shape)
        self.assertEqual(y_bal.shape[0], y[y == 1].shape[0] * 2)

    def test_add_imageclass(self):
        data = FaceDataset.load('tests/fixtures/faces/testset.pkl')
        init_value = data.classes_length
        images = FaceImageClass('new_class')
        images.append(np.zeros((3, 512)))
        data._add_imageclass(images)
        self.assertEqual(data.classes_length, init_value + 1)

    def test_split_instances_of_classes(self):
        data = FaceDataset()
        data.read_dataset('tests/fixtures/faces/newdata')
        data1, data2 = data.split_instances_of_classes()

        self.assertEqual(data1.classes_length, data2.classes_length)
        for cls, images in data.items():
            self.assertEqual(data[cls].length, data1[cls].length + data2[cls].length)

    def test_split_classes(self):
        data = FaceDataset()
        data.read_dataset('tests/fixtures/faces/same_faces')
        data1, data2 = data.split_classes()

        self.assertEqual(data.classes_length, data1.classes_length + data2.classes_length)

    def test_get_embeddings(self):
        data = FaceDataset()
        data.read_dataset('tests/fixtures/faces/same_faces')
        X, y = data.get_embeddings()

        self.assertTrue(isinstance(X, np.ndarray))
        self.assertEqual(X.shape[0], y.shape[0])

    def test_unknown_dataset(self):
        data = FaceDataset()
        data.read_dataset('tests/fixtures/faces/same_faces')
        unk_data = data.create_unknown_class()
        self.assertEqual(unk_data.classes_length, 1)
        self.assertTrue(Config.UNKNOWN_TAG in unk_data.dataset)

        s = 0
        for _, images in unk_data.items():
            s += images.length

        self.assertEqual(unk_data[Config.UNKNOWN_TAG].length, s)
