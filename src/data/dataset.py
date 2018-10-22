import logging.config
import os
import pickle
import random
import shutil
from enum import Enum
from random import shuffle

import numpy as np
import sklearn.utils

from config import Config

logging.config.fileConfig(Config.LOGGING_FILE)
logger = logging.getLogger()


class Balance(Enum):
    DOWNSAMPLE = 'downsample'
    UPSAMPLE = 'upsample'


def downsample(x, y, k):
    rand_idx = sklearn.utils.resample(range(0, x.shape[0]), replace=False, n_samples=k)

    return x[rand_idx], y[rand_idx]


def upsample(x, y, k):
    rand_idx = sklearn.utils.resample(range(0, x.shape[0]), replace=True, n_samples=k)

    return x[rand_idx], y[rand_idx]


def balance_classes(x, y, downsampling):
    """
    It balances the classes in x and y. I suppose the pos class has fewer instances and the neg class has more, 
    behaviour that should be improved. 
    :param y:
    :param x:
    :param downsampling: parameter that shows which approach to use for the class imbalance problem:
    downsample or upsample.  
    :return: 
    """

    pos = y == 1
    x_pos, y_pos = x[pos], y[pos]

    neg = y == 0
    x_neg, y_neg = x[neg], y[neg]

    if downsampling == Balance.DOWNSAMPLE:
        x_neg, y_neg = downsample(x_neg, y_neg, x_pos.shape[0])

        x, y = np.concatenate([x_pos, x_neg]), np.concatenate([y_pos, y_neg])
    else:
        x_add, y_add = upsample(x_pos, y_pos, x_neg.shape[0] - x_pos.shape[0])
        x, y = np.concatenate([x_pos, x_add, x_neg]), np.concatenate([y_pos, y_add, y_neg])

    return x, y

def split_array_randomly(arr, k_test):
    random.shuffle(arr)

    return arr[:-k_test], arr[-k_test:]

class NotProperTypeException(Exception):
    def __init__(self, wrong_type, correct_type):
        Exception.__init__(self, "The object that you send is of type {wrong} instead of {correct}."
                           .format(wrong=wrong_type, correct=correct_type))


class FaceImage:
    """
    Class representing an image.
    """
    def __init__(self, embedding_path, png_path, embedding):
        self.embedding_path = embedding_path
        self.png_path = png_path
        # self.org_embedding = embedding
        self.embedding = embedding

    def __sub__(self, other):
        return self.embedding - other.embedding


class FaceImageClass:
    """
    Class holding images from the same class.
    """
    def __init__(self, name):
        self.images = []
        self.name = name

    def __getitem__(self, index):
        return self.images[index]

    def __iter__(self):
        return iter(self.images)

    def append(self, img):
        self.images.append(img)

    def extend(self, imgs):
        self.images.extend(imgs)

    @property
    def length(self):
        return len(self.images)

    def embeddings(self):
        embeddings = []
        for image in self.images:
            embeddings.append(image.embedding)
        return embeddings

    def get_paths(self):
        paths = []
        for image in self.images:
            paths.append((image.embedding_path, image.png_path))
        return paths


class FaceDataset:
    """Class for all the images data set."""
    def __init__(self):
        self.dataset = {}

    def __getitem__(self, key):
        return self.dataset[key]

    def append(self, other):
        if not isinstance(other, FaceDataset):
            raise NotProperTypeException(wrong_type=type(other), correct_type=FaceDataset)
        for cls, images in other.items():
            if cls not in self.dataset:
                self.dataset[cls] = images
            else:
                self.dataset[cls].extend(images)

    @property
    def classes_length(self):
        return len(self.dataset.keys())

    def get_classes(self):
        return self.dataset.keys()

    def items(self):
        return self.dataset.items()

    def _add_imageclass(self, face_imageclass):
        if not isinstance(face_imageclass, FaceImageClass):
            raise NotProperTypeException(wrong_type=type(face_imageclass), correct_type=FaceDataset)

        if face_imageclass.name not in self.dataset:
            self.dataset[face_imageclass.name] = face_imageclass
        else:
            self.dataset[face_imageclass.name].extend(face_imageclass)

    def append_image(self, class_name, image):
        if not isinstance(image, FaceImage):
            raise NotProperTypeException(wrong_type=type(image), correct_type=FaceDataset)

        if class_name not in self.dataset:
            self.dataset[class_name] = FaceImageClass(class_name)
        self.dataset[class_name].append(image)

    def read_dataset(self, path):
        logger.info("Read data set from {}".format(path))
        for folder in os.listdir(path):
            if os.path.isdir(os.path.join(path, folder)):
                images = FaceImageClass(folder)
                for file in os.listdir(os.path.join(path, folder)):
                    if file.endswith('.npy'):
                        embedding = np.load(os.path.join(path, folder, file))
                        png_file = file[:-3] + "png"
                        image = FaceImage(embedding_path=os.path.join(path, folder, file),
                                          png_path=os.path.join(path, folder, png_file),
                                          embedding=embedding)
                        images.append(image)

                if images.length > 0:
                    self._add_imageclass(images)

    def split_classes(self, k=0.20):
        """
        Splits the set of classes in two subsets.
        :param k: represent the proportion of the dataset to include in the first split
        :return: two FaceDataset instances
        """
        k_dataset = FaceDataset()
        other_dataset = FaceDataset()

        k = max(round(self.classes_length * k), 1)

        keys = list(self.dataset.keys())

        random.shuffle(keys)

        for key in keys[:k]:
            k_dataset._add_imageclass(self.dataset[key])
        for key in keys[k:]:
            other_dataset._add_imageclass(self.dataset[key])

        return k_dataset, other_dataset

    def split_instances_of_classes(self, test_size=0.20):
        """
        Splits the dataset in two subsets, each subset containing different instances from each class.
        :param test_size: represent the proportion of each class instances to be include in the last split
        :return: two FaceDataset instances
        """
        train_data = FaceDataset()
        test_data = FaceDataset()

        for cls, images in self.dataset.items():
            k_test = max(round(test_size * images.length), 1)

            indices = [x for x in range(0, images.length)]
            train_cls, test_cls = split_array_randomly(indices, k_test)

            for idx, img in enumerate(images):
                if idx in train_cls:
                    train_data.append_image(cls, img)
                else:
                    test_data.append_image(cls, img)

        return train_data, test_data

    def _positive_instances(self):
        """Creates distance vectors of pairs of images from the same class.
        :return: an ndarray a distance per line and the labels
        """
        x, y = [], []

        for cls in self.dataset:
            for img1 in self.dataset[cls]:
                for img2 in self.dataset[cls]:
                    if not (np.equal(img1, img2).all()):
                        x.append(img1 - img2)
                        y.append(1)

        if x:
            x.append(np.zeros(x[0].shape[0]))
            y.append(1)
        return x, y

    def _negative_instances(self):
        """Creates distance vectors of pairs of images from different classes.
        :return: an ndarray a distance per line and the labels
        """
        x, y = [], []

        for cls1 in self.dataset:
            for cls2 in self.dataset:
                if cls1 != cls2:
                    for img1, img2 in zip(self.dataset[cls1], self.dataset[cls2]):
                        x.append(img1 - img2)
                        y.append(0)
                        x.append(img2 - img1)
                        y.append(0)

        return x, y

    def get_embeddings(self):
        """ Computes all the embeddings with the corresponding classes from the dataset.
        :return: 
        """
        x, y = [], []
        for cls, images in self.dataset.items():
            x.extend(images.embeddings())
            y.extend([cls] * len(images.embeddings()))

        x = np.array(x)
        y = np.array(y)

        logger.info(
            'Dataset dimensions: x shape = {x_shape}, y shape = {y_shape}'.format(x_shape=x.shape, y_shape=y.shape))

        return x, y

    def get_paths(self):
        """
        For each image in the data set it returns a list of pair containing the fisical path to the embedding vectors 
        and the .png images.
        :return: 
        """
        paths = []
        for cls, images in self.dataset.items():
            paths.extend(images.get_paths())

        return paths

    def get_distance_vectors(self, balance=None):
        """
        Computes the distance vectors between all the images from the dataset.
        :param balance: approach to the class imbalance problem
        :return: 
        """
        x, y = self._positive_instances()

        x_neg, y_neg = self._negative_instances()

        x.extend(x_neg)
        y.extend(y_neg)

        x = np.array(x)
        y = np.array(y)

        if balance:
            x, y = balance_classes(x, y, downsampling=(balance == Balance.DOWNSAMPLE))

        idx = [x for x in range(0, x.shape[0])]
        shuffle(idx)
        x = x[idx]
        y = y[idx]

        logger.info(
            'Dataset dimensions: x shape = {x_shape}, y shape = {y_shape}'.format(x_shape=x.shape, y_shape=y.shape))

        return np.array(x), np.array(y)

    def compute_distance_from_embedding(self, embedding):
        """
        Given the embedding of a new image, it returns the distances to each image in the data set.
        :param embedding: 
        :return: 
        """
        if not isinstance(embedding, FaceImage):
            raise NotProperTypeException(wrong_type=type(embedding), correct_type=FaceDataset)

        x, y = [], []
        for cls, items in self.dataset.items():
            for item in items:
                x.append(item - embedding)
                y.append(cls)

        return x, y

    def save(self, filename):
        """
        Serializes the dataset on disk.
        :param filename: 
        :return: 
        """
        with open(filename, 'wb') as fout:
            pickle.dump(self, fout)

    @staticmethod
    def load(filename):
        """loads a dataset from the disk"""
        with open(filename, 'rb') as fin:
            return pickle.load(fin)

    def copy_files(self, dst):
        """
        Copy the files from the data set to the dst folder. 
        :param dst: 
        :return: 
        """
        if not os.path.exists(dst):
            os.makedirs(dst)

        for embedding_path, png_path in self.get_paths():
            short_path, embedding_filename = os.path.split(embedding_path)
            _, png_filename = os.path.split(png_path)
            short_path, class_dir = os.path.split(short_path)

            if not os.path.exists(os.path.join(dst, class_dir)):
                os.mkdir(os.path.join(dst, class_dir))

            shutil.copy(embedding_path, os.path.join(dst, class_dir, embedding_filename))
            shutil.copy(embedding_path, os.path.join(dst, class_dir, png_filename))

    def create_unknown_class(self):
        """
        Eliminates the labels from the dataset.
        :return: 
        """
        unknown_people = FaceDataset()
        for cls, images in self.dataset.items():
            images.name = Config.UNKNOWN_TAG
            unknown_people._add_imageclass(images)
        return unknown_people
