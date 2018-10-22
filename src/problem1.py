import argparse
import logging
import os
import sys
from enum import Enum

from config import Config
from src.data.dataset import FaceDataset
from src.model.knn_model import KnnModel
from src.model.sequencial_model import SequentialModel

logging.config.fileConfig(Config.LOGGING_FILE)
logger = logging.getLogger()


class Models(Enum):
    KNN = 'knn'
    SEQUENTIAL = 'seq'

    def __str__(self):
        return self.value


def score(sequencial_model, knn_model, testset, trainset, model_type=Models.KNN):
    """Combines a sequential model and a KNN model to classify the images from a test set."""
    correct = 0
    total = 0

    for test_cls, test_images in testset.items():
        for image in test_images:
            total += 1
            if model_type == Models.SEQUENTIAL:
                cls_pred = sequencial_model.predict_class(image, trainset)
                if not cls_pred and knn_model:
                    cls_pred = knn_model.predict_class(image)[0]
            else:
                cls_pred = knn_model.predict_class(image, trainset, threshold=0.90)

            correct += (test_cls == cls_pred)

    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Path to the input folder.")
    parser.add_argument("-m", "--model", dest="model_dir", help="Path to the model folder.")
    parser.add_argument("-a", "--approach", dest="approach", type=Models, choices=list(Models), default=Models.SEQUENTIAL,
                        help="Path to the input model.")

    args = parser.parse_args()
    logger.info(sys.argv)

    trainset = FaceDataset.load(os.path.join(args.model_dir, Config.TRAINSET_FILE))
    testset = FaceDataset.load(os.path.join(args.input, Config.TESTSET_FILE))

    sequential_model = None
    if args.approach == Models.SEQUENTIAL:
        logger.info("Load the sequencial model.")
        sequential_model = SequentialModel()
        sequential_model.load(args.model_dir)

    logger.info("Load the knn model.")
    knn_model = KnnModel().load(args.model_dir)

    logger.info("Evaluate the model on the test set.")
    acc = score(sequential_model, knn_model, testset, trainset, args.approach)
    logger.info('Accuracy of the classifier on test set: {score:.4f}'.format(score=acc))
