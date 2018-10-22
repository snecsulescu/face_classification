import argparse
import logging.config
import os
import sys

from config import Config
from src.data.dataset import FaceDataset
from src.model.knn_model import KnnModel
from src.model.sequencial_model import SequentialModel
from src.problem1 import Models

logging.config.fileConfig('logconfig.ini')
logger = logging.getLogger()


def score(model, testset, trainset):
    """Uses a sequential model or a KNN model to classify the images from a test set."""
    correct = 0
    total = 0

    for test_cls, test_images in testset.items():
        for image in test_images:
            total += 1
            if isinstance(model, KnnModel):
                cls_pred = model.predict_class(image, trainset, threshold=0.90)
            else:
                cls_pred = model.predict_class(image, trainset, threshold=0.5)

            correct += (test_cls == cls_pred)

    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Path to the input folder.")
    parser.add_argument("-m", "--model", dest="model_dir", help="Path to the input model.")
    parser.add_argument("-a", "--approach", dest="approach", type=Models, choices=list(Models), default=Models.SEQUENTIAL,
                        help="Path to the input model.")

    args = parser.parse_args()
    logger.info(sys.argv)

    trainset = FaceDataset.load(os.path.join(args.model_dir, Config.TRAINSET_FILE))
    testset = FaceDataset.load(os.path.join(args.input, Config.TESTSET_FILE))
    newpeopleset = FaceDataset.load(os.path.join(args.input, Config.NEWDATA_FILE))
    newpeopleset = newpeopleset.create_unknown_class()

    if args.approach == Models.SEQUENTIAL:
        logger.info("Load the sequencial model.")
        model = SequentialModel()
        model.load(args.model_dir)
    else:
        logger.info("Load the knn model.")
        model = KnnModel().load(args.model_dir)

    logger.info("Evaluate the model on the test set.")
    acc = score(model, testset, trainset)
    logger.info('Accuracy of the classifier on test set: {score:.4f}'.format(score=acc))

    logger.info("Evaluate the model on a set containing only unknown people.")
    acc = score(model, newpeopleset, trainset)
    logger.info('Accuracy of the classifier on new faces: {score:.4f}'.format(score=acc))
