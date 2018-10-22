import argparse
import datetime
import logging
import os
import shutil
import sys

from config import Config
from src.data.dataset import FaceDataset
from src.model.knn_model import KnnModel

logging.config.fileConfig(Config.LOGGING_FILE)
logger = logging.getLogger()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Path to the input folder.")
    parser.add_argument("-t", "--tune", dest="tune", action='store_true', help="State if the model should be tuned.")
    parser.add_argument("-k", "--k", dest="k", default=3, type=int, help="Path to the input folder.")

    args = parser.parse_args()
    logger.info(sys.argv)

    trainset = FaceDataset.load(os.path.join(args.input, Config.TRAINSET_FILE))
    testset = FaceDataset.load(os.path.join(args.input, Config.TESTSET_FILE))

    x_train, y_train = trainset.get_embeddings()
    x_test, y_test = testset.get_embeddings()

    knnmodel = KnnModel()

    if args.tune:
        k = knnmodel.tune(x_train, y_train)
    else:
        k = args.k

    knnmodel.train(k=k, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    output_dir = os.path.join(Config.SAVED_MODELS, datetime.datetime.now().strftime('%Y-%m-%d'))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    knnmodel.save(output_dir)
    shutil.copyfile(os.path.join(args.input, Config.TRAINSET_FILE), os.path.join(output_dir, Config.TRAINSET_FILE))
