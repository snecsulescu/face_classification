import argparse
import datetime
import logging
import os
import shutil
import sys

from config import Config
from src.data.dataset import FaceDataset, Balance
from src.model.sequencial_model import SequentialModel

logging.config.fileConfig(Config.LOGGING_FILE)
logger = logging.getLogger()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Path to the input folder.")
    parser.add_argument("-b", "--balance", dest="balance", type=Balance, choices=list(Balance),
                        help="Approach for class balancing.")
    args = parser.parse_args()
    logger.info(sys.argv)

    trainset = FaceDataset.load(os.path.join(args.input, Config.TRAINSET_FILE))
    testset = FaceDataset.load(os.path.join(args.input, Config.TESTSET_FILE))

    x_train, y_train = trainset.get_distance_vectors(balance=args.balance)
    x_test, y_test = testset.get_distance_vectors()

    kmodel = SequentialModel()
    kmodel.train(input_size=x_train.shape[1], x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    output_dir = os.path.join(Config.SAVED_MODELS, datetime.datetime.now().strftime('%Y-%m-%d'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    kmodel.save(output_dir)
    shutil.copyfile(os.path.join(args.input, Config.TRAINSET_FILE), os.path.join(output_dir, Config.TRAINSET_FILE))
