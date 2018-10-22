import argparse
import logging
import os
import sys

from config import Config
from src.data.dataset import FaceDataset

logging.config.fileConfig(Config.LOGGING_FILE)
logger = logging.getLogger()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input",
                        help="Path to the input folder images.")
    parser.add_argument("-o", "--output", dest="output",
                        help="Path to the output folder.")

    parser.add_argument("-t", "--faces", dest="test_size", type= float,
                        help="The size of the faces set out of the input dataset.", default=0.20)
    parser.add_argument("-u", "--unknown", dest="unknown_size", type=float,
                        help="The size of the unknown faces set out of the input dataset.", default=0.00)
    parser.add_argument("-c", "--copy_images", action="store_true",
                        help="State if the images should be copy in the output folder.")

    args = parser.parse_args()
    logger.info(sys.argv)

    all_data = FaceDataset()
    all_data.read_dataset(args.input)

    if args.unknown_size > 0.0:
        unknown, dataset = all_data.split_classes(k=args.unknown_size)
        trainset, testset = dataset.split_instances_of_classes(test_size=args.test_size)
    else:
        trainset, testset = all_data.split_instances_of_classes(test_size=args.test_size)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    trainset.save(os.path.join(args.output, Config.TRAINSET_FILE))
    testset.save(os.path.join(args.output, Config.TESTSET_FILE))
    if args.unknown_size:
        unknown.save(os.path.join(args.output, Config.NEWDATA_FILE))

    if args.copy_images:
        trainset.copy_files(os.path.join(args.output, Config.TRAIN_DIR))
        testset.copy_files(os.path.join(args.output, Config.TEST_DIR))
        unknown.copy_files(os.path.join(args.output, Config.NEWDATA_DIR))