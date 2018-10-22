import argparse
import logging
import os
import shutil
import sys
import time
from collections import Counter
from enum import Enum
from shutil import copy

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from config import Config
from src.data.dataset import FaceDataset

logging.config.fileConfig(Config.LOGGING_FILE)
logger = logging.getLogger()


class Clustering(Enum):
    hierarchical = 'hierarchical'
    dbscan = 'dbscan'

    def __str__(self):
        return self.value


def analyze_clusters(classes, clusters):
    """
    Prints a DataFrame with the distribution of the instances in clusters.
    :param classes:
    :param clusters:
    :return:
    """
    a = []
    freq_classes = Counter(classes)
    freq_clusters = Counter(clusters)

    for index_class in freq_classes.keys():
        elem_class = classes == index_class
        elem_cluster = clusters[elem_class]
        splits = np.unique(elem_cluster)
        for index_split in splits:
            df_class = index_class
            df_freq_class = freq_classes[index_class]
            df_index_split = index_split
            df_homogenity = len(elem_cluster[elem_cluster == index_split]) / freq_clusters[index_split]
            df_coverage = len(elem_cluster[elem_cluster == index_split]) / freq_classes[index_class]
            a.append([df_class, df_freq_class, df_index_split, df_homogenity, df_coverage])

    df = pd.DataFrame(a, columns=['class', 'freq_class', 'cluster', 'homogenity', 'coverage'])

    df.sort_values(by=['freq_class', 'class', 'index_split'], inplace=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


def score_clusters(labels, classes):
    unique_labels = np.unique(labels)

    logger.info('Number of clusters: {:d}'.format(len(unique_labels)))
    logger.info("Homogeneity: {:0.3f}".format(metrics.homogeneity_score(classes, labels)))
    logger.info("Completeness: {:0.3f}".format(metrics.completeness_score(classes, labels)))
    logger.info("V-measure: {:0.3f}".format(metrics.v_measure_score(classes, labels)))
    logger.info("Adjusted Rand Index: {:0.3f}".format(metrics.adjusted_rand_score(classes, labels)))
    logger.info("Adjusted Mutual Information: {:0.3f}".format(metrics.adjusted_mutual_info_score(classes, labels)))


def get_dbscan():
    return DBSCAN(eps=0.7, min_samples=2, metric="euclidean")


def get_hierarchical(n_clusters):
    return AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean')


def write_clusters(output, data, labels):
    """Copies the images in different clusters."""
    if not os.path.exists(output):
        os.mkdir(output)

    paths = data.get_paths()
    cluster_names = {}
    for label in np.unique(labels):
        cluster_name = "Cluster {}".format(label + 1) if label != -1 else "Unclustered images"
        cluster_names[label] = os.path.join(output, cluster_name)

        if os.path.exists(cluster_names[label]):
            shutil.rmtree(cluster_names[label])
        os.mkdir(cluster_names[label])

    for label, (emb_path, png_path) in zip(labels, paths):
        copy(emb_path, os.path.join(cluster_names[label], emb_path.split(os.path.sep)[-1]))
        copy(png_path, os.path.join(cluster_names[label], png_path.split(os.path.sep)[-1]))


def clustering(data, clustering, analyze=False):
    """
    Cluster the images.
    :param data:
    :param clustering:
    :param analyze:
    :return:
    """
    embeddings, cls = data.get_embeddings()

    clustering.fit(embeddings)

    score_clusters(clustering.labels_, cls)

    if analyze:
        analyze_clusters(cls, clustering.labels_)

    return clustering.labels_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Path to the input folder.")
    parser.add_argument("-o", "--output", dest="output", help="Path to the output folder.")
    parser.add_argument("-n", "--n_clusters", type=int,
                        help="Number of clusters for hierarchical clustering.")
    parser.add_argument("--analyze", action='store_true',
                        help="Print clustering details.")

    args = parser.parse_args()
    logger.info(sys.argv)

    all_data = FaceDataset()
    all_data.read_dataset(args.input)

    clustering_algo = get_dbscan() if not hasattr(args, 'n_clusters') or not args.n_clusters else get_hierarchical(args.n_clusters)

    time_start = time.time()
    labels = clustering(all_data,clustering_algo, args.analyze)
    time_end = time.time()

    logger.info('The clustering took {:.3f} ms'.format( (time_start - time_end) * 1000.0))

    write_clusters(args.output, all_data, labels)
