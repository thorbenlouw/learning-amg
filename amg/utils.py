import json
import shutil
import os
import glob
from functools import lru_cache

import tensorflow as tf
import numpy as np
from collections import Counter
from scipy.spatial.qhull import Delaunay
from oct2py import octave


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def most_frequent_splitting(splittings):
    """Given a list of numpy array, returns the most frequent one"""
    list_of_tuples = [tuple(splitting) for splitting in splittings]  # we need a list of immutable types
    counter = Counter(list_of_tuples)
    most_frequent_tuple = counter.most_common(1)[0][0]
    return np.array(most_frequent_tuple)


def create_results_dir(run_name):
    results_dir = 'results/' + run_name
    os.makedirs(results_dir)

    # make a copy of all Python files, for reproducibility
    local_dir = os.path.dirname(__file__)
    for py_file in glob.glob(local_dir + '/*.py'):
        shutil.copy(py_file, results_dir)


def write_config_file(run_name, config, seed):
    results_dir = 'results/' + run_name
    config_dict = {'train_config': config.train_config.__dict__,
                   'data_config': config.data_config.__dict__,
                   'model_config': config.model_config.__dict__,
                   'run_config': config.run_config.__dict__,
                   'seed': seed}
    with open(f'{results_dir}/config.json', 'w') as outfile:
        json.dump(config_dict, outfile)


@lru_cache(maxsize=None)
def tril_indices(grid_size):
    # tril_indices returns the indices of the lower-triangular elemss in a grid_size x grid_size matrix
    """Cached version of np.tril_indices used for creating relaxation matrices"""
    return np.tril_indices(grid_size)


def get_accelerator_device():
    gpu_devices = tf.config.list_logical_devices('GPU')
    if len(gpu_devices) > 0:
        return tf.device(gpu_devices[0].name)
    return tf.device(tf.config.list_logical_devices('CPU')[0].name)


def generate_delaunay_triangulation(k):
    x = np.random.rand(k, 1)
    y = np.random.rand(k, 1)
    X = np.concatenate((
        x - 1,
        x - 1,
        x - 1,
        x,
        x,
        x,
        x + 1,
        x + 1,
        x + 1))
    Y = np.concatenate((
        y - 1,
        y,
        y + 1,
        y - 1,
        y,
        y + 1,
        y - 1,
        y,
        y + 1))
    points = np.dstack((X, Y)).reshape((-1, 2))
    tri = Delaunay(points)
    return tri


def init_octave(seed):
    octave.eval(f'rand("seed", {seed})')
    octave.eval(f'randn("seed", {seed})')
    octave.eval('pkg load statistics')
    octave.addpath(os.curdir + os.pathsep + 'amg')
    return octave
