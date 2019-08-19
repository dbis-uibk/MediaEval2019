from random import randrange

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

cache = {'X': None, 'y': None}


def cached_model_predict(model, X):
    if not np.array_equal(cache['X'], X):
        cache['X'] = X
        cache['y'] = model.predict(X)
    return cache['y']


def cached_model_predict_clear():
    cache['X'] = None
    cache['y'] = None


def find_elbow(x_values, y_values):
    origin = (x_values[0], y_values[0])
    baseline_vec = np.subtract((x_values[-1], y_values[-1]), origin)
    baseline_vec = (-baseline_vec[1], baseline_vec[0])  # rotate 90 degree
    baseline_vec = normalize(np.array(baseline_vec).reshape(1, -1))[0]

    idx = -1
    max_distance = 0
    for i, point in enumerate(zip(x_values, y_values)):
        point_vec = np.subtract(point, origin)
        distance = abs(np.dot(point_vec, baseline_vec))
        max_distance = max(max_distance, distance)

        if max_distance == distance:
            idx = i

    return idx


def load_set_info(path):
    SEP = '\t'
    data = []
    with open(path, 'r') as lines:
        headers = None

        for line in lines:
            fields = line.strip('\n').split(SEP)
            if headers is None:
                headers = fields
            else:
                tag_idx = len(headers) - 1
                current = fields[:tag_idx]
                current.append(fields[tag_idx:])
                data.append(current)

    return pd.DataFrame(data, columns=headers)


def get_windows(sample, window, window_size, num_windows):
    windows = []
    for i in range(num_windows):
        if window == 'center':
            start_idx = int((sample.shape[1] - window_size) / 2)
        elif window == 'random':
            start_idx = randrange(sample.shape[1] - window_size)
        elif window == 'sliding':
            step = int((sample.shape[1] - window_size) / num_windows)
            start_idx = step * i
        else:
            raise ValueError('Unknown window type.')

        end_idx = start_idx + window_size
        windows.append(sample[:, start_idx:end_idx])

    return windows
