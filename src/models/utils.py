from functools import lru_cache

import numpy as np
from sklearn.preprocessing import normalize


@lru_cache(maxsize=1)
def cached_model_predict(model, X):
    return model.predict(X)


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
