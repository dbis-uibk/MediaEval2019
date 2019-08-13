from random import randrange

import pandas as pd


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
            step = (sample.shape[1] - window_size) / num_windows
            start_idx = step * i
        else:
            raise ValueError('Unknown window type.')

        end_idx = start_idx + window_size
        windows.append(sample[:, start_idx:end_idx])

    return windows
