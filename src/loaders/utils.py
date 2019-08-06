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
