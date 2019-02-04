import random


def split_data(data, split_pct=.1):
    random.shuffle(data)  # make sure data is shuffled
    split = int(len(data) * split_pct)
    return data[split:], data[:split]  # training, test
