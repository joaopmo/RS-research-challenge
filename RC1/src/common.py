import numpy as np
import math


def get_ratings(ratings_path, centered=False):
    acc = 0
    ratings_dict = {}
    with open(ratings_path, 'r') as stream:
        next(stream)
        while line := stream.readline().strip():
            key, val = line.split(',')
            ratings_dict[key] = float(val)
            acc += float(val)

    mean = acc / len(ratings_dict)

    if centered:
        for k, r in ratings_dict.items():
            ratings_dict[k] = r - mean

    return ratings_dict, mean


def get_factors(ratings_dict, k):
    user_factor = {}
    item_factor = {}
    user_bias = {}
    item_bias = {}
    for ids in ratings_dict.keys():
        user, item = ids.split(':')
        user_factor[user] = np.random.uniform(-0.01, 0.01, k)
        item_factor[item] = np.random.uniform(-0.01, 0.01, k)
        user_bias[user] = np.random.uniform(-0.01, 0.01)
        item_bias[item] = np.random.uniform(-0.01, 0.01)

    return user_factor, item_factor, user_bias, item_bias
