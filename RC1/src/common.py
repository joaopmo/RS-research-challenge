import numpy as np
import math


def get_ratings(ratings_path):
    acc = 0
    ratings_dict = {}
    item_by_user = {}
    with open(ratings_path, 'r') as stream:
        next(stream)
        while line := stream.readline().strip():
            ui, r = line.split(',')
            u, i = ui.split(':')
            if u in item_by_user:
                item_by_user[u].add(i)
            else:
                item_by_user[u] = {i}

            ratings_dict[ui] = float(r)
            acc += float(r)

    mean = acc / len(ratings_dict)
    return ratings_dict, item_by_user, mean


def get_factors(ratings_dict, k):
    user_factor = {}
    item_factor = {}
    user_bias = {}
    item_bias = {}
    yj = {}
    for ids in ratings_dict.keys():
        user, item = ids.split(':')
        user_factor[user] = np.random.uniform(-0.01, 0.01, k)
        item_factor[item] = np.random.uniform(-0.01, 0.01, k)
        yj[item] = np.random.uniform(-0.01, 0.01, k)
        user_bias[user] = 0.0
        item_bias[item] = 0.0

    return user_factor, item_factor, user_bias, item_bias, yj
