import numpy as np
import random
import math
import time

from common import get_factors


def sgd_bias(s, mean, start, k=20, alpha=0.005, reg=0.02):
    user_f, item_f, user_b, item_b = get_factors(s, k)

    while time.time() - start < 60 * 4.5:
        ratings = list(s.items())
        random.shuffle(ratings)
        for ui, r in ratings:
            u, i = ui.split(':')
            p, q = user_f[u], item_f[i]
            r_pred = mean + user_b[u] + item_b[i] + np.dot(p, q)
            err = r - r_pred
            p_new = p + alpha * (err * q - reg * p)
            q_new = q + alpha * (err * p - reg * q)
            user_b[u] += alpha * (err - reg * user_b[u])
            item_b[i] += alpha * (err - reg * item_b[i])
            user_f[u], item_f[i] = p_new, q_new

        rmse = 0
        for ids, r in ratings:
            u, i = ids.split(':')
            p, q = user_f[u], item_f[i]
            r_pred = mean + user_b[u] + item_b[i] + np.dot(p, q)
            rmse += pow(r - r_pred, 2)
        rmse = rmse / len(ratings)
        rmse = math.sqrt(rmse)
        print(rmse)

    return user_f, item_f, user_b, item_b


def pred_bias(targets_path, mean, user_f, item_f, user_b, item_b):
    target_dict = {}
    with open(targets_path, 'r') as stream:
        next(stream)
        while line := stream.readline().strip():
            u, i = line.split(':')
            p, q = user_f[u], item_f[i]
            r = mean + user_b[u] + item_b[i] + np.dot(p, q)
            r = round(r)
            target_dict[line] = 1 if r < 1 else 5 if r > 5 else r

    with open("output.csv", "w") as stream:
        # print("UserId:ItemId,Rating")
        stream.write("UserId:ItemId,Rating\n")
        for key, val in target_dict.items():
            # print(f"{key},{val}")
            stream.write(f"{key},{val}\n")
