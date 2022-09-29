import numpy as np
import random
import math
import time

from common import get_factors


def sgd(s, Ru, mean, start, k=15, alpha=0.007, reg_f=0.02, reg_b=0.02):
    user_f, item_f, user_b, item_b, yj = get_factors(s, k)

    while time.time() - start < 60 * 4.25:
        ratings = list(s.items())
        random.shuffle(ratings)
        for ui, r in ratings:
            u, i = ui.split(':')
            p, q = user_f[u], item_f[i]
            sqrt_Ru = math.sqrt(len(Ru[u]))
            implicit_fb = np.sum([yj[j] for j in Ru[u]], axis=0) / sqrt_Ru

            dot = np.dot(q, p + implicit_fb)
            r_pred = mean + user_b[u] + item_b[i] + dot
            err = r - r_pred

            user_b[u] += alpha * (err - reg_b * user_b[u])
            item_b[i] += alpha * (err - reg_b * item_b[i])
            p_new = p + alpha * (err * q - reg_f * p)
            q_new = q + alpha * (err * (p + implicit_fb) - reg_f * q)
            err_q_sqrt = err * q / sqrt_Ru
            for j in Ru[u]:
                yj[j] += alpha * (err_q_sqrt - reg_f * yj[j])
            user_f[u], item_f[i] = p_new, q_new

    return user_f, item_f, user_b, item_b, yj


def pred(targets_path, Ru, mean, user_f, item_f, user_b, item_b, yj):
    target_dict = {}
    with open(targets_path, 'r') as stream:
        next(stream)
        while line := stream.readline().strip():
            u, i = line.split(':')
            p, q = user_f[u], item_f[i]
            sqrt_Ru = math.sqrt(len(Ru[u]))
            implicit_fb = np.sum([yj[j] for j in Ru[u]], axis=0) / sqrt_Ru
            dot = np.dot(q, p + implicit_fb)
            r = mean + user_b[u] + item_b[i] + dot
            r = round(r)
            target_dict[line] = 1 if r < 1 else 5 if r > 5 else r

    with open("output.csv", "w") as stream:
        # print("UserId:ItemId,Rating")
        stream.write("UserId:ItemId,Rating\n")
        for key, val in target_dict.items():
            # print(f"{key},{val}")
            stream.write(f"{key},{val}\n")
