import numpy as np
import random
import math
import time

from common import get_factors


def sgd(s, Ru, mean, start, k=20, alpha=0.009, reg_f=0.015, reg_b=0.005):
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
        alpha *= 0.95

    return user_f, item_f, user_b, item_b, yj



