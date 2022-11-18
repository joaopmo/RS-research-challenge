import numpy as np
import math


# Função que lê o arquivo "ratings.csv" e retorna:
# rantings_dict: estrutura de dados que armazena todos os ratings
# item_by_user: estrutura que armazena os items avaliados por cada usuário
# mean: média de todas as avaliações
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


# Função que inicializa as estruturas de dados que representam
# matrizes esparças de fatores ou estruturas que armazenam bias
# de usuários e items
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


# Função que lê o arquivo "target.csv", calcula
# a predição para os pares UserId:ItemId deste arquivo
# e gera um novo arquivo "output.csv" com as predições
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
        print("UserId:ItemId,Rating")
        stream.write("UserId:ItemId,Rating\n")
        for key, val in target_dict.items():
            print(f"{key},{val}")
            stream.write(f"{key},{val}\n")
