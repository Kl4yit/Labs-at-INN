import random


def get_random_values(n):
    r = []
    for i in range(n):
        r.append(round(random.random(), 5))
    return r


def threshold_fun(value):
    if value > 0:
        return 1
    return 0
