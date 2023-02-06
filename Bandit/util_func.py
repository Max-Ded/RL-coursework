import numpy as np
import random


def rolling_sum_numpy(a:np.array):
    return np.apply_along_axis(lambda index : a[:index].sum(),0,list(range(a.size)))

def bernoulli(mean,*args):
    return 1 if random.random() < mean else 0
