# -*- coding: utf-8 -*-
import numpy as np

class Calc():
    def jaccard(self, x, y):
        x = frozenset(x)
        y = frozenset(y)
        return len(x & y) / float(len(x | y))

    def dice(self, x, y):
        x = frozenset(x)
        y = frozenset(y)
        return 2 * len(x & y) / float(sum(map(len, (x, y))))

    def simpson(self, x, y):
        x = frozenset(x)
        y = frozenset(y)
        return len(x & y) / float(min(map(len, (x, y))))

    def normalization(self, current_x, list_x):
        x_min = min(list_x)
        x_max = max(list_x)
        x_norm = (current_x - x_min) / ( x_max - x_min)
        return x_norm + 0.001

    def kld(self, p, q):
        p = np.array(p)
        q = np.array(q)
        return np.sum(p * np.log1p(p / q), axis=(p.ndim - 1))

    def jsd(self, p, q):
        p = np.array(p)
        q = np.array(q)
        m = 0.5 * (p + q)
        return 0.5 * self.kld(p, m) + 0.5 * self.kld(q, m)

    def value_reverse(self, dictionary):
        listed = [dic for dic in dictionary.values()]
        for key in dictionary:
            dictionary[key] = max(listed) - dictionary[key]
        return dictionary

