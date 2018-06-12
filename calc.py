# -*- coding: utf-8 -*-

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

