import numpy as np


def safe_divide(a, b):
    return np.where(b <= 2, 0, a / b)
