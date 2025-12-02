import numpy as np

def efn(en, fn, *args):
    return np.exp(fn(np.log10(en), *args))


def qrt(x, t0, t1, t2, t3, t4):
    return t4*x**4 + t3*x**3 + t2*x**2 + t1*x + t0


def cbc(x, t0, t1, t2, t3):
    return t3*x**3 + t2*x**2 + t1*x + t0


def lin(x, t0, t1):
    return t1 * x + t0
