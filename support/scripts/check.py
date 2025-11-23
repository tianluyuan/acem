# coding: utf-8
import numpy as np
import scipy as sp
import sys
import Sample, FitSpline
from matplotlib import pyplot as plt

x = np.load(sys.argv[1])
en = float(sys.argv[2])
abx = Sample.sample_ab(x, en, 1000, ga_inv=lambda a: a, gb_inv=lambda b: b)
xs = np.arange(0, 1, 0.01)
X, Y = np.meshgrid(xs, xs)
knots = FitSpline.knots
print('a_k:', knots[0])
print('b_k:', knots[1])
print('E_k:', knots[2])

theta = np.load(f'theta_{sys.argv[1]}')
bsp = sp.interpolate.NdBSpline(knots, theta, 3)

for Z in [bsp(np.asarray([X, Y, np.ones_like(X)*np.log10(en)]).T).T,
          FitSpline.Eval_from_Coefs(X, Y, np.log10(en), x, knots)]:
    plt.clf();plt.pcolormesh(X, Y, np.exp(Z))
    plt.colorbar()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel("a'")
    plt.ylabel("b'")
    plt.scatter(abx[0], abx[1], s=0.2, c='r')

    plt.figure()
    plt.pcolormesh(X, Y, Z)
    plt.colorbar()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.scatter(abx[0], abx[1], s=0.2, c='r')
    plt.show()

a = np.arange(0., 1., 0.001)
plt.clf()
plt.plot(a, FitSpline.Eval_from_Coefs(a, 0.3, np.log10(en), x, knots))
[plt.axvline(_, linestyle='--', linewidth=0.5) for _ in knots[0]]
plt.xlabel('a\'')
plt.ylabel('spline value')
plt.show()

b = np.arange(0., 1., 0.001)
plt.clf()
plt.plot(b, FitSpline.Eval_from_Coefs(0.3, b, np.log10(en), x, knots))
[plt.axvline(_, linestyle='--', linewidth=0.5) for _ in knots[0]]
plt.xlabel('b\'')
plt.ylabel('spline value')
plt.show()
