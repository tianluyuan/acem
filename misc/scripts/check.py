# coding: utf-8
import numpy as np
import scipy as sp
import sys
from matplotlib import pyplot as plt
from shosim.model import Parametrization1D, RWParametrization1D
from shosim import media


if __name__=='__main__':
    logE = float(sys.argv[1])
    try:
        seed = int(sys.argv[2])
    except IndexError:
        seed = 12
    rng = np.random.default_rng(seed)
    curr = Parametrization1D(media.IC3)
    prev = RWParametrization1D(media.IC3)

    N = 100
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    X, Y = np.meshgrid(x, y)

    for pdg in curr.THETAS.keys():
        Z= curr.THETAS[pdg](X, Y, logE)

        plt.figure(figsize=(8, 6))
        plot = plt.pcolormesh(X, Y, np.exp(Z), cmap='viridis', shading='auto')

        cbar = plt.colorbar(plot, ax=plt.gca(), label='PDF')

        plt.plot(*curr.THETAS[pdg].sample_ab(logE,40,random_state=rng).T, 'k.', label='Rejection sampling')
        plt.plot(*curr.THETAS[pdg]._legacy_sample_ab(logE,40, num_quad_nodes=10, random_state=rng).T, 'r.', label='Binary sampling')
        plt.legend()
        plt.title(f"{pdg}")
        plt.xlabel("a'")
        plt.ylabel("b'")

    plt.show()

    xs=np.arange(0, 3000)
    for pdg in [11, 22, 211]:
        plt.figure()
        [plt.plot(xs, _.dldx(xs), color='k', linewidth=0.5) for _ in curr.sample(pdg, 10**logE, 10)]
        [plt.plot(xs, _.dldx(xs), color='r', linewidth=0.5, linestyle='--') for _ in prev.sample(pdg, 10**logE, 10)]
    plt.xlim(0, 2000)
    plt.xlabel('x [cm]')
    plt.ylabel('dl/dx')
    plt.title(f"{pdg}")
    plt.show()

    for pdg, B in curr.THETAS.items():
        bsp = sp.interpolate.NdBSpline(B.knots, B.coefs, 3)

        Z0 = B(X, Y, logE)
        Z1 = bsp(np.asarray([X, Y, np.ones_like(X)*logE]).T).T

        plt.figure()
        plt.pcolormesh(X, Y, np.exp(Z0)-np.exp(Z1))
        plt.colorbar()
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel("a'")
        plt.ylabel("b'")
        plt.title(f"{pdg} exp(B)-exp(NdB)")

        plt.figure()
        plt.pcolormesh(X, Y, Z0-Z1)
        plt.colorbar()
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel("a'")
        plt.ylabel("b'")
        plt.title(f"{pdg} B-NdB")
    
        ap = np.arange(0., 1., 0.001)
        plt.figure()
        plt.plot(ap, B(ap, 0.3, logE))
        [plt.axvline(_, linestyle='--', linewidth=0.5) for _ in B.knots[0]]
        plt.xlabel('a\'')
        plt.ylabel('spline value')
        plt.title(f"{pdg}")

        bp = np.arange(0., 1., 0.001)
        plt.figure()
        plt.plot(bp, B(0.3, bp, logE))
        [plt.axvline(_, linestyle='--', linewidth=0.5) for _ in B.knots[1]]
        plt.xlabel('b\'')
        plt.ylabel('spline value')
        plt.title(f"{pdg}")
    plt.show()
