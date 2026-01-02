#!/usr/bin/env python
# coding: utf-8
import numpy as np
import sys
from matplotlib import pyplot as plt
from acem.model import Parametrization1D, RWParametrization1D
from acem import util
from acem.pdg import PDG2FLUKA


if __name__=='__main__':
    logE = float(sys.argv[1])
    try:
        seed = int(sys.argv[2])
    except IndexError:
        seed = None
    rng = np.random.default_rng(seed)
    curr = Parametrization1D(Parametrization1D.FLUKA_MEDIUM, random_state=rng)
    prev = RWParametrization1D(Parametrization1D.FLUKA_MEDIUM, random_state=rng)

    for pdg in curr.THETAS.keys():
        try:
            df = util.load_csv(
                f"fluka/DataOutputs_{PDG2FLUKA[pdg]}/{PDG2FLUKA[pdg]}_{util.format_energy(10**logE)}.csv", False)
            bins = np.linspace(df["ltot"].min(), df["ltot"].max(), 100).tolist()
            _, _, lines = plt.hist(
                df["ltot"],
                bins=bins,
                density=True,
                histtype="step",
            )
            xs = np.linspace(bins[0], bins[-1], 1000)
            plt.plot(xs,
                     curr.ltot_dist(pdg, 10**logE).pdf(xs), "--",
                     color=lines[0].get_edgecolor(), label=f"{pdg}")
        except FileNotFoundError:
            pass
    plt.legend()
    plt.xlabel("ltot [cm]")
    plt.ylabel("Density [1/cm]")
    plt.yscale("log")
    plt.show()
        
    N = 100
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    X, Y = np.meshgrid(x, y)

    for pdg in curr.THETAS.keys():
        Z= curr.THETAS[pdg](X, Y, logE)

        plt.figure(figsize=(8, 6))
        plot = plt.pcolormesh(X, Y, np.exp(Z), cmap='viridis', shading='auto')

        cbar = plt.colorbar(plot, ax=plt.gca(), label='PDF')

        plt.plot(*curr.THETAS[pdg].sample(logE,40,random_state=rng).T,
                 'k.',
                 label='samples (via rejection)',
                 markersize=1.5)
        # plt.plot(*curr.THETAS[pdg]._igrid_sample(logE,40, num_quad_nodes=10, random_state=rng).T,
        #          'r.',
        #          label='Binary sampling',
        #          markersize=1.5)
        plt.plot(*curr.THETAS[pdg].mode(logE), 'r*', label='mode')
        plt.plot(*curr.THETAS[pdg].mean(logE), 'b*', label='mean')
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
        Z0 = B._legacy_eval(X, Y, logE)
        Z1 = B(X, Y, logE)

        plt.figure()
        plt.pcolormesh(X, Y, (np.exp(Z0)-np.exp(Z1)) / np.exp(Z1))
        plt.colorbar()
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel("a'")
        plt.ylabel("b'")
        plt.title(f"{pdg} (exp(B)-exp(NdB)) / exp(NdB)")

        plt.figure()
        plt.pcolormesh(X, Y, (Z0-Z1) / Z1)
        plt.colorbar()
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel("a'")
        plt.ylabel("b'")
        plt.title(f"{pdg} (B-NdB) / NdB")
    
        ap = np.arange(0., 1., 0.001)
        plt.figure()
        plt.plot(ap, B(ap, 0.3, logE))
        [plt.axvline(_, linestyle='--', linewidth=0.5) for _ in B.bspl.t[0]]
        plt.xlabel("a'")
        plt.ylabel('spline value')
        plt.title(f"{pdg}")

        bp = np.arange(0., 1., 0.001)
        plt.figure()
        plt.plot(bp, B(0.3, bp, logE))
        [plt.axvline(_, linestyle='--', linewidth=0.5) for _ in B.bspl.t[1]]
        plt.ylabel("b'")
        plt.ylabel('spline value')
        plt.title(f"{pdg}")
    plt.show()
