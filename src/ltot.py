import numpy as np
from scipy import stats, optimize
from matplotlib import pyplot as plt
from FitSpline import load_ian
import argparse


log_ens = np.linspace(1,6,51) # log (base 10) of the energy values used for fitting
n_E = len(log_ens) # Number of energy levels used for fitting


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fitting script')
    parser.add_argument('particles', nargs='+')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show fitted mu and sigmas')

    args = parser.parse_args()

    def lmu(x, t0, t1, t2, t3):
        return t3*x**3 + t2*x**2 + t1*x + t0

    def lsg(x, alpha, beta):
        return np.log(alpha) + beta * x

    for particle in args.particles:
        Dat = load_ian(particle, f'DataOutputs_{particle}')
        energy_strs = list(Dat.keys())
        mus = np.zeros(n_E)
        sgs = np.zeros(n_E)
        for i in range(n_E):
            df = Dat[energy_strs[i]]
            _mu, _sig = stats.norm.fit(df['ltot'])
            mus[i] = _mu
            sgs[i] = _sig
        lmu_fit = optimize.curve_fit(lmu, log_ens, np.log(mus))
        lsg_fit = optimize.curve_fit(lsg, log_ens, np.log(sgs))
        print(lmu_fit[0])
        print(lsg_fit[0])
        if args.show:
            plt.plot(log_ens, np.log(mus), 'bo', label='mu')
            plt.plot(log_ens, lmu(log_ens, *lmu_fit[0]), 'b-')
            plt.plot(log_ens, np.log(sgs), 'ro', label='sigma')
            plt.plot(log_ens, lsg(log_ens, *lsg_fit[0]), 'r-')
            plt.show()
