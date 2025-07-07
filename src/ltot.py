import numpy as np
from scipy import stats, optimize
from FitSpline import load_ian
import argparse


log_ens = np.linspace(1,6,51) # log (base 10) of the energy values used for fitting
n_E = len(log_ens) # Number of energy levels used for fitting


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fitting script')
    parser.add_argument('particles', nargs='+')

    args = parser.parse_args()

    def mu(x, t0, t1, t2, t3):
        return np.exp(t3*x**3 + t2*x**2 + t1*x + t0)

    def sigma(x, alpha, beta):
        return alpha * np.exp(x)**beta

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
        mures = optimize.curve_fit(mu, log_ens, mus)
        sgres = optimize.curve_fit(sigma, log_ens, sgs)
        print(mures)
        print(sgres)
