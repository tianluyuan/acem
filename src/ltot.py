import numpy as np
from scipy import stats, optimize
from matplotlib import pyplot as plt
import argparse
plt.style.use('present')


log_ens = np.linspace(1,6,51) # log (base 10) of the energy values used for fitting
n_E = len(log_ens) # Number of energy levels used for fitting
fluka_bin_volume = 1000 * 1000 * 10


def ply(x, t0, t1, t2, t3):
    return t3*x**3 + t2*x**2 + t1*x + t0


def pwl(x, alpha, beta):
    return np.log(alpha) + beta * x


if __name__ == '__main__':
    from FitSpline import load_ian
    parser = argparse.ArgumentParser('Fitting script')
    parser.add_argument('particles', nargs='+')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show fitted pars')
    parser.add_argument('--savefig', default=None, type=str,
                        help='Save fitted plots to specified directory')
    parser.add_argument('--sshow', action='store_true', default=False,
                        help='Show distributions')
    parser.add_argument('--ssavefig', default=None, type=str,
                        help='Save distributions to specified directory')

    args = parser.parse_args()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    def custom_optimizer(func, x0, args=(), disp=0):
        res = optimize.minimize(func, x0, args, method="Powell",
                                options={"disp": disp})
        if res.success:
            return res.x
        raise RuntimeError('optimization routine failed')

    for particle in args.particles:
        Dat = load_ian(particle, f'DataOutputs_{particle}')
        energy_strs = list(Dat.keys())
        results = []
        if particle in ['ELECTRON', 'PHOTON']:
            form = stats.loggamma
            # c (shape), loc, scale
            p_fn = [ply, pwl, ply]
        else:
            form = stats.norm
            # pwl for loc (mean), ply for scale (sigma)
            p_fn = [pwl, ply]

        for i in range(n_E):
            df = Dat[energy_strs[i]]
            ltots = df['ltot'] * fluka_bin_volume
            _res = form.fit(ltots)
            results.append(_res)
            if args.sshow or args.ssavefig:
                plt.clf()
                bins = np.linspace(ltots.min() * 0.9, ltots.max() * 1.1, 50)
                plt.hist(ltots, bins=bins, histtype='step')
                plt.hist(
                    form.rvs(*_res, size=len(ltots)),
                    bins=bins, histtype='step')
                plt.yscale('log')
                if args.sshow:
                    plt.show()
                if args.ssavefig:
                    plt.savefig(f'{args.ssavefig}/ltot_dist_{particle}_{energy_strs[i]}.pdf')

        results = np.asarray(results)
        par_fits = [optimize.curve_fit(_f, log_ens, np.log(_y))[0] for _f, _y in zip(p_fn, results.T)]

        if args.show or args.savefig:
            plt.clf()
            for i, (_f, _y, _p) in enumerate(zip(p_fn, results.T, par_fits)):
                plt.plot(log_ens, _y, 'o', color=colors[i], label=f'p{i}')
                plt.plot(log_ens, np.exp(_f(log_ens, *_p)), color=colors[i])
            plt.yscale('log')
            plt.legend()
            if args.show:
                plt.show()
            if args.savefig:
                plt.savefig(f'{args.savefig}/ltot_{particle}.pdf')
                plt.savefig(f'{args.savefig}/ltot_{particle}.png')

        np.savez(f'ltot_{particle}.npz', **{f'p{_i}': _par for _i, _par in enumerate(par_fits)})
