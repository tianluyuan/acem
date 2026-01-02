#!/usr/bin/env python
import numpy as np
from scipy import stats, optimize
from matplotlib import pyplot as plt
import argparse
from importlib.resources import files, as_file
from acem.util import load_batch
from acem.maths import sxt
plt.style.use('present')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fitting script')
    parser.add_argument('particles', nargs='+')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show fitted pars')
    parser.add_argument('--savefig', default=None, type=str,
                        help='Save fitted plots to specified directory')
    mxg = parser.add_mutually_exclusive_group()
    mxg.add_argument('--sshow', action='store_true', default=False,
                     help='Show distributions')
    mxg.add_argument('--sshowlogy', action='store_true', default=False,
                     help='Show distributions with yscale log')
    parser.add_argument('--ssavefig', default=None, type=str,
                        help='Save distributions to specified directory')

    args = parser.parse_args()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    def custom_optimizer(func, x0, args=(), disp=0):
        res = optimize.minimize(func, x0, args, method="slsqp",
                                bounds=[(0, np.inf)] * len(x0),
                                options={"disp": disp})
        if res.success:
            return res.x
        raise RuntimeError('optimization routine failed')

    for particle in args.particles:
        if particle in ['ELECTRON', 'PHOTON']:
            form = stats.norminvgauss
            # 2x shape, loc, scale
            p_fn = [sxt, sxt, sxt, sxt]
            sgns = [1, -1, 1, 1]
            clean = False
        else:
            form = stats.skewnorm
            # shape then loc (mean) and scale (sigma)
            p_fn = [sxt, sxt, sxt]
            sgns = [1, 1, 1]
            clean = True  # mask tricky decays

        Dat = load_batch(f'fluka/DataOutputs_{particle}/*.csv', clean=clean)
        ens = list(Dat.keys())
        log_ens = np.log10(ens)
        n_E = len(log_ens)
        results = []

        for i in range(n_E):
            df = Dat[ens[i]]
            ltots = df['ltot']
            _res = form.fit(ltots, method='MLE')
            results.append(_res)
            if args.sshow or args.sshowlogy or args.ssavefig:
                _rvs = form.rvs(*_res, size=len(ltots))
                bins = np.linspace(min(_rvs.min(), ltots.min()) * 0.9,
                                   max(_rvs.max(), ltots.max()) * 1.1, 50)
                # bins = np.logspace(np.log10(min(_rvs.min(), ltots.min()) * 0.9),
                #                    np.log10(max(_rvs.max(), ltots.max()) * 1.1), 50)
                plt.clf()
                plt.hist(ltots, bins=bins, histtype='step', label='FLUKA')
                plt.hist(_rvs, bins=bins, histtype='step', label='Fit')
                plt.xlabel('Total Cerenkov weighted length')
                plt.ylabel('N')
                plt.legend()
                if args.sshow:
                    plt.show()
                if args.sshowlogy:
                    plt.yscale('log')
                    # plt.xscale('log')
                    plt.show()
                if args.ssavefig:
                    plt.yscale('linear')
                    plt.savefig(f'{args.ssavefig}/ltot_dist_{particle}_{ens[i]}.pdf', bbox_inches='tight')
                    plt.yscale('log')
                    plt.savefig(f'{args.ssavefig}/ltot_logdist_{particle}_{ens[i]}.pdf', bbox_inches='tight')

        results = np.asarray(results) * sgns
        _sel = results[:, 0] > 0
        par_fits = [optimize.curve_fit(_f, log_ens[_sel], np.log(_y[_sel]))[0]
                    for _f, _y in zip(p_fn, results.T)]

        if args.show or args.savefig:
            plt.clf()
            for i, (_f, _y, _p) in enumerate(zip(p_fn, results.T, par_fits)):
                plt.plot(log_ens[_sel], _y[_sel], 'o', color=colors[i], label=f'p{i}', markersize=2)
                plt.plot(log_ens[~_sel], _y[~_sel], 'x', color=colors[i], markersize=2)
                plt.plot(log_ens, np.exp(_f(log_ens, *_p)), color=colors[i])
            plt.yscale('log')
            plt.legend()
            plt.xlabel(r'$\log_{10} (E / \mathrm{GeV})$')
            plt.ylabel('Parameter values')
            if args.savefig:
                plt.savefig(f'{args.savefig}/ltot_{particle}.pdf', bbox_inches='tight')
                plt.savefig(f'{args.savefig}/ltot_{particle}.png', bbox_inches='tight')
            if args.show:
                plt.show()

        pdict = {f'p{_i}': _par for _i, _par in enumerate(par_fits)}
        pdict['s'] = sgns
        outf = files("acem") / "resources" / "ltot" / f"{particle}.npz"
        with as_file(outf) as fpath:
            np.savez(fpath, **pdict)
