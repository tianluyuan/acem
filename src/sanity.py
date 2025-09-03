import numpy as np
from matplotlib import pyplot as plt
import FitSpline
from scipy import stats
import argparse
from pathlib import Path
plt.style.use('present')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

lrad = 0.358/0.9216 * 100 # cm

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fitting script')
    parser.add_argument('fpath', type=Path)
    parser.add_argument('-N', '--npart', type=int, default=3,
                        help='number of profiles to show')
    parser.add_argument('-0', '--istart', type=int, default=0,
                        help='index to start at')
    parser.add_argument('--reference', type=Path, default=None,
                        help='path to reference for ltot comp')
    parser.add_argument('--label', type=Path, default='current',
                        help='label for the file of interest')
    parser.add_argument('--rlabel', type=Path, default='reference',
                        help='label for the reference file')
    parser.add_argument('--clean', type=bool, default=False,
                        help='remove outliers in data')
    parser.add_argument('--savefig', default=None, type=str,
                        help='Save plots to specified directory')
    args = parser.parse_args()

    a = FitSpline.load_npy(args.fpath, args.clean)
    df = FitSpline.load_csv(args.fpath, args.clean)

    for _ in range(args.npart):
        _row = df.iloc[args.istart+_]
        _nbins = int(_row['Zbins'])
        xs = np.arange(_nbins) * _row['Zwidth']
        ltot = _row['ltot']
        plt.plot(xs, a[args.istart+_, :_nbins], color=colors[_], label=f'{ltot}')
        plt.plot(xs, ltot*stats.gamma.pdf(xs, _row['gammaA'], scale=lrad/_row['gammaB']), color=colors[_], linestyle='--')
    plt.ylabel('dl/dx')
    plt.xlabel('[cm]')
    plt.xlim(0, 2000)
    plt.legend()
    if args.savefig:
        plt.savefig(f'{args.savefig}/dldx.pdf', bbox_inches='tight')
        plt.savefig(f'{args.savefig}/dldx.png', bbox_inches='tight')
    plt.show()

    xdf = df[df['gammaA'] < 1.5]
    x = a[df['gammaA'] < 1.5]
    for _ in range(args.istart, min(args.istart + args.npart, len(x))):
        _row = xdf.iloc[_]
        _nbins = int(_row['Zbins'])
        xs = np.arange(_nbins) * _row['Zwidth']
        plt.plot(xs, x[_, :_nbins], color=colors[_-args.istart])
        ltot = _row['ltot']
        plt.plot(xs, ltot*stats.gamma.pdf(xs, _row['gammaA'], scale=lrad/_row['gammaB']), color=colors[_-args.istart], linestyle='--')
    plt.ylabel('dl/dx')
    plt.xlabel('[cm]')
    plt.xlim(0, 2000)
    if args.savefig:
        plt.savefig(f'{args.savefig}/dldx_smallgA.pdf', bbox_inches='tight')
        plt.savefig(f'{args.savefig}/dldx_smallgA.png', bbox_inches='tight')
    plt.show()

    # compare ltot vs a reference
    _s = np.s_[:, 501]
    if args.reference is not None and args.reference.is_file():
        b = FitSpline.load_npy(args.reference, args.clean)
        bins = np.linspace(min(a[_s].min(), b[_s].min()),
                           max(a[_s].max(), b[_s].max()),
                           100)
        plt.clf()
        plt.hist(a[_s], bins=bins, density=True, histtype='step', label=args.label)
        plt.hist(b[_s], bins=bins, density=True, histtype='step', label=args.rlabel)
        plt.xlabel('Total track length [cm]')
        plt.ylabel('density')
        plt.legend()
        plt.yscale('log')
        if args.savefig:
            plt.savefig(f'{args.savefig}/ltots.pdf', bbox_inches='tight')
            plt.savefig(f'{args.savefig}/ltots.png', bbox_inches='tight')
        plt.show()

    # check saved dl/dx for random 3 vs bottom-n%-ltot 3 events
    plt.clf()
    al = a[a[_s] < np.quantile(a[_s], 0.01)]
    [plt.plot(range(500), a[_,:500], label=f'{a[_, 501]}', c=colors[_-args.istart]) for _ in range(args.istart, args.istart+args.npart)]
    [plt.plot(range(500), al[_,:500], linestyle='--', label=f'{al[_, 501]}', c=colors[_-args.istart]) for _ in range(args.istart, min(args.istart+args.npart, len(al)))]
    plt.legend()
    if args.savefig:
        plt.savefig(f'{args.savefig}/dldx_smallL.pdf', bbox_inches='tight')
        plt.savefig(f'{args.savefig}/dldx_smallL.png', bbox_inches='tight')
    plt.show()
