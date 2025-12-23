#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import argparse
from pathlib import Path
from acem import util, media
plt.style.use('present')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

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
    parser.add_argument('--clean', default=False, action='store_true',
                        help='remove outliers in data')
    parser.add_argument('--savefig', default=None, type=str,
                        help='Save plots to specified directory')
    args = parser.parse_args()

    a = util.load_npy(args.fpath, args.clean)
    df = util.load_csv(args.fpath, args.clean)

    for _ in range(args.npart):
        _row = df.iloc[args.istart+_]
        _nbins = int(_row['Zbins'])
        xs = np.arange(_nbins) * _row['Zwidth']
        ltot = _row['ltot']
        c = colors[_ % len(colors)]
        # from acem import maths
        # if maths.aprime(_row['gammaA']) < 0.9:
        #     continue
        # if (_row['gammaA'] - 1) *media.IC3.lrad/_row['gammaB'] < 2e3:
        #     continue
        # print(maths.aprime(_row['gammaA']), maths.bprime(_row['gammaB']))
        plt.plot(xs, a[args.istart+_, :_nbins], color=c, label=f'{ltot}')
        plt.plot(xs,
                 ltot*stats.gamma.pdf(xs, _row['gammaA'],
                                      scale=media.IC3.lrad/_row['gammaB']),
                 color=c, linestyle='--')
    plt.ylabel('dl/dx')
    plt.xlabel('[cm]')
    plt.xlim(0, 5000)
    plt.legend()
    if args.savefig:
        plt.savefig(f'{args.savefig}/dldx.pdf', bbox_inches='tight')
        plt.savefig(f'{args.savefig}/dldx.png', bbox_inches='tight')
        plt.yscale('log')
        plt.savefig(f'{args.savefig}/dldx_logy.pdf', bbox_inches='tight')
        plt.savefig(f'{args.savefig}/dldx_logy.png', bbox_inches='tight')
    plt.show()

    xdf = df[df['gammaA'] < 1.5]
    x = a[df['gammaA'] < 1.5]
    for _ in range(args.istart, min(args.istart + args.npart, len(x))):
        _row = xdf.iloc[_]
        _nbins = int(_row['Zbins'])
        xs = np.arange(_nbins) * _row['Zwidth']
        c = colors[_-args.istart % len(colors)]
        plt.plot(xs, x[_, :_nbins], color=c)
        ltot = _row['ltot']
        plt.plot(xs,
                 ltot*stats.gamma.pdf(xs, _row['gammaA'],
                                      scale=media.IC3.lrad/_row['gammaB']),
                 color=c, linestyle='--')
    plt.ylabel('dl/dx')
    plt.xlabel('[cm]')
    plt.xlim(0, 3000)
    if args.savefig:
        plt.savefig(f'{args.savefig}/dldx_smallgA.pdf', bbox_inches='tight')
        plt.savefig(f'{args.savefig}/dldx_smallgA.png', bbox_inches='tight')
    plt.show()

    # compare ltot vs a reference
    if args.reference is not None and args.reference.is_file():
        df_b = util.load_csv(args.reference, args.clean)
        bins = np.linspace(min(df['ltot'].min(), df_b['ltot'].min()),
                           max(df['ltot'].max(), df_b['ltot'].max()),
                           100)
        plt.clf()
        plt.hist(df['ltot'], bins=bins, density=True, histtype='step', label=args.label)
        plt.hist(df_b['ltot'], bins=bins, density=True, histtype='step', label=args.rlabel)
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
    al = df[df['ltot'] < np.quantile(df['ltot'], 0.01)].to_numpy()
    [plt.plot(xs, a[_,:_nbins], label=f'{a[_, _nbins+1]:.2f}', c=colors[_-args.istart]) for _ in range(args.istart, args.istart+args.npart)]
    [plt.plot(xs, al[_,:_nbins], linestyle='--', label=f'{al[_, _nbins+1]:.2f}', c=colors[_-args.istart]) for _ in range(args.istart, min(args.istart+args.npart, len(al)))]
    plt.legend()
    plt.yscale('log')
    plt.xlabel('[cm]')
    plt.ylabel('dl/dx')
    if args.savefig:
        plt.savefig(f'{args.savefig}/dldx_smallL.pdf', bbox_inches='tight')
        plt.savefig(f'{args.savefig}/dldx_smallL.png', bbox_inches='tight')
    plt.show()
