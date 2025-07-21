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
estr = '1.00000E3'
ppart = 'PROTON'
npart = 3

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fitting script')
    parser.add_argument('fpath', type=Path)
    parser.add_argument('--reference', type=Path, default=None)
    args = parser.parse_args()

    slt = np.s_[:, 501]
    a = np.loadtxt(args.fpath, delimiter=',')
    df = FitSpline.load_csv(args.fpath)

    for _ in range(npart):
        _row = df.iloc[_]
        _nbins = int(_row['Zbins'])
        xs = np.arange(_nbins) * _row['Zwidth']
        ltot = _row['ltot']
        plt.plot(xs, a[_, :_nbins], color=colors[_], label=f'{ltot}')
        plt.plot(xs, ltot*stats.gamma.pdf(xs, _row['gammaA'], scale=lrad/_row['gammaB']), color=colors[_], linestyle='--')
    plt.ylabel('dl/dx')
    plt.xlabel('[cm]')
    plt.xlim(0, 2000)
    plt.legend()
    plt.show()

    xdf = df[df['gammaA'] < 1.5]
    x = a[df['gammaA'] < 1.5]
    for _ in range(min(npart, len(x))):
        _row = xdf.iloc[_]
        _nbins = int(_row['Zbins'])
        xs = np.arange(_nbins) * _row['Zwidth']
        plt.plot(xs, x[_, :_nbins], color=colors[_])
        ltot = _row['ltot']
        plt.plot(xs, ltot*stats.gamma.pdf(xs, _row['gammaA'], scale=lrad/_row['gammaB']), color=colors[_], linestyle='--')
    plt.ylabel('dl/dx')
    plt.xlabel('[cm]')
    plt.xlim(0, 2000)
    plt.show()

    # check ltot vs out.v1 (unweighted track length)
    if args.reference.is_file():
        b = np.loadtxt(args.reference, delimiter=',')
        bins = np.linspace(min(a[slt].min(), b[slt].min() * 1e7),
                           max(a[slt].max(), b[slt].max() * 1e7),
                           100)
        plt.clf()
        plt.hist(a[slt], bins=bins, density=True, histtype='step')
        plt.hist(b[slt] * 1e7, bins=bins, density=True, histtype='step')
        plt.xlabel('Total track length [cm]')
        plt.ylabel('density')
        plt.legend(['xy +/- 15m, density=0.9216', 'raw, xy +/- 5m, density=0.917'])
        plt.yscale('log')
        plt.show()

    # check saved dl/dx for random 3 vs bottom-n%-ltot 3 events
    plt.clf()
    al = a[a[slt] < np.quantile(a[slt], 0.01)]
    [plt.plot(range(500), a[_,:500], label=f'{a[_, 501]}', c=colors[_]) for _ in range(3)];
    [plt.plot(range(500), al[_,:500], linestyle='--', label=f'{al[_, 501]}', c=colors[_]) for _ in range(3)]
    plt.legend()
    plt.show()
