import numpy as np
from matplotlib import pyplot as plt
import FitSpline
from scipy import stats
plt.style.use('present')

lrad = 0.358/0.9216 * 100 # cm
estr = '1.00000E3'
ppart = 'ELECTRON'
npart = 3

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

a = np.loadtxt(f'DataOutputs_{ppart}/{ppart}_{estr}.csv', delimiter=',')
dfs = FitSpline.load_ian(ppart, f'DataOutputs_{ppart}')
df = dfs[estr]

xdf = df[df['gammaA'] < 1.0001]
x = a[df['gammaA'] < 1.0001]


for _ in range(npart):
    _row = df.iloc[_]
    _nbins = int(_row['Zbins'])
    xs = np.arange(_nbins) * _row['Zwidth']
    ltot = _row['ltot']
    plt.plot(xs, a[_, :_nbins], color=colors[_])
    plt.plot(xs, ltot*stats.gamma.pdf(xs, _row['gammaA'], scale=lrad/_row['gammaB']), color=colors[_], linestyle='--')
plt.ylabel('dl/dx')
plt.xlabel('[cm]')
plt.xlim(0, 2000)
plt.show()

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
