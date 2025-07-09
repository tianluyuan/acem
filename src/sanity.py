import numpy as np
from matplotlib import pyplot as plt
import FitSpline
from scipy import stats
plt.style.use('present')

lrad = 0.358/0.9216 * 100 # cm
fluka_bin_width = 1000 * 1000 * 10 # cm**3
bin_width = 10 # cm
estr = '1.00000E1'
ppart = 'PION+'
npart = 3

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

a = np.loadtxt(f'DataOutputs_{ppart}/{ppart}_{estr}.csv', delimiter=',')
dfs = FitSpline.load_ian(ppart, f'DataOutputs_{ppart}')
df = dfs[estr]

xdf = df[df['gammaA'] < 1.0001]
x = a[df['gammaA'] < 1.0001]
xs = np.arange(500) * bin_width

for _ in range(npart):
    plt.plot(xs, a[_, :500]/bin_width * fluka_bin_width, color=colors[_])
    _row = df.iloc[_]
    ltot = np.sum(a[_, :500]) * fluka_bin_width
    plt.plot(xs, ltot*stats.gamma.pdf(xs, _row['gammaA'], scale=lrad/_row['gammaB']), color=colors[_], linestyle='--')
plt.ylabel('dl/dx')
plt.xlabel('[cm]')
plt.xlim(0, 2000)
plt.show()

for _ in range(min(npart, len(x))):
    plt.plot(xs, x[_, :500]/bin_width * fluka_bin_width, color=colors[_])
    _row = xdf.iloc[_]
    ltot = np.sum(x[_, :500]) * fluka_bin_width
    plt.plot(xs, ltot*stats.gamma.pdf(xs, _row['gammaA'], scale=lrad/_row['gammaB']), color=colors[_], linestyle='--')
plt.ylabel('dl/dx')
plt.xlabel('[cm]')
plt.xlim(0, 2000)
plt.show()
