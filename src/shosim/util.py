from collections import OrderedDict
import numpy as np
import pandas as pd


def format_energy(num: float):
    num_str = "{:.5e}".format(num)
    coefficient, exponent = num_str.split('e')
    exponent2 = exponent.replace('+0', '')
    if exponent2==exponent:
        raise Exception("Invalid energy for this formating. Energy must be within the interval [1,1e10)")
    coefficient = coefficient.ljust(7, '0')
    return f"{coefficient}E{exponent2}"


def load_npy(fpath, clean=True):
    a = np.loadtxt(fpath, delimiter=',')
    assert len(np.unique(a[:, -6]) == 1)
    assert len(np.unique(a[:, -7]) == 1)
    nzbins = int(a[0, -6])

    if not clean:
        return a

    lo, hi = np.quantile(a[:, nzbins+1], [0.005, 1.0])
    return a[~np.isnan(a[:, nzbins+2]) & (a[:, nzbins+1] >= lo) & (a[:, nzbins+1] <= hi)]


def load_csv(fpath, clean=True):
    # probe number of zbins
    _a = load_npy(fpath, clean)
    nzbins = int(_a[0, -6])
    nzwide = _a[0, -7]

    header = [str(i) for i in np.linspace(0,nzbins * nzwide,nzbins)] + \
        ['Energy','ltot','gammaA','gammaB','covAA','covAB','covBB','NumPeaks','Zwidth','Zbins','Peak1','Peak2','Peak3','Peak4','Peak5']
    df = pd.read_csv(fpath, names=header)
    # DEBUG
    # df['Zbins'] = 500
    # df['Zwidth'] = 10
    # END
    if not clean:
        return df
    
    df.dropna(subset='gammaA', inplace=True)
    lo, hi = np.quantile(df['ltot'], [0.005, 1.0])
    return df[(df['ltot'] >= lo) & (df['ltot'] <= hi)]


def load_batch(log_ens, particle, directory, clean=True):
    energy_strs = [format_energy(num) for num in 10**log_ens]
    Dat = OrderedDict()
    for energy_str in energy_strs:
        Dat[energy_str] = load_csv(f'{directory}/{particle}_' + energy_str + '.csv', clean)
    return Dat
