from pathlib import Path
from collections import OrderedDict
import numpy as np
from typing import Dict, Callable, TypeVar
try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False


def get_header(nzbins, nzwide):
    return [str(i) for i in np.linspace(0,nzbins * nzwide,nzbins)] + \
        ['Energy','ltot','gammaA','gammaB','covAA','covAB','covBB','NumPeaks','Zwidth','Zbins','Peak1','Peak2','Peak3','Peak4','Peak5']


def get_dtype(header):
    return [(_, 'i4' if _ in ['Zbins', 'NumPeaks'] else 'f8') for _ in header]


def format_energy(num: float) -> str:
    num_str = "{:.5e}".format(num)
    coefficient, exponent = num_str.split('e')
    exponent2 = exponent.replace('+0', '')
    if exponent2==exponent:
        raise Exception("Invalid energy for this formating. Energy must be within the interval [1,1e10)")
    coefficient = coefficient.ljust(7, '0')
    return f"{coefficient}E{exponent2}"


def load_npy(fpath: str | Path, clean: bool=True) -> np.ndarray:
    a = np.loadtxt(fpath, delimiter=',')
    assert len(np.unique(a[:, -6]) == 1)
    assert len(np.unique(a[:, -7]) == 1)
    nzbins = int(a[0, -6])

    if not clean:
        return a

    lo, hi = np.quantile(a[:, nzbins+1], [0.005, 1.0])
    return a[~np.isnan(a[:, nzbins+2]) & (a[:, nzbins+1] >= lo) & (a[:, nzbins+1] <= hi)]


def load_csv(fpath: str | Path, clean: bool=True) -> 'pd.DataFrame | pd.Series':
    if not HAVE_PANDAS:
        raise RuntimeError(
            "The 'load_csv' function requires the 'pandas' package to be installed."
        )
    # probe number of zbins
    _a = load_npy(fpath, clean)
    nzbins = int(_a[0, -6])
    nzwide = _a[0, -7]

    df = pd.read_csv(fpath, names=get_header(nzbins, nzwide))
    # DEBUG
    # df['Zbins'] = 500
    # df['Zwidth'] = 10
    # END
    if not clean:
        return df

    df.dropna(subset='gammaA', inplace=True)
    lo, hi = np.quantile(df['ltot'], [0.005, 1.0])
    return df[(df['ltot'] >= lo) & (df['ltot'] <= hi)]


T = TypeVar('T')
def load_batch(pattern: str,
               *args,
               loader: Callable[..., T]=load_csv,
               **kwargs) -> Dict[float, T]:
    fglobs = Path('.').glob(pattern)
    edict = {float(_.stem.split('_')[-1]):_  for _ in fglobs}
    Dat = OrderedDict()
    for ene in sorted(edict.keys()):
        if ene in Dat:
            raise RuntimeError('Multiple files with identical energies were found.')
        Dat[ene] = loader(edict[ene], *args, **kwargs)
    return Dat
