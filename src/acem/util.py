from pathlib import Path
from collections import OrderedDict
from typing import Callable, TypeVar
import numpy as np
from scipy import stats
from acem import model
try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False


def get_wasserstein_dist(darr):
    nbins = int(darr[0, 509])
    bwidt = darr[0, 508]

    bin_centers = (np.arange(nbins) + 0.5) * bwidt

    current_ws = []

    Xa, Ya = np.meshgrid(np.arange(0, nbins) * bwidt, darr[:, nbins + 2])
    _, Yb = np.meshgrid(np.arange(0, nbins) * bwidt, darr[:, nbins + 3])

    parr = stats.gamma(
        Ya, scale=model.Parametrization1D.FLUKA_MEDIUM.lrad / Yb
    ).pdf(Xa)

    for i in range(len(darr)):
        try:
            dist = stats.wasserstein_distance(bin_centers, bin_centers, darr[i, :nbins] / darr[i, 501], parr[i])
            current_ws.append(dist)
        except ValueError:
            current_ws.append(np.nan)

    return np.array(current_ws)


def get_wasserstein_rw(darr, pid, ene):
    nbins = int(darr[0, 509])
    bwidt = darr[0, 508]

    bin_centers = (np.arange(nbins) + 0.5) * bwidt
    rwth = model.RWParametrization1D(model.Parametrization1D.FLUKA_MEDIUM)
    gamm = rwth._shape(pid, ene)

    Xa, Ya = np.meshgrid(np.arange(0, nbins)*bwidt, gamm.args[0])
    _, Yb = np.meshgrid(np.arange(0, nbins)*bwidt, gamm.kwds['scale'])
    parr = stats.gamma(
        Ya, scale=model.Parametrization1D.FLUKA_MEDIUM.lrad/Yb).pdf(Xa)

    current_ws_rw = []
    for i in range(len(darr)):
        dist_rw = stats.wasserstein_distance(bin_centers, bin_centers, darr[i, :nbins] / darr[i, 501], parr[0])
        current_ws_rw.append(dist_rw)

    return np.array(current_ws_rw)


def get_header(fpath: str | Path):
    # probe number of zbins
    _a = np.loadtxt(fpath, delimiter=',', max_rows=1)
    nzbins = int(_a[-6])
    nzwide = _a[-7]

    return [str(i) for i in np.linspace(0,nzbins * nzwide,nzbins)] + \
        ['Energy','ltot','gammaA','gammaB','covAA','covAB','covBB','NumPeaks','Zwidth','Zbins','Peak1','Peak2','Peak3','Peak4','Peak5']


def get_dtype(header: list[str]) -> list[tuple[str, str]]:
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
    w1 = get_wasserstein_dist(a)
    return a[~np.isnan(a[:, nzbins+2]) &
             (a[:, nzbins+1] >= lo) &
             (a[:, nzbins+1] <= hi) &
             (w1 < np.nanquantile(w1, 0.998))]


def load_csv(fpath: str | Path, clean: bool=True) -> 'pd.DataFrame | pd.Series':
    if not HAVE_PANDAS:
        raise RuntimeError(
            "The 'load_csv' function requires the 'pandas' package to be installed."
        )

    df = pd.read_csv(fpath, names=get_header(fpath))
    # DEBUG
    # df['Zbins'] = 500
    # df['Zwidth'] = 10
    # END
    if not clean:
        return df

    df.dropna(subset='gammaA', inplace=True)
    lo, hi = np.quantile(df['ltot'], [0.005, 1.0])
    w1 = get_wasserstein_dist(df.to_numpy())
    return df[(df['ltot'] >= lo) & (df['ltot'] <= hi) & (w1 < np.nanquantile(w1, 0.998))]


T = TypeVar('T')
def load_batch(pattern: str,
               *args,
               loader: Callable[..., T]=load_csv,
               **kwargs) -> dict[float, T]:
    fglobs = Path('.').glob(pattern)
    edict = {float(_.stem.split('_')[-1]):_  for _ in fglobs}
    Dat = OrderedDict()
    for ene in sorted(edict.keys()):
        if ene in Dat:
            raise RuntimeError('Multiple files with identical energies were found.')
        Dat[ene] = loader(edict[ene], *args, **kwargs)
    return Dat
