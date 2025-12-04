from abc import ABC, abstractmethod
from pathlib import Path
from importlib.resources import files, as_file
from typing import Callable, Dict, NamedTuple, List
import numpy as np
from numpy.random import Generator
import numpy.typing as npt
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen
from . import media
from .pdg import FLUKA2PDG
from .maths import efn, lin, cbc, qrt, a, b, BSpline3D


def ltot_scale(m0: media.Medium, m1: media.Medium):
    return m0.density / m1.density * (1. - 1./m1.nphase) * (1. + 1./m1.nphase) / ((1. - 1./m0.nphase) * (1. + 1./m0.nphase))


class Shower1D(NamedTuple):
    """
    A one-dimensional representation of the Cherenkov-weighted track lengths of a particle shower, along the shower axis.

    Parameters
    ----------
    ltot (float) : The total Cherenkov-weighted track length
    shape (rv_frozen) : A frozen scipy.stats distribution that describes the shape of the shower profile
    """
    ltot: float
    shape: rv_frozen

    def dldx(self, x: npt.ArrayLike) -> np.ndarray:
        return self.ltot * self.shape.pdf(x)


class ModelBase(ABC):
    @abstractmethod
    def ltot_dist(self, pdg: int, energy: float) -> rv_frozen:
        pass

    @abstractmethod
    def sample(self,
               pdg: int,
               energy: float,
               size: None | int) -> Shower1D | List[Shower1D]:
        pass


class RWParametrization1D(ModelBase):
    """
    Generates shower profiles for Cherenkov light yields, given a media.Medium.

    Based on: https://doi.org/10.1016/j.astropartphys.2013.01.015
    """

    MEAN_ALPHAS: Dict[int, float] = {
        11: 532.07078881,
        -11: 532.11320598,
        22: 532.08540905,
        211: 333.55182722
    }
    MEAN_BETAS: Dict[int, float] = {
        11: 1.00000211,
        -11: 0.99999254,
        22: 0.99999877,
        211: 1.03662217
    }
    SIGMA_ALPHAS: Dict[int, float] = {
        11: 5.78170887,
        -11: 5.73419669,
        22: 5.66586567,
        211: 119.20455395
    }
    SIGMA_BETAS: Dict[int, float] = {
        11: 0.5,
        -11: 0.5,
        22: 0.5,
        211: 0.80772057
    }
    GAMMA_A: Dict[int, Callable] = {
        11: lambda x: 2.01849 + 0.63176 * np.log(x),
        -11: lambda x: 2.00035 + 0.63190 * np.log(x),
        22: lambda x: 2.83923 + 0.58209 * np.log(x),
        211: lambda x: 1.58357292 + 0.41886807 * np.log(x),
    }
    GAMMA_B: Dict[int, float] = {
        11: 0.63207,
        -11: 0.63008,
        22: 0.64526,
        211: 0.33833116
    }

    # density and nphase used in Geant4 MC
    G4_MEDIUM: media.Medium = media.Medium(0.91, 1.33)

    def __init__(self, medium: media.Medium,
                 random_state: Generator | None=None):
        self.medium = medium
        self._rng = np.random.default_rng() if random_state is None else random_state
        self._scale = ltot_scale(self.G4_MEDIUM, self.medium)
        assert self._scale >= 0.

    def _ltot_mean(self, pdg: int, energy: float) -> float:
        return self.MEAN_ALPHAS[pdg] * energy**self.MEAN_BETAS[pdg] * self._scale

    def _ltot_sigma(self, pdg: int, energy: float) -> float:
        return self.SIGMA_ALPHAS[pdg] * energy**self.SIGMA_BETAS[pdg] * self._scale

    def _shape(self, pdg: int, energy: float) -> rv_frozen:
        return stats.gamma(self.GAMMA_A[pdg](energy),
                           scale=self.medium.lrad / self.GAMMA_B[pdg])

    def ltot_dist(self, pdg: int, energy: float) -> rv_frozen:
        """
        Retrieves the ltot distribution for a specified particle
        type and energy.  The returned object can be used to sample
        random values of the total Cherenkov track length, or to
        calculate its statistical properties.

        Parameters
        ----------
        pdg: The PDG (Particle Data Group) identifier of the particle. This integer
             specifies the particle type (e.g., 2112 for neutron, 2212 for proton).
        energy: The energy of the particle in GeV

        Returns
        -------
        rv_frozen: A frozen normal distribution object from scipy.stats.
        """
        return stats.norm(self._ltot_mean(pdg, energy), self._ltot_sigma(pdg, energy))
    
    def mean_1d(self, pdg: int, energy: float) -> Shower1D:
        """
        Retrieves the average Shower1D object for a specified
        particle type and energy.

        Parameters
        ----------
        pdg: The PDG (Particle Data Group) identifier of the particle. This integer
             specifies the particle type (e.g., 2112 for neutron, 2212 for proton).
        energy: The energy of the particle in GeV

        Returns
        -------
        Shower1D: A 1D shower profile using mean(ltot); shape is invariant given energy
        """
        return Shower1D(self._ltot_mean(pdg, energy), self._shape(pdg, energy))

    def sample(self,
               pdg: int,
               energy: float,
               size: None | int=None) -> Shower1D | List[Shower1D]:
        """
        Samples an individual Shower1D object for a specified
        particle type and energy. Only ltot is randomly sampled.

        Parameters
        ----------
        pdg: The PDG (Particle Data Group) identifier of the particle. This integer
             specifies the particle type (e.g., 2112 for neutron, 2212 for proton).
        energy: The energy of the particle in GeV
        size: int or None. If None, a single Shower1D is returned, otherwise
        a list of Shower1Ds of length size is returned

        Returns
        -------
        Shower1D samples (see size above)
        """
        _size = 1 if size is None else size
        _samp = [Shower1D(_, self._shape(pdg, energy))
                 for _ in self.ltot_dist(pdg, energy).rvs(_size, random_state=self._rng)]
        return _samp[0] if size is None else _samp


class Parametrization1D(ModelBase):
    """
    Generates shower profiles for Cherenkov light yields, given a media.Medium. Includes fluctuations in shape and nuclear effects for total light yield.

    Based on: TBD
    """
    @staticmethod
    def load_ltots() -> Dict[int, np.lib.npyio.NpzFile]:
        data = {}
        for entry in (files("shosim") / "resources" / "ltot").iterdir():
            if not entry.is_file():
                continue
            with as_file(entry) as fpath:
                data[FLUKA2PDG[Path(entry.name).stem]] = np.load(fpath)
        return data

    @staticmethod
    def load_thetas() -> Dict[int, BSpline3D]:
        data = {}
        for entry in (files("shosim") / "resources" / "theta").iterdir():
            if not entry.is_file():
                continue
            with as_file(entry) as fpath:
                theta = np.load(fpath)
                c = theta["c"]
                t = tuple(theta[f"t{_}"] for _ in range(c.ndim))
                k = theta["k"]
                data[FLUKA2PDG[Path(entry.name).stem]] = BSpline3D.create(t, c, k)
        return data

    LTOTS: Dict[int, np.lib.npyio.NpzFile] = load_ltots()
    THETAS: Dict[int, BSpline3D] = load_thetas()
    # density and nphase used in FLUKA MC
    FLUKA_MEDIUM = media.Medium(0.9216, 1.33)

    def __init__(self, medium: media.Medium,
                 converter: Callable[[int], int] | None=None,
                 random_state: Generator | None=None):
        self.medium: media.Medium = medium
        self._rng: Generator = np.random.default_rng() if random_state is None else random_state
        self._converter: Callable = self._default_converter if converter is None else converter
        self._scale: float = ltot_scale(self.FLUKA_MEDIUM, self.medium)
        assert self._scale >= 0.
    
    def _default_converter(self, pdg: int):
        """
        Default generalization of existing parametrizations to a
        larger subset of PDG codes. Here, the only assumptions are:

        * antiparticles (negative PDG codes) are assumed to be
        equivalent to their particle counterparts.
        * K0 is treated as a 0.5 mixture of K0_L and K0_S

        User defined converters can be passed as an argument when
        during instantiation
        """
        _pdg = abs(pdg)

        if _pdg == 311:
            # K0
            return 130 if self._rng.uniform() < 0.5 else 310

        return _pdg

    def _shape(self, a: float, b: float) -> rv_frozen:
        return stats.gamma(a, scale=self.medium.lrad / b)

    def ltot_dist(self, pdg: int, energy: float) -> rv_frozen:
        """
        Retrieves the ltot distribution for a specified particle
        type and energy.  The returned object can be used to sample
        random values of the total Cherenkov track length, or to
        calculate its statistical properties.

        Parameters
        ----------
        pdg: The PDG (Particle Data Group) identifier of the particle. This integer
             specifies the particle type (e.g., 2112 for neutron, 2212 for proton).
        energy: The energy of the particle in GeV

        Returns
        -------
        rv_frozen: A frozen skewnormal (EM) or NIG (otherwise) distribution
        object from scipy.stats.
        """
        ltpars = self.LTOTS[self._converter(pdg)]
        # since the fit is performed in log-space, distribution
        # parameters with all-negative values are abs'd the stored 's'
        # keeps track of the final sign to apply
        sgns = ltpars['s']
        if len(sgns) == 3:
            sdist = stats.skewnorm
        elif len(sgns) == 4:
            sdist = stats.norminvgauss
        else:
            raise RuntimeError('Unable to match l_tot distributions')

        sdist_args = []
        for i, sgn in enumerate(sgns):
            _p = ltpars[f'p{i}']
            if len(_p) == 2:
                _fn = lin
            elif len(_p) == 4:
                _fn = cbc
            elif len(_p) == 5:
                _fn = qrt
            else:
                raise RuntimeError('Unable to match parameters to function')
            sdist_args.append(sgn * efn(energy, _fn, *_p))
        # scipy distributions are loc-scale families so we only need
        # to rescale the loc and scale parameters
        sdist_args[-1] *= self._scale
        sdist_args[-2] *= self._scale
        return sdist(*sdist_args)

    def mean_ab(self, pdg: int, energy: float) -> Shower1D:
        """
        Retrieves the average Shower1D object for a specified
        particle type and energy.

        Parameters
        ----------
        pdg: The PDG (Particle Data Group) identifier of the particle. This integer
             specifies the particle type (e.g., 2112 for neutron, 2212 for proton).
        energy: The energy of the particle in GeV

        Returns
        -------
        Shower1D: The average 1D shower profile

        Note
        ----
        The shape is taken as the average over parameters a and b, which differs
        from the average over 1/ltot * dl/dx
        """
        bspl = self.THETAS[pdg]
        ap, bp = bspl.mean(np.log10(energy))
        return Shower1D(self.ltot_dist(pdg, energy).mean(), self._shape(a(ap), b(bp)))
    
    def sample(self,
               pdg: int,
               energy: float,
               size: None | int=None) -> Shower1D | List[Shower1D]:
        """
        Samples an individual Shower1D object for a specified
        particle type and energy.

        Parameters
        ----------
        pdg: The PDG (Particle Data Group) identifier of the particle. This integer
             specifies the particle type (e.g., 2112 for neutron, 2212 for proton).
        energy: The energy of the particle in GeV
        size: int or None. If None, a single Shower1D is returned, otherwise
        a list of Shower1Ds of length size is returned

        Returns
        -------
        Shower1D samples (see size above)
        """
        _size = 1 if size is None else size
        ltots = self.ltot_dist(pdg, energy).rvs(_size, random_state=self._rng)
        bspl = self.THETAS[pdg]
        aps, bps = bspl.sample(np.log10(energy), _size, random_state=self._rng).T

        _samp = [Shower1D(ltot, self._shape(a(ap), b(bp)))
                 for ltot, ap, bp in zip(ltots, aps, bps)]
        return _samp[0] if size is None else _samp
