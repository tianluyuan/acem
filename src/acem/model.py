from abc import ABC, abstractmethod
from pathlib import Path
from importlib.resources import files, as_file
from typing import Callable, NamedTuple
import numpy as np
from numpy.random import Generator
import numpy.typing as npt
from scipy import stats, interpolate
from scipy.stats._distn_infrastructure import rv_frozen
from . import media
from .pdg import FLUKA2PDG
from .maths import a, b, BSpline3D


def ltot_scale(m0: media.Medium, m1: media.Medium) -> float:
    """
    Computes the ltot rescaling factor to convert from one medium to another of differing density and index of refraction.

    Parameters
    ----------
    m0 : Medium
        the medium to convert from
    m1 : Medium
        the medium to convert to

    Returns
    -------
    scale : float
        scale factor to be applied to the total, weighted Cherenkov track length
    """
    return m0.density / m1.density * (1. - 1./m1.nphase) * (1. + 1./m1.nphase) / ((1. - 1./m0.nphase) * (1. + 1./m0.nphase))


class Shower1D(NamedTuple):
    """
    A one-dimensional representation of the Cherenkov-weighted track lengths of a particle shower, along the shower axis.
    This class is a simple container to represent the amplitude and shape of the shower profile.

    Parameters
    ----------
    ltot : float
        The total Cherenkov-weighted track length
    shape : rv_frozen
        A frozen scipy.stats distribution that describes the shape of the shower profile
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
    def mean(self, pdg: int, energy: float) -> Shower1D:
        pass

    @abstractmethod
    def sample(self,
               pdg: int,
               energy: float,
               size: None | int) -> Shower1D | list[Shower1D]:
        pass


class RWParametrization1D(ModelBase):
    """
    Generates shower profiles for Cherenkov light yields, given a media.Medium.

    Based on: https://doi.org/10.1016/j.astropartphys.2013.01.015
    """

    MEAN_ALPHAS: dict[int, float] = {
        11: 532.07078881,
        -11: 532.11320598,
        22: 532.08540905,
        211: 333.55182722,
        -211: 335.84489578,
        130: 326.00450524,
        2212: 287.37183922,
        -2212: 303.33074914,
        2112: 278.43854660
    }
    MEAN_BETAS: dict[int, float] = {        
        11: 1.00000211,
        -11: 0.99999254,
        22: 0.99999877,
        211: 1.03662217,
        -211: 1.03584394,
        130: 1.03931457,
        2212: 1.05172118,
        -2212: 1.04322206,
        2112: 1.05582906
    }
    SIGMA_ALPHAS: dict[int, float] = {
        11: 5.78170887,
        -11: 5.73419669,
        22: 5.66586567,
        211: 119.20455395,
        -211: 122.50188073,
        130: 121.41970572,
        2212: 88.04581378,
        -2212: 113.23088104,
        2112: 93.22787137
    }
    SIGMA_BETAS: dict[int, float] = {
        11: 0.5,
        -11: 0.5,
        22: 0.5,
        211: 0.80772057,
        -211: 0.80322520,
        130: 0.80779629,
        2212: 0.82445572,
        -2212: 0.77134060,
        2112: 0.81776503
    }
    GAMMA_A: dict[int, Callable] = {
        11: lambda x: 2.01849 + 1.45469 * np.log10(x),
        -11: lambda x: 2.00035 + 1.45501 * np.log10(x),
        22: lambda x: 2.83923 + 1.34031 * np.log10(x),
        211: lambda x: 1.81098 + 0.90572 * np.log10(x),
        -211: lambda x: 1.81430 + 0.90165 * np.log10(x),
        130: lambda x: 1.99751 + 0.80628 * np.log10(x),
        2212: lambda x: 1.62345 + 0.90875 * np.log10(x),
        -2212: lambda x: 1.88676 + 0.78825 * np.log10(x),
        2112: lambda x: 1.78137 + 0.87687 * np.log10(x)
    }
    GAMMA_B: dict[int, float] = {
        11: 0.63207,
        -11: 0.63008,
        22: 0.64526,
        211: 0.34347,
        -211: 0.34131,
        130: 0.35027,
        2212: 0.35871,
        -2212: 0.35063,
        2112: 0.35473
    }

    # density and nphase used in Geant4 MC
    G4_MEDIUM: media.Medium = media.Medium(0.91, 1.33)

    def __init__(self, medium: media.Medium,
                 random_state: Generator | None=None):
        self.medium: media.Medium = medium
        self._rng: Generator = np.random.default_rng() if random_state is None else random_state
        self._scale: float = max(ltot_scale(self.G4_MEDIUM, self.medium), 0.)

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
        pdg : int
            The PDG (Particle Data Group) identifier of the particle. This integer
            specifies the particle type (e.g., 2112 for neutron, 2212 for proton).
        energy : float
            The energy of the particle in GeV

        Returns
        -------
        ltot_dist : rv_frozen
            A frozen normal distribution object from scipy.stats.
        """
        return stats.norm(self._ltot_mean(pdg, energy), self._ltot_sigma(pdg, energy))
    
    def mean(self, pdg: int, energy: float) -> Shower1D:
        """
        Retrieves the average Shower1D object for a specified
        particle type and energy.

        Parameters
        ----------
        pdg : int
            The PDG (Particle Data Group) identifier of the particle. This integer
            specifies the particle type (e.g., 2112 for neutron, 2212 for proton).
        energy : float
            The energy of the particle in GeV

        Returns
        -------
        mean : Shower1D
            A 1D shower profile using mean(ltot); shape is invariant given energy

        Notes
        -----
        Since the shape is fixed this simply returns Shower1D(mean(ltot), shape)
        """
        return Shower1D(self._ltot_mean(pdg, energy), self._shape(pdg, energy))

    def sample(self,
               pdg: int,
               energy: float,
               size: None | int=None) -> Shower1D | list[Shower1D]:
        """
        Samples an individual or list of Shower1D for a specified
        particle type and energy. Only ltot is randomly sampled.

        Parameters
        ----------
        pdg: int
            The PDG (Particle Data Group) identifier of the particle. This integer
            specifies the particle type (e.g., 2112 for neutron, 2212 for proton).
        energy: float
            The energy of the particle in GeV
        size: int or None
            If None, a single Shower1D is returned, otherwise a list of Shower1Ds of
            length size is returned

        Returns
        -------
        sample: Shower1D or list[Shower1D]
            samples (see size above)
        """
        _size = 1 if size is None else size
        _samp = [Shower1D(_, self._shape(pdg, energy))
                 for _ in self.ltot_dist(pdg, energy).rvs(_size, random_state=self._rng)]
        return _samp[0] if size is None else _samp


class Parametrization1D(ModelBase):
    """
    Generates shower profiles for Cherenkov light yields, given a media.Medium.
    Includes fluctuations in shape and nuclear effects for total light yield.

    Handles the loading of parameters that govern the amplitude and shape distributions,
    while exposing convenience functions to sample / obtain Shower1D objects.
    
    Parameters
    ----------
    medium: Medium
        A acem.media.Medium object with fixed density and nphase
    converter: Callable[[int], int], or None, optional
        A function callable that takes a PDG code and returns one as proxy for use
    random_state: Generator, or None, optional
        A numpy random number generator

    Returns
    -------
    ret: Parameterization1D
        A Parametrization1D object

    Notes
    -----
    Based on: TBD

    
    >>> a = Parametrization1D(media.ICE)
    >>> x = np.linspace(0, 1, 100)
    >>> y = np.linspace(0, 1, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> for pdg in a.THETAS:
    ...     log_elo = 0 if pdg in [11, 22] else 1
    ...     bspl = a.THETAS[pdg]
    ...     for en in np.linspace(log_elo, 6, 100):
    ...         Z0 = bspl(X, Y, en)
    ...         Z1 = bspl._legacy_eval(X, Y, en)
    ...         assert np.all(np.isclose(np.exp(Z0), np.exp(Z1), rtol=0.002))
    ...         assert np.isclose(bspl.integrate_grid(en).sum(), 1., atol=0.05)
    """
    @staticmethod
    def load_ltots() -> dict[int, np.lib.npyio.NpzFile]:
        data = {}
        for entry in (files("acem") / "resources" / "ltot").iterdir():
            if not entry.is_file():
                continue
            with as_file(entry) as fpath:
                data[FLUKA2PDG[Path(entry.name).stem]] = np.load(fpath)
        return data

    @staticmethod
    def load_thetas() -> dict[int, BSpline3D]:
        data = {}
        for entry in (files("acem") / "resources" / "theta").iterdir():
            if not entry.is_file():
                continue
            with as_file(entry) as fpath:
                theta = np.load(fpath)
                c = theta["c"]
                t = tuple(theta[f"t{_}"] for _ in range(c.ndim))
                k = theta["k"]
                data[FLUKA2PDG[Path(entry.name).stem]] = BSpline3D(interpolate.NdBSpline(t, c, k, extrapolate=False))
        return data

    LTOTS: dict[int, np.lib.npyio.NpzFile] = load_ltots()
    THETAS: dict[int, BSpline3D] = load_thetas()
    # density and nphase used in FLUKA MC
    FLUKA_MEDIUM = media.Medium(0.9216, 1.33)

    def __init__(self, medium: media.Medium,
                 converter: Callable[[int], int] | None=None,
                 random_state: Generator | None=None):
        if set(self.LTOTS.keys()) != set(self.THETAS.keys()):
            raise RuntimeError("Set of particles in LTOT and THETAS do not agree")

        self.medium: media.Medium = medium
        self._rng: Generator = np.random.default_rng() if random_state is None else random_state
        self._converter: Callable = self._default_converter if converter is None else converter
        self._scale: float = max(ltot_scale(self.FLUKA_MEDIUM, self.medium), 0.)
    
    def _default_converter(self, pdg: int):
        """
        Default generalization of existing parametrizations to a
        larger subset of PDG codes. Here, the only assumptions are:

        * antiparticles (negative PDG codes) are assumed to be
        equivalent to their particle counterparts.
        * K0 is treated as a 0.5 mixture of K0_L and K0_S

        User-defined converters can be passed as an argument 
        during instantiation

        Parameters
        ----------
        pdg : int
            The PDG (Particle Data Group) identifier of the particle. This integer
            specifies the particle type (e.g., 2112 for neutron, 2212 for proton).
        energy : float
            The energy of the particle in GeV

        Returns
        -------
        proxy_pdg : int
            abs(pdg) or mixture for K0
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
        pdg : int
            The PDG (Particle Data Group) identifier of the particle. This integer
            specifies the particle type (e.g., 2112 for neutron, 2212 for proton).
        energy : float
            The energy of the particle in GeV

        Returns
        -------
        ltot_dist : rv_frozen
            A frozen skewnormal (EM) or NIG (otherwise) distribution object

        
        >>> a = Parametrization1D(media.ICE)
        >>> b = RWParametrization1D(media.ICE)
        >>> for pdg in b.MEAN_ALPHAS:
        ...     for en in np.logspace(1, 6, 10):
        ...         ltot_a = a.ltot_dist(pdg, en).mean()
        ...         ltot_diff = np.abs(ltot_a - b.ltot_dist(pdg, en).mean())
        ...         assert ltot_diff / ltot_a < 0.26
        """
        ltpars = self.LTOTS[self._converter(pdg)]
        # since the fit is performed in log-space, distribution
        # parameters with all-negative values are abs'd and the stored 's'
        # keeps track of the final sign to apply
        sgns = ltpars['s']
        if len(sgns) == 3:
            sdist = stats.skewnorm
        elif len(sgns) == 4:
            sdist = stats.norminvgauss
        else:
            raise RuntimeError('Unable to match l_tot distributions')

        sdist_args = [sgn * np.exp(
            np.polyval(ltpars[f'p{i}'][::-1], np.log10(energy)))
                      for i, sgn in enumerate(sgns)]
        # scipy distributions are loc-scale families so we only need
        # to rescale the loc and scale parameters
        sdist_args[-1] *= self._scale
        sdist_args[-2] *= self._scale
        return sdist(*sdist_args)

    def mean(self, pdg: int, energy: float) -> Shower1D:
        """
        Retrieves the average Shower1D object for a specified
        particle type and energy.

        Parameters
        ----------
        pdg : int
            The PDG (Particle Data Group) identifier of the particle. This integer
            specifies the particle type (e.g., 2112 for neutron, 2212 for proton).
        energy : float
            The energy of the particle in GeV

        Returns
        -------
        mean : Shower1D
            The average 1D shower profile (see note)

        Note
        ----
        The shape is taken as the average over parameters a' and b', which differs
        from the average over 1/ltot * dl/dx
        """
        ap, bp = self.THETAS[self._converter(pdg)].mean(np.log10(energy))
        return Shower1D(self.ltot_dist(pdg, energy).mean(), self._shape(a(ap), b(bp)))
    
    def sample(self,
               pdg: int,
               energy: float,
               size: None | int=None) -> Shower1D | list[Shower1D]:
        """
        Samples an individual Shower1D object for a specified
        particle type and energy.

        Parameters
        ----------
        pdg : int
            The PDG (Particle Data Group) identifier of the particle. This integer
            specifies the particle type (e.g., 2112 for neutron, 2212 for proton).
        energy : float
            The energy of the particle in GeV
        size : int or None (optional)
            If None, a single Shower1D is returned, otherwise a list of Shower1Ds of
        length size is returned

        Returns
        -------
        sample : Shower1D or list[Shower1D]
            A sample or list of multiple samples (see size above)

        
        >>> a = Parametrization1D(media.ICE)
        >>> rng = np.random.default_rng(4)
        >>> for pdg in a.THETAS:
        ...     for en in np.linspace(1, 6, 10):
        ...         samp = a.THETAS[pdg].sample(en, 100, random_state=rng)
        ...         lsam = a.THETAS[pdg]._igrid_sample(en, 100, random_state=rng)
        ...         assert stats.ks_2samp(samp[:,0], lsam[:,0]).pvalue>0.01
        ...         assert stats.ks_2samp(samp[:,1], lsam[:,1]).pvalue>0.005
        """
        _size = 1 if size is None else size
        ltots = self.ltot_dist(pdg, energy).rvs(_size, random_state=self._rng)
        aps, bps = self.THETAS[self._converter(pdg)].sample(
            np.log10(energy),
            _size,
            random_state=self._rng).T

        _samp = [Shower1D(ltot, self._shape(a(ap), b(bp)))
                 for ltot, ap, bp in zip(ltots, aps, bps)]
        return _samp[0] if size is None else _samp
