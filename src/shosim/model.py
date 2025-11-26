from abc import ABC, abstractmethod
import numpy as np
from numpy.random import Generator
from scipy import stats
from typing import Optional, TYPE_CHECKING
from .media import Medium

if TYPE_CHECKING:
    from scipy.stats._distn_infrastructure import rv_frozen
    import numpy.typing as npt


def ltot_scale(m0: Medium, m1: Medium):
    return m0.density / m1.density * (1. - 1./m1.nphase) * (1. + 1./m1.nphase) / ((1. - 1./m0.nphase) * (1. + 1./m0.nphase))


class Shower:
    """
    A one-dimensional representation of the Cherenkov-weighted track lengths of a particle shower, along the shower axis.

    Parameters
    ----------
    ltot (float) : The total Cherenkov-weighted track length
    shape (rv_frozen) : A frozen scipy.stats distribution that describes the shape of the shower profile
    """
    def __init__(self, ltot: float, shape: 'rv_frozen'):
        self.ltot = ltot
        self.shape = shape

    def __repr__(self) -> str:
        """
        Provides a comprehensive string representation of the object.
        """
        return (
            f'Shower('
            f'ltot={self.ltot}, '
            f'shape={self.shape!r})'
        )

    def dldx(self, x: 'npt.ArrayLike') -> 'npt.ArrayLike':
        return self.ltot * self.shape.pdf(x)
        

class ModelBase(ABC):
    @abstractmethod
    def ltot_dist(self, pdg: int, energy: float) -> 'rv_frozen':
        pass

    @abstractmethod
    def avg(self, pdg: int, energy: float) -> Shower:
        pass

    @abstractmethod
    def sample(self,
               pdg: int,
               energy: float,
               rng: Optional[Generator]) -> Shower:
        pass


class RWShowerModel(ModelBase):
    """
    Generates shower profiles for Cherenkov light yields, given a Medium.

    Based on: https://doi.org/10.1016/j.astropartphys.2013.01.015
    """

    MEAN_ALPHAS = {11: 532.07078881,
                   -11: 532.11320598,
                   22: 532.08540905,
                   211: 333.55182722}
    MEAN_BETAS = {11: 1.00000211,
                  -11: 0.99999254,
                  22: 0.99999877,
                  211: 1.03662217}

    SIGMA_ALPHAS = {11: 5.78170887,
                    -11: 5.73419669,
                    22: 5.66586567,
                    211: 119.20455395}
    SIGMA_BETAS = {11: 0.5,
                   -11: 0.5,
                   22: 0.5,
                   211: 0.80772057}

    GAMMA_A = {
        11: lambda x: 2.01849 + 0.63176 * np.log(x),
        -11: lambda x: 2.00035 + 0.63190 * np.log(x),
        22: lambda x: 2.83923 + 0.58209 * np.log(x),
        211: lambda x: 1.58357292 + 0.41886807 * np.log(x),
    }
    GAMMA_B = {11: 0.63207,
               -11: 0.63008,
               22: 0.64526,
               211: 0.33833116}

    # density and nphase used in Geant4 MC
    G4_MEDIUM = Medium(0.91, 1.33)

    def __init__(self, medium: Medium):
        self.medium = medium
        self._scale = ltot_scale(self.G4_MEDIUM, self.medium)

    def _ltot_mean(self, pdg: int, energy: float) -> float:
        return self.MEAN_ALPHAS[pdg] * energy**self.MEAN_BETAS[pdg] * self._scale

    def _ltot_sigma(self, pdg: int, energy: float) -> float:
        return self.SIGMA_ALPHAS[pdg] * energy**self.SIGMA_BETAS[pdg] * self._scale

    def _shape(self, pdg: int, energy: float) -> 'rv_frozen':
        return stats.gamma(self.GAMMA_A[pdg](energy),
                           scale=self.medium.lrad / self.GAMMA_B[pdg])

    def ltot_dist(self, pdg: int, energy: float) -> 'rv_frozen':
        return stats.norm(self._ltot_mean(pdg, energy), self._ltot_sigma(pdg, energy))
    
    def avg(self, pdg: int, energy: float) -> Shower:
        return Shower(self._ltot_mean(pdg, energy), self._shape(pdg, energy))

    def sample(self,
               pdg: int,
               energy: float,
               rng: Optional[Generator] = None) -> Shower:
        if rng is None:
            rng = np.random.default_rng(42)
        return Shower(self.ltot_dist(pdg, energy).rvs(random_state=rng),
                      self._shape(pdg, energy))


class ShowerModel(ModelBase):
    """
    Generates shower profiles for Cherenkov light yields, given a Medium. Includes fluctuations in shape and nuclear effects for total light yield.

    Based on: TBD
    """
    @staticmethod
    def aprime(a: float) -> float:
        """
        Function to transform a into range (0,1)
        """
        return 1./np.sqrt(a)

    @staticmethod
    def bprime(b: float) -> float:
        """
        Function to transform b into range (0,1)
        """
        return 1./(1.+b**2)

    @staticmethod
    def a(aprime: float) -> float:
        """
        Function to transform a' with domain (0,1) back to a
        """
        return 1./aprime**2

    @staticmethod
    def b(bprime: float) -> float:
        """
        Function to transform b' with domain (0,1) back to b
        """
        return np.sqrt(1./bprime-1.)

    # density and nphase used in FLUKA MC
    FLUKA_MEDIUM = Medium(0.9216, 1.33)

    def __init__(self, medium: Medium):
        self.medium = medium
        self._scale = ltot_scale(self.FLUKA_MEDIUM, self.medium)

    
    def avg(self, pdg: int, energy: float) -> Shower:
        pass

    def sample(self,
               pdg: int,
               energy: float,
               rng: Optional[Generator] = None) ->Shower:
        if rng is None:
            rng = np.random.default_rng(42)
        pass
