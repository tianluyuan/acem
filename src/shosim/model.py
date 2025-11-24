import numpy as np
from scipy import stats
from .media import Medium


def ltot_scale(m0: 'Medium', m1: 'Medium'):
    return m0.density / m1.density * (1. - 1./m1.nphase) * (1. + 1./m1.nphase) / ((1. - 1./m0.nphase) * (1. + 1./m0.nphase))
        

class RWShower:
    """
    Calculates Cherenkov light yield and profile for EM and Hadronic showers,
    managing density and radiation length as material properties.

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

    G4_MEDIUM = Medium(0.91, 1.33)

    def __init__(self, medium: 'Medium'):
        self.medium = medium
        self._scale = ltot_scale(self.G4_MEDIUM, self.medium)

    def ltot_mean(self, pdg: int, energy: float):
        alpha = self.MEAN_ALPHAS[pdg]
        beta = self.MEAN_BETAS[pdg]
        return alpha * energy**beta * self._scale

    def ltot_sigma(self, pdg: int, energy: float):
        alpha = self.SIGMA_ALPHAS[pdg]
        beta = self.SIGMA_BETAS[pdg]
        return alpha * energy**beta * self._scale

    def ltot(self, pdg: int, energy: float):
        return stats.norm(self.ltot_mean(pdg, energy), self.ltot_sigma(pdg, energy))
    
    def gamma(self, pdg: int, energy: float):
        _a = self.GAMMA_A[pdg](energy)
        _b = self.GAMMA_B[pdg]
        return stats.gamma(_a, scale=self.medium.lrad / _b)

    def dldx(self, pdg: int, energy: float):
        return lambda x: self.ltot_mean(pdg, energy) * self.gamma(pdg, energy).pdf(x)
