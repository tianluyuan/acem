import numpy as np
from scipy import stats

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .media import Medium

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

    G4_DENSITY = 0.91
    G4_NPHASE = 1.33

    def __init__(self, medium: 'Medium'):
        self.density = medium.density
        self.lrad = medium.lrad
        self.nphase = medium.nphase

    def ltot_mean(self, pdg: int, energy: float):
        alpha = self.MEAN_ALPHAS[pdg]
        beta = self.MEAN_BETAS[pdg]
        return alpha * energy**beta * self.G4_DENSITY / self.density

    def ltot_sigma(self, pdg: int, energy: float):
        alpha = self.SIGMA_ALPHAS[pdg]
        beta = self.SIGMA_BETAS[pdg]
        return alpha * energy**beta * self.G4_DENSITY / self.density

    def ltot(self, pdg: int, energy: float):
        return stats.norm(self.ltot_mean(pdg, energy), self.ltot_sigma(pdg, energy))
    

    def gamma(self, pdg: int, energy: float):
        a_em = self.GAMMA_A[pdg](energy)
        b_em = self.GAMMA_B[pdg]
        return stats.gamma(a_em, scale=self.lrad / b_em)

    
    def dldx(self, pdg: int, energy: float):
        return lambda x: self.ltot_mean(pdg, energy) * self.gamma(pdg, energy).pdf(x)
