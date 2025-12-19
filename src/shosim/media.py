from typing import NamedTuple

class Medium(NamedTuple):
    """
    A representation of the physical properties of the detection medium.

    This class stores the density and optical parameters required for 
    properly normalizing the Cherenkov light yield.

    Parameters
    ----------
    density : float
        The mass density of the medium in g/cm³
    nphase : float
        The phase refractive index of the medium
    x0 : float, optional
        The radiation length of the medium in g/cm². The default is 36.08.
    """
    density: float
    nphase: float
    x0: float = 36.08

    @property
    def lrad(self) -> float:
        return self.x0 / self.density

# standard ice
ICE = Medium(0.917, 1.31)

# standard water
WATER = Medium(1., 1.33)

# ice at center of IceCube
IC3 = Medium(0.9216, 1.3195)

# seawater at depth
SEA = Medium(1.044, 1.35)
