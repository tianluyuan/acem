class Medium:
    def __init__(self, density: float, nphase: float, x0: float=36.08):
        self.density = density
        self.nphase = nphase
        self.x0 = x0

    @property
    def lrad(self):
        return self.x0 / self.density

ICE = Medium(0.91, 1.33)
WATER = Medium(1., 1.33)
IC3 = Medium(0.9216, 1.3195)
