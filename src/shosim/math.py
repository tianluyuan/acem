from typing import NamedTuple, Self
import numpy as np
import itertools
from scipy import interpolate

def efn(en, fn, *args):
    return np.exp(fn(np.log10(en), *args))


def qrt(x, t0, t1, t2, t3, t4):
    return t4*x**4 + t3*x**3 + t2*x**2 + t1*x + t0


def cbc(x, t0, t1, t2, t3):
    return t3*x**3 + t2*x**2 + t1*x + t0


def lin(x, t0, t1):
    return t1 * x + t0


def make_knots(
        c_a: int,
        c_b: int,
        c_E: int,
        a_min: float = 0.0,
        a_max: float = 1.0,
        b_min: float = 0.0,
        b_max: float = 1.0,
        E_min: float = 1.0,
        E_max: float = 6.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Makes the knots for the given Coefficients
    Provided a range of valid values for each of the three parameters
    Params:
        a_reg - number of valid regions in the a dimension
        b_reg - number of valid regions in the b dimension
        c_E - number of valid regions in the E dimension
        a_min - minumum value in the a dimension, [Default = 0]
        a_max - maximum value in the a dimension, [Default = 1]
        b_min - minumum value in the b dimension, [Default = 0]
        b_max - maximum value in the b dimension, [Default = 1]
        E_min - minumum value in the E dimension, [Default = 1]
        E_max - maximum value in the E dimension, [Default = 6]
                Note that returned knot values will extend past these values for b-spline fitting purposes
    Return:
        a_k - 1d array of knots for the a dimension
        b_k - 1d array of knots for the b dimension
        E_k - 1d array of knots for the E dimension
    """
    ## Define the knots
    a_k = np.linspace(a_min,a_max,c_a - 3 + 1)
    b_k = np.linspace(b_min,b_max,c_b - 3 + 1)
    E_k = np.linspace(E_min,E_max,c_E - 3 + 1)
    ## add knot values on above and below the range of interest
    a_k = interpolate.interp1d(np.arange(c_a - 3 + 1),a_k,bounds_error=False,fill_value="extrapolate")(np.arange(-3,c_a + 1))
    b_k = interpolate.interp1d(np.arange(c_b - 3 + 1),b_k,bounds_error=False,fill_value="extrapolate")(np.arange(-3,c_b + 1))
    E_k = interpolate.interp1d(np.arange(c_E - 3 + 1),E_k,bounds_error=False,fill_value="extrapolate")(np.arange(-3,c_E + 1))
    return a_k,b_k,E_k


class BSpline(NamedTuple):
    """
    initialize with two parameters
    BSpline coefs: Array with shape (c_a,c_b,c_E) of coefficients for a basis spline and its knots
    knots: tuple (a_k,b_k,E_k) where each element is the 1d array of the knots defining the spline regions along each dimension
                if not supplied, make_knots is called with the default values.
    """
    coefs: np.ndarray
    knots: tuple[np.ndarray, np.ndarray, np.ndarray]
    poly_coefs: np.ndarray

    @classmethod
    def create(cls, coefs: np.ndarray, knots: tuple[np.ndarray, np.ndarray, np.ndarray]) -> Self:
        """
        Sets basis spline coefficients into coefficients in the polynomial basis
        poly_coefs - Array with shape (c_a-3, c_b-3, c_E-3, 4, 4, 4) . 
        poly_coefs[i,j,k,q,r,s] is the coeficient on the a**q b**r E**s term in the space right after knot a_k[i+3], b_k[j+3], and E_k[k+3]
        """
        assert np.all(np.asarray([len(_) for _ in knots]) == np.asarray([_ + 7 for _ in coefs.shape]))
        a_k = knots[0]
        b_k = knots[1]
        E_k = knots[2]

        D_a = a_k[1] - a_k[0]
        D_b = b_k[1] - b_k[0]
        D_E = E_k[1] - E_k[0]
        poly_coefs = np.zeros((coefs.shape[0] - 3,coefs.shape[1] - 3,coefs.shape[2] - 3,4,4,4))
        BSplinePieces_a = np.zeros((coefs.shape[0],4,4))
        BSplinePieces_b = np.zeros((coefs.shape[1],4,4))
        BSplinePieces_E = np.zeros((coefs.shape[2],4,4))

        # BSplinePieces_?[i,j,k] holds the coefficient on the x**k term of the j-th piece of the i-th basis spline
        for i in range(coefs.shape[0]):
            for j in range(4):
                BSplinePieces_a[i,j,:] = cls.BSplinePiece(j,a_k[0],D_a,i)
        for i in range(coefs.shape[1]):
            for j in range(4):
                BSplinePieces_b[i,j,:] = cls.BSplinePiece(j,b_k[0],D_b,i)
        for i in range(coefs.shape[2]):
            for j in range(4):
                BSplinePieces_E[i,j,:] = cls.BSplinePiece(j,E_k[0],D_E,i)
        for q, r, s, l, m, n in itertools.product(*[range(4)]*6):
            poly_coefs[:,:,:,q,r,s] += coefs[l:poly_coefs.shape[0]+l,m:poly_coefs.shape[1]+m,n:poly_coefs.shape[2]+n] \
                * np.tile(BSplinePieces_a[l:poly_coefs.shape[0]+l,3-l,q].reshape((poly_coefs.shape[0],1,1)),(1,poly_coefs.shape[1],poly_coefs.shape[2])) \
                * np.tile(BSplinePieces_b[m:poly_coefs.shape[1]+m,3-m,r].reshape((1,poly_coefs.shape[1],1)),(poly_coefs.shape[0],1,poly_coefs.shape[2])) \
                * np.tile(BSplinePieces_E[n:poly_coefs.shape[2]+n,3-n,s].reshape((1,1,poly_coefs.shape[2])),(poly_coefs.shape[0],poly_coefs.shape[1],1))

        return cls(
            coefs=coefs,
            knots=knots,
            poly_coefs=poly_coefs
        )

    @staticmethod
    def BSplinePiece(
            piece_num: int,
            x_0: float,
            D: float,
            n: int
    ) -> np.ndarray | None:
        """
        constructs the coefficients for a 3rd order BSpline with evenly space knots in the polynomial basis
        Params:
        piece_num - which of the four sections of the BSpline basis element (integer 0 (leftmost) to 3 (rightmost))
        x_0 - leftmost knot position
        D - distance between knots
        n - index of basis spline (0 corresponds to basis element originating from x_0)
        Returns:
        coefs: coefs[i] is the coefficient on the x**i term in the specified section of the basis element
        """
        match piece_num:
            case 0:
                return np.array([-D**3*n**3 - 3*D**2*n**2*x_0 - 3*D*n*x_0**2 - x_0**3,3*D**2*n**2 + 6*D*n*x_0 + 3*x_0**2,-3*D*n - 3*x_0,1])/(6*D**3)
            case 1:
                return np.array([3*D**3*n**3 + 12*D**3*n**2 + 12*D**3*n + 4*D**3 + 9*D**2*n**2*x_0 + 24*D**2*n*x_0 + 12*D**2*x_0 + 9*D*n*x_0**2 + 12*D*x_0**2 + 3*x_0**3,
                                -9*D**2*n**2 - 24*D**2*n - 12*D**2 - 18*D*n*x_0 - 24*D*x_0 - 9*x_0**2,
                                9*D*n + 12*D + 9*x_0,-3])/(6*D**3)
            case 2:
                return np.array([-3*D**3*n**3 - 24*D**3*n**2 - 60*D**3*n - 44*D**3 - 9*D**2*n**2*x_0 - 48*D**2*n*x_0 - 60*D**2*x_0 - 9*D*n*x_0**2 - 24*D*x_0**2 - 3*x_0**3,
                                 9*D**2*n**2 + 48*D**2*n + 60*D**2 + 18*D*n*x_0 + 48*D*x_0 + 9*x_0**2,
                                 -9*D*n - 24*D - 9*x_0,3])/(6*D**3)
            case 3:
                return np.array([D**3*n**3 + 12*D**3*n**2 + 48*D**3*n + 64*D**3 + 3*D**2*n**2*x_0 + 24*D**2*n*x_0 + 48*D**2*x_0 + 3*D*n*x_0**2 + 12*D*x_0**2 + x_0**3,
                                -3*D**2*n**2 - 24*D**2*n - 48*D**2 - 6*D*n*x_0 - 24*D*x_0 - 3*x_0**2,
                                3*D*n + 12*D + 3*x_0,-1])/(6*D**3)

    def __call__(self, a, b, E):
        """
        Params:
            a,b,E - input parameter values
        Returns:
            Result of evaluating BSpline at the provided a,b,E values
        """
        ## if knots aren't specified generate them from default values
        a_k = self.knots[0]
        b_k = self.knots[1]
        E_k = self.knots[2]
        a_i = np.searchsorted(a_k[3:-3], a, side='right')
        b_i = np.searchsorted(b_k[3:-3], b, side='right')
        E_i = np.searchsorted(E_k[3:-3], E, side='right')

        a_i -= (a_i > self.poly_coefs.shape[0]) # so that things don't break at the upper boundaries
        b_i -= (b_i > self.poly_coefs.shape[1])
        E_i -= (E_i > self.poly_coefs.shape[2])
        Z = 0
        for l in range(4):
            for m in range(4):
                for n in range(4):
                    Z += self.poly_coefs[a_i-1,b_i-1,E_i-1,l,m,n] * a**l * b**m * E**n
        return Z
