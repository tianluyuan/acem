from typing import NamedTuple, Self
import itertools
import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from scipy import interpolate
from scipy.optimize import minimize

def efn(en, fn, *args):
    return np.exp(fn(np.log10(en), *args))


def qrt(x, t0, t1, t2, t3, t4):
    return t4*x**4 + t3*x**3 + t2*x**2 + t1*x + t0


def cbc(x, t0, t1, t2, t3):
    return t3*x**3 + t2*x**2 + t1*x + t0


def lin(x, t0, t1):
    return t1 * x + t0


def aprime(a):
    """
    Function to transform a into range (0,1)
    """
    return 1./np.sqrt(a)


def bprime(b):
    """
    Function to transform b into range (0,1)
    """
    return 1./(1.+b**2)


def a(aprime):
    """
    Function to transform a' with domain (0,1) back to a
    """
    return 1./aprime**2


def b(bprime):
    """
    Function to transform b' with domain (0,1) back to b
    """
    return np.sqrt(1./bprime-1.)


class BSpline(NamedTuple):
    """
    instantiate with create factory
    """
    poly_coefs: np.ndarray
    bspl: interpolate.NdBSpline

    @classmethod
    def create(cls,
               knots: tuple[np.ndarray, np.ndarray, np.ndarray],
               coefs: np.ndarray) -> Self:
        """
        Given 3D knots and associated coefs, instantiates a BSpline object

        Additionally converts basis spline coefficients into polynomial
        coefficients and stores them in poly_coefs.
        poly_coefs: Array with shape (c_a-3, c_b-3, c_E-3, 4, 4, 4) . 
        poly_coefs[i,j,k,q,r,s] is the coeficient on the a**q b**r logE**s term in the space right after knot a_k[i+3], b_k[j+3], and E_k[k+3]

        Parameters
        ----------
        knots: a 3-element tuple (a_k,b_k,E_k) where each element is the 1d array
               of the knots defining the spline regions along each dimension
        coefs: Array with shape (c_a,c_b,c_E) of coefficients for a basis spline and its knots

        Returns
        -------
        BSpline object for coefs with corresponding knots and poly_coefs

        """
        assert np.all(np.asarray([len(_) for _ in knots]) == np.asarray([_ + 4 for _ in coefs.shape]))
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
                BSplinePieces_a[i,j,:] = cls._BSplinePiece(j,a_k[0],D_a,i)
        for i in range(coefs.shape[1]):
            for j in range(4):
                BSplinePieces_b[i,j,:] = cls._BSplinePiece(j,b_k[0],D_b,i)
        for i in range(coefs.shape[2]):
            for j in range(4):
                BSplinePieces_E[i,j,:] = cls._BSplinePiece(j,E_k[0],D_E,i)
        for q, r, s, l, m, n in itertools.product(*[range(4)]*6):
            poly_coefs[:,:,:,q,r,s] += coefs[l:poly_coefs.shape[0]+l,m:poly_coefs.shape[1]+m,n:poly_coefs.shape[2]+n] \
                * np.tile(BSplinePieces_a[l:poly_coefs.shape[0]+l,3-l,q].reshape((poly_coefs.shape[0],1,1)),(1,poly_coefs.shape[1],poly_coefs.shape[2])) \
                * np.tile(BSplinePieces_b[m:poly_coefs.shape[1]+m,3-m,r].reshape((1,poly_coefs.shape[1],1)),(poly_coefs.shape[0],1,poly_coefs.shape[2])) \
                * np.tile(BSplinePieces_E[n:poly_coefs.shape[2]+n,3-n,s].reshape((1,1,poly_coefs.shape[2])),(poly_coefs.shape[0],poly_coefs.shape[1],1))

        return cls(
            poly_coefs=poly_coefs,
            bspl=interpolate.NdBSpline(knots, coefs, 3)
        )

    @staticmethod
    def _make_knots(
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

        Parameters
        ----------
        a_reg: number of valid regions in the a dimension
        b_reg: number of valid regions in the b dimension
        c_E: number of valid regions in the logE dimension
        a_min: minumum value in the a dimension, [Default = 0]
        a_max: maximum value in the a dimension, [Default = 1]
        b_min: minumum value in the b dimension, [Default = 0]
        b_max: maximum value in the b dimension, [Default = 1]
        E_min: minumum value in the logE dimension, [Default = 1]
        E_max: maximum value in the logE dimension, [Default = 6]

        Note that returned knot values will extend past these values for b-spline fitting purposes
        Returns
        -------
        a_k: 1d array of knots for the a dimension
        b_k: 1d array of knots for the b dimension
        E_k: 1d array of knots for the logE dimension
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

    @staticmethod
    def _BSplinePiece(
            piece_num: int,
            x_0: float,
            D: float,
            n: int
    ) -> np.ndarray | None:
        """
        constructs the coefficients for a 3rd order BSpline with evenly space knots in the polynomial basis

        Parameters
        ----------
        piece_num: which of the four sections of the BSpline basis element (integer 0 (leftmost) to 3 (rightmost))
        x_0: leftmost knot position
        D: distance between knots
        n: index of basis spline (0 corresponds to basis element originating from x_0)

        Returns
        -------
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

    def __call__(self,
                 aprime: float | npt.ArrayLike,
                 bprime: float | npt.ArrayLike,
                 logE: float | npt.ArrayLike) -> float:
        """
        Parameters
        ----------
        aprime: first parameter for gamma distribution, a', transformed to lie on [0, 1]
        bprime: second parameter for gamma distribution, b', transformed to lie on [0, 1]
        logE: primary particle energy in log10(logE [GeV])

        Returns
        -------
        ndarray: Result of evaluating BSpline at the provided a,b,logE values
        """
        A, B, LogE = np.broadcast_arrays(aprime, bprime, logE)
        return self.bspl(np.asarray([A, B, LogE]).T).T

    def mean(self, logE: float) -> tuple[float, float]:
        """
        Parameters
        ----------
        logE: primary particle energy in log10(logE [GeV])

        Returns
        -------
        tuple[float, float]: mean(a', b') for given logE
        """
        return self.integrate_grid(logE, (1, 0)).sum(), self.integrate_grid(logE, (0, 1)).sum()
    
    def mode(self, logE: float):
        """
        Parameters
        ----------
        logE: primary particle energy in log10(logE [GeV])

        Returns
        -------
        tuple[float, float]: mode(a', b') for given logE
        """
        mini = minimize(lambda x: -self.bspl((x[0], x[1], logE)), [0.5, 0.5],
                        method='Nelder-Mead', bounds=[(0., 1.), (0., 1.)])
        if not mini.success:
            raise RuntimeError(f"Minimization failure in mode search; {mini.message}")
        return mini.x[0], mini.x[1]
            
    def integrate_grid(self,
                       logE: float,
                       moment: tuple[int, int]=(0,0),
                       num_quad_nodes: int=7) -> np.ndarray:
        """
        Integrate each region of the spline at a specified energy
        and return a grid of the integral in each region

        Parameters
        ----------
        logE: log10(logE [GeV])
        num_quad_nodes: number of nodes used for guassian quadrature, [Default = 7]

        Returns
        -------
        ndarray result where result[i,j,k] is the integral over the intersection of
        the i-th a region, j-th b region, and k-th logE region
        """
        a_k, b_k, E_k = self.bspl.t

        ## create nodes for gaussian quadrature
        nodes_1d, weights_1d = np.polynomial.legendre.leggauss(num_quad_nodes)
        weights = np.tile(weights_1d,(num_quad_nodes,1)) * np.tile(weights_1d,(num_quad_nodes,1)).T
        nodes = np.tile(nodes_1d,(num_quad_nodes,1))
        nodes_array = np.tile(nodes.reshape(1,1,num_quad_nodes,num_quad_nodes),(self.poly_coefs.shape[0],self.poly_coefs.shape[1],1,1))
        nodesT_array = np.tile(nodes.T.reshape(1,1,num_quad_nodes,num_quad_nodes),(self.poly_coefs.shape[0],self.poly_coefs.shape[1],1,1))
        # weights_array = np.tile(weights.reshape(1,1,num_quad_nodes,num_quad_nodes),(self.poly_coefs.shape[0],self.poly_coefs.shape[1],1,1))
        E_i = np.searchsorted(E_k[3:-3],logE,side='right')
        E_i -= (E_i > self.poly_coefs.shape[2])

        Z = np.zeros((*self.poly_coefs.shape[:2],num_quad_nodes,num_quad_nodes))
        ## Z[grid of regions a][grid of regions b][grid of nodes a][grid of nodes b]

        l_a = np.tile(a_k[3:-4].reshape(-1,1,1,1),(1,self.poly_coefs.shape[1],num_quad_nodes,num_quad_nodes))
        h_a = np.tile(a_k[4:-3].reshape(-1,1,1,1),(1,self.poly_coefs.shape[1],num_quad_nodes,num_quad_nodes))
        l_b = np.tile(b_k[3:-4].reshape(1,-1,1,1),(self.poly_coefs.shape[0],1,num_quad_nodes,num_quad_nodes))
        h_b = np.tile(b_k[4:-3].reshape(1,-1,1,1),(self.poly_coefs.shape[0],1,num_quad_nodes,num_quad_nodes))

        X_quad = nodes_array * (h_a - l_a) / 2 + (l_a + h_a) / 2
        Y_quad = nodesT_array * (h_b - l_b) / 2 + (l_b + h_b) / 2
        for l, m, n in itertools.product(*[range(4)]*3):
            Z += np.tile(self.poly_coefs[:,:,E_i-1,l,m,n].reshape((self.poly_coefs.shape[0],self.poly_coefs.shape[1],1,1)),(1,1,num_quad_nodes,num_quad_nodes)) \
                * X_quad**l * Y_quad**m * logE**n

        alpha, beta = moment
        moment_kernel = (X_quad**alpha) * (Y_quad**beta)
        return np.sum(np.exp(Z) * moment_kernel * weights * (a_k[1]-a_k[0])*(b_k[1]-b_k[0])/4,axis=(2,3))
            
    def sample(self,
               logE: float,
               size: None | int=None,
               random_state: None | Generator=None) -> tuple | np.ndarray:
        """
        Samples (a', b') for given log10E via rejection sampling

        Parameters
        ----------
        logE: log10(logE [GeV])
        size: number of (a', b') samples to draw
        random_state: random_state state

        Returns
        -------
        ndarray of sampled np.array([(a', b')_0, (a', b')_1, ...])
        """
        if random_state is None:
            random_state = np.random.default_rng()
        
        ap, bp = self.mode(logE)
        f_max = self.__call__(ap, bp, logE)
        samples = []
        _size = 1 if size is None else size
        while len(samples) < _size:
            x_star = random_state.uniform()
            y_star = random_state.uniform()
            z_star = random_state.uniform(0., f_max)
            if z_star <= self.__call__(x_star, y_star, logE):
                samples.append((x_star, y_star))
            
        return samples[0] if size is None else np.asarray(samples)

    def _legacy_sample(self,
                       logE: float,
                       size: None | int=None,
                       random_state: None | Generator=None,
                       sample_depth: int=7,
                       binning_offset: bool=True,
                       num_quad_nodes: int=7) -> tuple | np.ndarray:
        """
        Samples (a', b') for given log10E via binary split algorithm.

        Parameters
        ----------
        logE: log10(logE [GeV])
        size: number of (a', b') samples to draw
        random_state: rng state, if None initialize from scratch [Default: None]
        sample_depth: Number of times to divide regions during binary grid sampling.
                      A sample_depth of n will give a sample precision of of 1/2^n
                      times the size of each spline region [Default: 10]
        binning_offset: Determines whether to random offset to reduce binning
                        errors [Default: True]
        num_quad_nodes: Number of nodes used for gaussian quadrature [Default: 7]

        Returns
        -------
        ndarray of sampled np.array([(a', b')_0, ...])

        Notes
        -----
        The resulting sample does not seem to exactly match rejection sampling,
        but is kept for historical purposes.
        """
        if random_state is None:
            random_state = np.random.default_rng()
        ## Create nodes and weights for 2d gaussian quadrature
        nodes_1d, weights_1d = np.polynomial.legendre.leggauss(num_quad_nodes)
        weights = np.tile(weights_1d, (num_quad_nodes, 1)) * np.tile(weights_1d, (num_quad_nodes, 1)).T

        a_k, b_k, E_k = self.bspl.t
        ## Subrouting to integrate the spline within a specified region
        def integrate(l_a, h_a, l_b, h_b, Coefs_abE):
            nodes_ = nodes_1d.reshape(1, num_quad_nodes, 1)
            nodesT_ = nodes_1d.reshape(1, 1, num_quad_nodes)
            Z = np.zeros([Coefs_abE.shape[0], num_quad_nodes, num_quad_nodes])
            for l in range(4):
                for m in range(4):
                    Z += Coefs_abE[:, :, :, l, m] * \
                                np.tile((nodes_*(h_a-l_a)/2 + (l_a+h_a)/2)**l, (1, 1, num_quad_nodes)) * \
                                np.tile((nodesT_*(h_b-l_b)/2 + (l_b+h_b)/2)**m, (1, num_quad_nodes, 1))
            return np.sum(np.exp(Z) * weights.reshape(1, num_quad_nodes, num_quad_nodes), axis=(-2, -1), keepdims=True) * (h_a-l_a)*(h_b-l_b)/4
        E_i = np.searchsorted(E_k[3:-3], logE, side="right")
        E_i -= (E_i > self.poly_coefs.shape[2])
        CoefsE = (self.poly_coefs[:, :, E_i-1, :, :, :] * np.array([1, logE, logE**2, logE**3]).reshape(1, 1, 1, 1, 4)).sum(axis = -1)

        ## Generate the indices for the regions by integrating over every region and randomly selecting based on their total probability
        nodes = np.tile(nodes_1d, (num_quad_nodes, 1))
        nodes_array = np.tile(nodes.reshape(1, 1, num_quad_nodes, num_quad_nodes), (CoefsE.shape[0], CoefsE.shape[1], 1, 1))
        nodesT_array = np.tile(nodes.T.reshape(1, 1, num_quad_nodes, num_quad_nodes), (CoefsE.shape[0], CoefsE.shape[1], 1, 1))

        Z = np.zeros((*CoefsE.shape[:2], num_quad_nodes, num_quad_nodes))
        """
        Z[grid of regions a][grid of regions b][grid of nodes a][grid of nodes b]
        """
        l_a = np.tile(a_k[3:-4].reshape(-1, 1, 1, 1), (1, CoefsE.shape[1], num_quad_nodes, num_quad_nodes))
        h_a = np.tile(a_k[4:-3].reshape(-1, 1, 1, 1), (1, CoefsE.shape[1], num_quad_nodes, num_quad_nodes))
        l_b = np.tile(b_k[3:-4].reshape(1, -1, 1, 1), (CoefsE.shape[0], 1, num_quad_nodes, num_quad_nodes))
        h_b = np.tile(b_k[4:-3].reshape(1, -1, 1, 1), (CoefsE.shape[0], 1, num_quad_nodes, num_quad_nodes))

        for l in range(4):
            for m in range(4):
                Z += np.tile(CoefsE[:, :, l, m].reshape(CoefsE.shape[0], CoefsE.shape[1], 1, 1), (1, 1, num_quad_nodes, num_quad_nodes)) \
                        * (nodes_array*(h_a-l_a)/2 + (l_a+h_a)/2)**l * (nodesT_array*(h_b-l_b)/2 + (l_b+h_b)/2)**m
        integrated_grid = np.sum(np.exp(Z) * weights * (a_k[1]-a_k[0])*(b_k[1]-b_k[0])/4, axis=(2, 3))

        ranges = np.insert(integrated_grid.reshape(-1).cumsum(), 0, 0)
        _size = 1 if size is None else size
        indices = np.searchsorted(ranges, random_state.random(size=_size)*ranges[-1]) - 1
        a_regions = indices // CoefsE.shape[1]
        b_regions = indices % CoefsE.shape[1]


        Coefs_abE = CoefsE[a_regions, b_regions, :, :].reshape(_size, 1, 1, 4, 4)

        l_a = np.reshape(a_k[a_regions + 3], (_size, 1, 1))
        h_a = np.reshape(a_k[a_regions + 4], (_size, 1, 1))
        l_b = np.reshape(b_k[b_regions + 3], (_size, 1, 1))
        h_b = np.reshape(b_k[b_regions + 4], (_size, 1, 1))

        for _ in range(sample_depth):
            m_a = (l_a + h_a)/2
            # Integrate left and right of m_a
            Z_left = integrate(l_a, m_a, l_b, h_b, Coefs_abE)
            Z_right = integrate(m_a, h_a, l_b, h_b, Coefs_abE)
            choose_left = random_state.random(size=(_size, 1, 1)) < (Z_left / (Z_left + Z_right))
            l_a = np.where(choose_left, l_a, m_a)
            h_a = np.where(choose_left, m_a, h_a)

            m_b = (l_b + h_b)/2
            # Integrate above and bellow
            Z_bottom = integrate(l_a, h_a, l_b, m_b, Coefs_abE)
            Z_top = integrate(l_a, h_a, m_b, h_b, Coefs_abE)
            choose_bottom = random_state.random(size=(_size, 1, 1)) < (Z_bottom / (Z_bottom + Z_top))
            l_b = np.where(choose_bottom, l_b, m_b)
            h_b = np.where(choose_bottom, m_b, h_b)
        if binning_offset:
            res_a = l_a + random_state.random(size=(_size, 1, 1))*(a_k[1]-a_k[0])/2**sample_depth
            res_b = l_b + random_state.random(size=(_size, 1, 1))*(b_k[1]-b_k[0])/2**sample_depth
        else:
            res_a = (l_a + h_a)/2
            res_b = (l_b + h_b)/2
        _res = np.asarray([res_a.reshape(-1), res_b.reshape(-1)]).T
        return _res[0] if size is None else _res
