from dataclasses import dataclass
import itertools
from functools import cached_property
import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from scipy import interpolate
from scipy.optimize import minimize

def efn(en, fn, *args):
    return np.exp(fn(np.log10(en), *args))


def spt(x, t0, t1, t2, t3, t4, t5, t6, t7):
    return np.polyval([t7, t6, t5, t4, t3, t2, t1, t0], x)

def sxt(x, t0, t1, t2, t3, t4, t5, t6):
    return np.polyval([t6, t5, t4, t3, t2, t1, t0], x)


def qnt(x, t0, t1, t2, t3, t4, t5):
    return np.polyval([t5, t4, t3, t2, t1, t0], x)


def qrt(x, t0, t1, t2, t3, t4):
    return np.polyval([t4, t3, t2, t1, t0], x)


def cbc(x, t0, t1, t2, t3):
    return np.polyval([t3, t2, t1, t0], x)


def qdt(x, t0, t1, t2):
    return np.polyval([t2, t1, t0], x)


def lin(x, t0, t1):
    return np.polyval([t1, t0], x)


def aprime(a):
    """
    Function to transform a into range (0,1)

    >>> a = np.logspace(0., 6., 100)
    >>> np.all(aprime(a) <= 1.) and np.all(0. < aprime(a))
    np.True_
    """
    return 1./np.sqrt(a)


def bprime(b):
    """
    Function to transform b into range (0,1)

    >>> b = np.linspace(0.00, 100., 1000)
    >>> np.all(bprime(b) <= 1.) and np.all(0. < bprime(b))
    np.True_
    """
    return 1./(1.+b**2)


def a(aprime):
    """
    Function to transform a' with domain (0,1) back to a

    >>> x = np.linspace(1.00, 100., 1000)
    >>> np.all(np.abs(a(aprime(x)) - x) < 1e-8)
    np.True_
    """
    return 1./aprime**2


def b(bprime):
    """
    Function to transform b' with domain (0,1) back to b

    >>> x = np.linspace(0.00, 100., 1000)
    >>> np.all(np.abs(b(bprime(x)) - x) < 1e-8)
    np.True_
    """
    return np.sqrt(1./bprime-1.)


@dataclass(frozen=True)
class BSpline3D:
    bspl: interpolate.NdBSpline

    def __post_init__(self):
        assert np.all([_ == 3 for _ in self.bspl.k])
        assert len(self.bspl.t) == 3
        assert self.bspl.c.ndim == 3
        
    @cached_property
    def c_poly(self) -> np.ndarray: 
        """
        Returns
        -------
        Returns the converted basis spline coefficients in terms of polynomial
        coefficients.

        c_poly: Array with shape (c_a-3, c_b-3, c_E-3, 4, 4, 4) . 
        c_poly[i,j,k,q,r,s] is the coeficient on the a**q b**r logE**s term in the space right after knot a_k[i+3], b_k[j+3], and E_k[k+3]
        BSpline3D object

        """
        a_k, b_k, E_k = self.bspl.t

        D_a = a_k[1] - a_k[0]
        D_b = b_k[1] - b_k[0]
        D_E = E_k[1] - E_k[0]
        c_poly = np.zeros((self.bspl.c.shape[0] - 3,self.bspl.c.shape[1] - 3,self.bspl.c.shape[2] - 3,4,4,4))
        BSplinePieces_a = np.zeros((self.bspl.c.shape[0],4,4))
        BSplinePieces_b = np.zeros((self.bspl.c.shape[1],4,4))
        BSplinePieces_E = np.zeros((self.bspl.c.shape[2],4,4))

        # BSplinePieces_?[i,j,k] holds the coefficient on the x**k term of the j-th piece of the i-th basis spline
        for i in range(self.bspl.c.shape[0]):
            for j in range(4):
                BSplinePieces_a[i,j,:] = self._BSplinePiece(j,a_k[0],D_a,i)
        for i in range(self.bspl.c.shape[1]):
            for j in range(4):
                BSplinePieces_b[i,j,:] = self._BSplinePiece(j,b_k[0],D_b,i)
        for i in range(self.bspl.c.shape[2]):
            for j in range(4):
                BSplinePieces_E[i,j,:] = self._BSplinePiece(j,E_k[0],D_E,i)
        for q, r, s, k, m, n in itertools.product(*[range(4)]*6):
            c_poly[:,:,:,q,r,s] += self.bspl.c[k:c_poly.shape[0]+k,m:c_poly.shape[1]+m,n:c_poly.shape[2]+n] \
                * np.tile(BSplinePieces_a[k:c_poly.shape[0]+k,3-k,q].reshape((c_poly.shape[0],1,1)),(1,c_poly.shape[1],c_poly.shape[2])) \
                * np.tile(BSplinePieces_b[m:c_poly.shape[1]+m,3-m,r].reshape((1,c_poly.shape[1],1)),(c_poly.shape[0],1,c_poly.shape[2])) \
                * np.tile(BSplinePieces_E[n:c_poly.shape[2]+n,3-n,s].reshape((1,1,c_poly.shape[2])),(c_poly.shape[0],c_poly.shape[1],1))
        return c_poly

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
                 logE: float | npt.ArrayLike) -> np.ndarray:
        """
        Parameters
        ----------
        aprime: first parameter for gamma distribution, a', transformed to lie on [0, 1]
        bprime: second parameter for gamma distribution, b', transformed to lie on [0, 1]
        logE: primary particle energy in log10(E [GeV])

        Returns
        -------
        ndarray: Result of evaluating BSpline at the provided a,b,logE values

        Notes
        -----
        The return values correspond to ln(pdf(a', b'; logE))
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
        for seed in [[0.99, 0.99],[0.01, 0.01]]:
            _mini = minimize(lambda x: -self.bspl((x[0], x[1], logE)), seed,
                             method='Nelder-Mead', bounds=[(0., 1.), (0., 1.)])
            if _mini.fun < mini.fun:
                mini = _mini

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
        nodes_array = np.tile(nodes.reshape(1,1,num_quad_nodes,num_quad_nodes),(self.c_poly.shape[0],self.c_poly.shape[1],1,1))
        nodesT_array = np.tile(nodes.T.reshape(1,1,num_quad_nodes,num_quad_nodes),(self.c_poly.shape[0],self.c_poly.shape[1],1,1))
        # weights_array = np.tile(weights.reshape(1,1,num_quad_nodes,num_quad_nodes),(self.c_poly.shape[0],self.c_poly.shape[1],1,1))
        E_i = np.searchsorted(E_k[3:-3],logE,side='right')
        E_i -= (E_i > self.c_poly.shape[2])

        Z = np.zeros((*self.c_poly.shape[:2],num_quad_nodes,num_quad_nodes))
        ## Z[grid of regions a][grid of regions b][grid of nodes a][grid of nodes b]

        l_a = np.tile(a_k[3:-4].reshape(-1,1,1,1),(1,self.c_poly.shape[1],num_quad_nodes,num_quad_nodes))
        h_a = np.tile(a_k[4:-3].reshape(-1,1,1,1),(1,self.c_poly.shape[1],num_quad_nodes,num_quad_nodes))
        l_b = np.tile(b_k[3:-4].reshape(1,-1,1,1),(self.c_poly.shape[0],1,num_quad_nodes,num_quad_nodes))
        h_b = np.tile(b_k[4:-3].reshape(1,-1,1,1),(self.c_poly.shape[0],1,num_quad_nodes,num_quad_nodes))

        X_quad = nodes_array * (h_a - l_a) / 2 + (l_a + h_a) / 2
        Y_quad = nodesT_array * (h_b - l_b) / 2 + (l_b + h_b) / 2
        for k, m, n in itertools.product(*[range(4)]*3):
            Z += np.tile(self.c_poly[:,:,E_i-1,k,m,n].reshape((self.c_poly.shape[0],self.c_poly.shape[1],1,1)),(1,1,num_quad_nodes,num_quad_nodes)) \
                * X_quad**k * Y_quad**m * logE**n

        alpha, beta = moment
        moment_kernel = (X_quad**alpha) * (Y_quad**beta)
        return np.sum(np.exp(Z) * moment_kernel * weights * (a_k[1]-a_k[0])*(b_k[1]-b_k[0])/4,axis=(2,3))
            
    def sample(self,
               logE: float,
               size: None | int=None,
               random_state: None | Generator=None) -> np.ndarray:
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
        f_max = np.exp(self.__call__(ap, bp, logE))
        samples = []
        _size = 1 if size is None else size
        current_count = 0

        oversample_factor = int(f_max) + 1
        while current_count < _size:
            n_remaining = _size - current_count
            batch_size = n_remaining * oversample_factor

            x_stars = random_state.uniform(size=batch_size)
            y_stars = random_state.uniform(size=batch_size)
            z_stars = random_state.uniform(0., f_max, size=batch_size)

            mask = z_stars <= np.exp(self.__call__(x_stars, y_stars, logE))

            batch_samples = np.column_stack((x_stars[mask], y_stars[mask]))

            samples.append(batch_samples)
            current_count += len(batch_samples)

        result = np.vstack(samples)[:_size]
        return result[0] if size is None else result

    def _igrid_sample(self,
                      logE: float,
                      size: None | int=None,
                      random_state: None | Generator=None,
                      sample_depth: int=7,
                      binning_offset: bool=True,
                      num_quad_nodes: int=7) -> np.ndarray:
        """
        Samples (a', b') for given log10E via iterative grid search.

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
            for k in range(4):
                for m in range(4):
                    Z += Coefs_abE[:, :, :, k, m] * \
                                np.tile((nodes_*(h_a-l_a)/2 + (l_a+h_a)/2)**k, (1, 1, num_quad_nodes)) * \
                                np.tile((nodesT_*(h_b-l_b)/2 + (l_b+h_b)/2)**m, (1, num_quad_nodes, 1))
            return np.sum(np.exp(Z) * weights.reshape(1, num_quad_nodes, num_quad_nodes), axis=(-2, -1), keepdims=True) * (h_a-l_a)*(h_b-l_b)/4
        E_i = np.searchsorted(E_k[3:-3], logE, side="right")
        E_i -= (E_i > self.c_poly.shape[2])
        CoefsE = (self.c_poly[:, :, E_i-1, :, :, :] * np.array([1, logE, logE**2, logE**3]).reshape(1, 1, 1, 1, 4)).sum(axis = -1)

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

        for k in range(4):
            for m in range(4):
                Z += np.tile(CoefsE[:, :, k, m].reshape(CoefsE.shape[0], CoefsE.shape[1], 1, 1), (1, 1, num_quad_nodes, num_quad_nodes)) \
                        * (nodes_array*(h_a-l_a)/2 + (l_a+h_a)/2)**k * (nodesT_array*(h_b-l_b)/2 + (l_b+h_b)/2)**m
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

    def _legacy_eval(self,
                     aprime: float | npt.ArrayLike,
                     bprime: float | npt.ArrayLike,
                     logE: float | npt.ArrayLike) -> np.ndarray:
        """
        Parameters
        ----------
        aprime: first parameter for gamma distribution, a', transformed to lie on [0, 1]
        bprime: second parameter for gamma distribution, b', transformed to lie on [0, 1]
        logE: primary particle energy in log10(E [GeV])

        Returns
        -------
        ndarray: Result of evaluating BSpline at the provided a,b,logE values

        Notes
        -----
        The return values correspond to ln(pdf(a', b'; logE))
        """
        a_k, b_k, E_k = self.bspl.t
        a_i = np.searchsorted(a_k[3:-3], aprime, side='right')
        b_i = np.searchsorted(b_k[3:-3], bprime, side='right')
        E_i = np.searchsorted(E_k[3:-3], logE, side='right')

        a_i -= (a_i > self.c_poly.shape[0]) # so that things don't break at the upper boundaries
        b_i -= (b_i > self.c_poly.shape[1])
        E_i -= (E_i > self.c_poly.shape[2])
        Z = 0.
        for k in range(4):
            for m in range(4):
                for n in range(4):
                    Z += self.c_poly[a_i-1,b_i-1,E_i-1,k,m,n] * aprime**k * bprime**m * logE**n
        return Z
