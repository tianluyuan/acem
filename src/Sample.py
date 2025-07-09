#!/usr/bin/env python

# Given a Coefs_ab file that represents the probability distribution of the a and b parameteres, we use this function to sample from it

# In[122]:


import numpy as np
from scipy.interpolate import interp1d


# In[123]:

## Create knot locations
def get_knots(Coefs, Energy_Min=10, Energy_Max=1e6):
    a_k = interp1d(np.arange(Coefs.shape[0] + 1), np.linspace(0, 1, Coefs.shape[0] + 1),
                   bounds_error=False, fill_value="extrapolate")(np.arange(-3, Coefs.shape[0] + 4))
    b_k = interp1d(np.arange(Coefs.shape[1] + 1), np.linspace(0, 1, Coefs.shape[1] + 1),
                   bounds_error=False, fill_value="extrapolate")(np.arange(-3, Coefs.shape[1] + 4))
    E_k = interp1d(np.arange(Coefs.shape[2] + 1), np.linspace(np.log10(Energy_Min), np.log10(Energy_Max), Coefs.shape[2] + 1),
                   bounds_error=False, fill_value="extrapolate")(np.arange(-3, Coefs.shape[2] + 4))
    return a_k, b_k, E_k

"""
Creates samples from the spline specified by Coefs, at a specified energy E
Required Parameters:
    Coefs: Coefficients that represent the spline
    E: Energy value (GeV) at which we are taking samples
    num_samples: The number of samples to take
Optional Parameters:
    sample_depth: Number of times to divide regions during binary grid sampling. A sample_depth
                  of n will give a sample precision of of 1/2^n times the size of each spline region
                  [Default: 10]
    binning_offset: Determines whether to random offset to reduce binning errors [Default: True]
    num_quad_nodes: Number of nodes used for gaussian quadrature [Default: 7]
    raw_output: Determines whether to output raw values from the spline, or their transformed values [Default: False]
    ga_inv: Inverse transform function for the a parameter [Default: a**2]
    gb_inv: Inverse transform function for the b parameter [Defaule sqrt((1/b) - 1)]
"""
def sample_ab(Coefs, E, num_samples,
              sample_depth=7, binning_offset=True,
              num_quad_nodes=7, raw_output=False,
              ga_inv=lambda a: a**-2, gb_inv=lambda b: np.sqrt(b**-1 - 1),
              rng=np.random.default_rng(250528)):
    ## Create nodes and weights for 2d gaussian quadrature
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(num_quad_nodes)
    weights = np.tile(weights_1d, (num_quad_nodes, 1)) * np.tile(weights_1d, (num_quad_nodes, 1)).T

    E = np.log10(E)
    a_k, b_k, E_k = get_knots(Coefs)
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
    E_i = np.searchsorted(E_k[3:-3], E, side="right")
    E_i -= (E_i > Coefs.shape[2])
    CoefsE = (Coefs[:, :, E_i-1, :, :, :] * np.array([1, E, E**2, E**3]).reshape(1, 1, 1, 1, 4)).sum(axis = -1)

    ## Generate the indices for the regions by integrating over every region and randomly selecting based on their total probability
    nodes = np.tile(nodes_1d, (num_quad_nodes, 1))
    nodes_array = np.tile(nodes.reshape(1, 1, num_quad_nodes, num_quad_nodes), (CoefsE.shape[0], CoefsE.shape[1], 1, 1))
    nodesT_array = np.tile(nodes.T.reshape(1, 1, num_quad_nodes, num_quad_nodes), (CoefsE.shape[0], CoefsE.shape[1], 1, 1))
    weights_array = np.tile(weights.reshape(1, 1, num_quad_nodes, num_quad_nodes), (CoefsE.shape[0], CoefsE.shape[1], 1, 1))

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
    indices = np.searchsorted(ranges, rng.random(size=num_samples)*ranges[-1]) - 1
    a_regions = indices // CoefsE.shape[1]
    b_regions = indices % CoefsE.shape[1]


    Coefs_abE = CoefsE[a_regions, b_regions, :, :].reshape(num_samples, 1, 1, 4, 4)

    l_a = np.reshape(a_k[a_regions + 3], (num_samples, 1, 1))
    h_a = np.reshape(a_k[a_regions + 4], (num_samples, 1, 1))
    l_b = np.reshape(b_k[b_regions + 3], (num_samples, 1, 1))
    h_b = np.reshape(b_k[b_regions + 4], (num_samples, 1, 1))

    for division in range(sample_depth):
        m_a = (l_a + h_a)/2
        # Integrate left and right of m_a
        Z_left = integrate(l_a, m_a, l_b, h_b, Coefs_abE)
        Z_right = integrate(m_a, h_a, l_b, h_b, Coefs_abE)
        choose_left = rng.random(size=(num_samples, 1, 1)) < (Z_left / (Z_left + Z_right))
        l_a = np.where(choose_left, l_a, m_a)
        h_a = np.where(choose_left, m_a, h_a)

        m_b = (l_b + h_b)/2
        # Integrate above and bellow
        Z_bottom = integrate(l_a, h_a, l_b, m_b, Coefs_abE)
        Z_top = integrate(l_a, h_a, m_b, h_b, Coefs_abE)
        choose_bottom = rng.random(size=(num_samples, 1, 1)) < (Z_bottom / (Z_bottom + Z_top))
        l_b = np.where(choose_bottom, l_b, m_b)
        h_b = np.where(choose_bottom, m_b, h_b)
    if binning_offset:
        res_a = l_a + rng.random(size=(num_samples, 1, 1))*(a_k[1]-a_k[0])/2**sample_depth
        res_b = l_b + rng.random(size=(num_samples, 1, 1))*(b_k[1]-b_k[0])/2**sample_depth
    else:
        res_a = (l_a + h_a)/2
        res_b = (l_b + h_b)/2
    if raw_output:
        return res_a.reshape(-1), res_b.reshape(-1)
    return np.vectorize(ga_inv)(res_a.reshape(-1)), np.vectorize(gb_inv)(res_b.reshape(-1))

# In[121]:


"""
Integrate each region of the spline at a specified energy and return a grid of the integral in region
Useful for renormalization of spline and testing
"""
def integrate_grid(Coefs, E, num_quad_nodes=7):
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(num_quad_nodes)
    weights = np.tile(weights_1d, (num_quad_nodes, 1)) * np.tile(weights_1d, (num_quad_nodes, 1)).T
    nodes = np.tile(nodes_1d, (num_quad_nodes, 1))
    nodes_array = np.tile(nodes.reshape(1, 1, num_quad_nodes, num_quad_nodes), (Coefs.shape[0], Coefs.shape[1], 1, 1))
    nodesT_array = np.tile(nodes.T.reshape(1, 1, num_quad_nodes, num_quad_nodes), (Coefs.shape[0], Coefs.shape[1], 1, 1))
    weights_array = np.tile(weights.reshape(1, 1, num_quad_nodes, num_quad_nodes), (Coefs.shape[0], Coefs.shape[1], 1, 1))
    E_i = np.vectorize(lambda E : bisect.bisect_right(E_k[3:-3], E))(E)
    E_i -= (E_i > Coefs.shape[2])

    Z = np.zeros((*Coefs.shape[:2], num_quad_nodes, num_quad_nodes))
    """
    Z[grid of regions a][grid of regions b][grid of nodes a][grid of nodes b]
    """
    l_a = np.tile(a_k[3:-4].reshape(-1, 1, 1, 1), (1, Coefs.shape[1], num_quad_nodes, num_quad_nodes))
    h_a = np.tile(a_k[4:-3].reshape(-1, 1, 1, 1), (1, Coefs.shape[1], num_quad_nodes, num_quad_nodes))
    l_b = np.tile(b_k[3:-4].reshape(1, -1, 1, 1), (Coefs.shape[0], 1, num_quad_nodes, num_quad_nodes))
    h_b = np.tile(b_k[4:-3].reshape(1, -1, 1, 1), (Coefs.shape[0], 1, num_quad_nodes, num_quad_nodes))

    for l in range(4):
        for m in range(4):
            for n in range(4):
                Z += np.tile(Coefs[:, :, E_i-1, l, m, n].reshape((Coefs.shape[0], Coefs.shape[1], 1, 1)), (1, 1, num_quad_nodes, num_quad_nodes)) \
                        * (nodes_array*(h_a-l_a)/2 + (l_a+h_a)/2)**l * (nodesT_array*(h_b-l_b)/2 + (l_b+h_b)/2)**m * E**n
    return np.sum(np.exp(Z) * weights * (a_k[1]-a_k[0])*(b_k[1]-b_k[0])/4, axis=(2, 3))

