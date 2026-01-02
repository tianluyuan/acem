#!/usr/bin/env python
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse
from acem import maths, util
from importlib.resources import files, as_file

### Fit Parameters
dif_orders = [3, 3, 3] # order of difference to penalize for smoothing [a,b,E]
smoothness = [0.1, 0.1, 0.01] # degree of smoothing along each dimension [a,b,E]
# c_a = 12 # Number of basis splines along a-dimension
# c_b = 12 # Number of basis splines along b-dimension
# c_E = 6 # Number of basis splines along E-dimension
c_a = 17 # Number of basis splines along a-dimension
c_b = 17 # Number of basis splines along b-dimension
n_a = 450 # Number of histogram bins along a-dimension
n_b = 450 # Number of histogram bins along b-dimension
deg = 3 # degree of the BSpline

## Define the ab ranges
ab_min = 0
ab_max = 1

### Fitting process parameters
num_iters = 1000 # Number of iterations of least square regression for fitting, can be quit early
perform_likelihood_test = True # Perform likelihood test on test sample for monitoring progress
test_sample_size = 4301 # Number of elements in the testing data set

## other useful constants
bins_a = np.linspace(ab_min,ab_max,n_a + 1)
bins_b = np.linspace(ab_min,ab_max,n_b + 1)

ga = maths.aprime # Function to transform a values into range (0,1)
gb = maths.bprime # Function to transform a values into range (0,1)


'''
perform a likelihood_test on the provided sample using the given Coefs
Params:
    a_sample: array of a values in the test sample
    b_sample: array of b values in the test sample
    E: log energy value at which we are testing
    Coefs: coefficients defining the model we are testing
           Coefs[i,j,k,q,r,s] is the coefficient on the (a**q b**r E**s) term in the intersection of the 
           i-th a region, j-th b region, and k-th E region
Returns:
    log-likelihood of the given sample
'''
def likelihood_test(a_sample,b_sample,E,BSpl):
    ## If knots aren't specified, generate them from default values
    norm_factor = BSpl.integrate_grid(E).sum()
    # print('... normalization:', norm_factor)
    return BSpl(a_sample,b_sample,E).sum() - (np.size(a_sample) * np.log(norm_factor))

'''
Constructs difference matrix of specified size and order, used for penalizing differences
'''
def dif_mat(size,order):
    D1 = np.identity(size) - np.vstack((np.zeros((1,size)),np.identity(size)[:-1,:]))
    D = np.identity(size)
    k = order
    while k > 0:
        D = D @ D1
        k -= 1
    return D[order:,:]

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fitting script')
    parser.add_argument('particles', nargs='+')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show fitted a\', b\' distributions for each E slice')
    args = parser.parse_args()

    a_k = np.linspace(ab_min,ab_max,c_a - deg + 1)
    b_k = np.linspace(ab_min,ab_max,c_b - deg + 1)

    ## add knot values on above and below the range of interest
    a_k = sc.interpolate.interp1d(np.arange(c_a - deg + 1),a_k,bounds_error=False,fill_value='extrapolate')(np.arange(-deg,c_a + 1))
    b_k = sc.interpolate.interp1d(np.arange(c_b - deg + 1),b_k,bounds_error=False,fill_value='extrapolate')(np.arange(-deg,c_b + 1))
    for particle in args.particles:
        Dat = util.load_batch(f'fluka/DataOutputs_{particle}/*.csv',
                              clean=particle not in ('ELECTRON', 'PHOTON'))
        energies = list(Dat.keys())
        log_ens = np.log10(energies) # log (base 10) of the energy values used for fitting
        n_E = len(log_ens) # Number of energy levels used for fitting
        c_E = int(log_ens[-1] - log_ens[0]) + 3 # Number of basis splines along E-dimension
        E_k = np.linspace(log_ens[0],log_ens[-1],c_E - deg + 1)
        E_k = sc.interpolate.interp1d(np.arange(c_E - deg + 1),E_k,bounds_error=False,fill_value='extrapolate')(np.arange(-deg,c_E + 1))
        knots = (a_k, b_k, E_k)
        theta_0 = np.random.default_rng(250611).random(c_a*c_b*c_E) # Initial guess for spline parameters

        ## Make Y matrix of histogram values to fit 
        Y = np.zeros((n_a,n_b,n_E))
        S = np.zeros((n_a,n_b,n_E))
        for i in range(n_E):
            df = Dat[energies[i]]
            avals = ga(df.gammaA)
            bvals = gb(df.gammaB)
            Y[:,:,i],_,_ = np.histogram2d(avals,bvals,bins=(bins_a,bins_b))#,density=True)
            S[:,:,i] = len(avals) * (bins_a[1] - bins_a[0]) * (bins_b[1] - bins_b[0])
            if args.show:
                plt.hist2d(avals,bvals,bins=(bins_a,bins_b),#density=True,
                           norm=colors.LogNorm())
                plt.xlabel(r"$a' = 1 / \sqrt{a}$")
                plt.ylabel(r"$b' = 1/(1 + b^2)$")
                plt.show()
                plt.close('all')

        # Make B_i and G(B_i)
        x_a,x_b,x_E = (bins_a[1:] + bins_a[:-1])/2, (bins_b[1:] + bins_b[:-1])/2, log_ens
        B_a = np.zeros((n_a,c_a))
        B_b = np.zeros((n_b,c_b))
        B_E = np.zeros((n_E,c_E))
        for i in range(n_a):
            for j in range(c_a):
                B_a[i,j] = sc.interpolate.BSpline.basis_element(a_k[j:j + deg + 2],extrapolate = False)(x_a[i])
                if np.isnan(B_a[i,j]):
                    B_a[i,j] = 0
        for i in range(n_b):
            for j in range(c_b):
                B_b[i,j] = sc.interpolate.BSpline.basis_element(b_k[j:j + deg + 2],extrapolate = False)(x_b[i])
                if np.isnan(B_b[i,j]):
                    B_b[i,j] = 0
        for i in range(n_E):
            for j in range(c_E):
                B_E[i,j] = sc.interpolate.BSpline.basis_element(E_k[j:j + deg + 2],extrapolate = False)(x_E[i])
                if np.isnan(B_E[i,j]):
                    B_E[i,j] = 0
        B_a_pinv = np.linalg.pinv(B_a)
        B_b_pinv = np.linalg.pinv(B_b)
        B_E_pinv = np.linalg.pinv(B_E)
        GB_a = np.kron(B_a,np.ones(c_a))*np.kron(np.ones(c_a),B_a)
        GB_b = np.kron(B_b,np.ones(c_b))*np.kron(np.ones(c_b),B_b)
        GB_E = np.kron(B_E,np.ones(c_E))*np.kron(np.ones(c_E),B_E)

        GB_a_sT = sc.sparse.csr_array(GB_a.T)
        GB_b_sT = sc.sparse.csr_array(GB_b.T)
        GB_E_sT = sc.sparse.csr_array(GB_E.T)

        # Make difference matrix, P
        P_a = sc.sparse.csr_array(dif_mat(c_a,dif_orders[0]).T @ dif_mat(c_a,dif_orders[0]))
        P_b = sc.sparse.csr_array(dif_mat(c_b,dif_orders[1]).T @ dif_mat(c_b,dif_orders[1]))
        P_E = sc.sparse.csr_array(dif_mat(c_E,dif_orders[2]).T @ dif_mat(c_E,dif_orders[2]))

        P_s = smoothness[0] * sc.sparse.kron(P_a,np.identity(c_b * c_E)) \
          + smoothness[1] * sc.sparse.kron(sc.sparse.kron(np.identity(c_a),P_b),np.identity(c_E)) \
          + smoothness[2] * sc.sparse.kron(np.identity(c_a * c_b),P_E)

        del P_a,P_b,P_E,GB_a,GB_b,GB_E

        # In[67]:


        '''
        This block performs the fitting.
        It can be interupted at any point.
        The value of the coefficients in the spline basis will be saved to "theta"
        '''
        if perform_likelihood_test:
            test_sample = np.zeros([test_sample_size,2,len(energies)])
            for index, energy in enumerate(energies):
                samp = Dat[energy].sample(n = test_sample_size)
                test_sample[:,0,index] = ga(samp.gammaA)
                test_sample[:,1,index] = gb(samp.gammaB)
        theta = theta_0
        print(f'        S.unique: {np.unique(S)}')
        # this rescales the coefficients such fitted spline evaluates to log(density)
        coeff_shift = np.einsum('ijk,ai,bj,ck',np.log(S), B_a_pinv, B_b_pinv, B_E_pinv,
                                optimize=True)
        i = 0
        while True:
            i += 1
            print(f'Running iteration {i}...')
            # thetas[:,i] = theta
            _ = np.moveaxis((B_a @ theta.reshape(c_a,c_b*c_E)).reshape(n_a,c_b,c_E),0,-1)
            _ = np.moveaxis((B_b @ _.reshape(c_b,c_E*n_a)).reshape(n_b,c_E,n_a),0,-1)
            Eta = np.moveaxis((B_E @ _.reshape(c_E,n_a*n_b)).reshape(n_E,n_a,n_b),0,-1)

            W = np.exp(Eta)
            # W = np.exp(Eta) * S

            # Z = Eta + np.exp(-Eta)*Y - np.ones_like(Eta)
            ## Make the updated (B.T * W_delta * z)
            WZ = (Eta - 1) * W + Y
            # WZ = (Eta + Y - np.exp(Eta)) * W # (rev1)
            # WZ = (Eta * S - 1) * np.exp(Eta) + Y # (rev2)

            _ = np.moveaxis((B_a.T @ WZ.reshape(n_a,n_b*n_E)).reshape(c_a,n_b,n_E),0,-1)
            _ = np.moveaxis((B_b.T @ _.reshape(n_b,n_E*c_a)).reshape(c_b,n_E,c_a),0,-1)
            BWz = np.moveaxis((B_E.T @ _.reshape(n_E,c_a*c_b)).reshape(c_E,c_a,c_b),0,-1).reshape(-1)
            W_s = sc.sparse.csc_array(W.reshape((n_a,n_b*n_E)))

            GW  = (GB_a_sT @ W_s).tocoo()
            H1_s = sc.sparse.csc_array((GW.data,(GW.coords[1]//n_E, c_a**2 * (GW.coords[1] % n_E) + GW.coords[0])), shape=(n_b,n_E*c_a**2))

            GH1 = (GB_b_sT @ H1_s).tocoo()
            H2_s = sc.sparse.csc_array((GH1.data,(GH1.coords[1]//c_a**2, c_b**2 * (GH1.coords[1] % c_a**2) + GH1.coords[0])), shape=(n_E,c_a**2 * c_b**2))

            GH2 = (GB_E_sT @ H2_s).tocoo()
            H3_s = sc.sparse.coo_array((GH2.data,(GH2.coords[1]//c_b**2, c_E**2 * (GH2.coords[1] % c_b**2) + GH2.coords[0])), shape=(c_a**2,c_b**2 * c_E**2))

            # Reshape into (B.T * W * B) matrix form
            ll,m = H3_s.coords
            p = (ll//c_a) % c_a
            pp = ll % c_a
            q = (m//(c_b*c_E**2)) % c_b
            qp = (m//c_E**2) % c_b
            r = (m//c_E) % c_E
            rp = m % c_E
            BWB_s = sc.sparse.csr_array((H3_s.data,(p*c_b*c_E + q*c_E + r,pp*c_b*c_E + qp*c_E + rp)),shape=(c_a*c_b*c_E,c_a*c_b*c_E))
            print('    Running least squares...')
            print('Pred (slice):\n', np.exp(Eta[...,40]))
            print('Data (slice):\n', Y[...,40])
            # res = sc.sparse.linalg.lsqr(BWB_s+P_s,BWz,atol=0,btol=0,conlim=conlim,iter_lim=int(1e6))
            assert np.all(sc.sparse.linalg.eigsh(BWB_s+P_s)[0] > 0)
            res = sc.sparse.linalg.cg(BWB_s+P_s,BWz,theta,rtol=1e-8,maxiter=100000)
            delta = res[0] - theta
            theta = res[0]
            # print(f'        Least squares R^2: {res[3]:5.3e}')
            print(f'        Convergence / Niterations: {res[1]:}')
            print(f'        Min / max theta: {theta.min()}, {theta.max()}')
            Bspl = maths.BSpline3D(sc.interpolate.NdBSpline(knots,
                                                            theta.reshape((c_a,c_b,c_E)) - coeff_shift,
                                                            deg))
            if perform_likelihood_test:
                print('    Performing likelihood test...')
                lls = [likelihood_test(test_sample[:,0,i],test_sample[:,1,i],log_ens[i],Bspl) for i in range(n_E)]
                print(f'        Log-likelihood: {np.sum(lls):5.3e}')

            if res[1] == 0:
                print('        Saving coefficients')
                outf = files("acem") / "resources" / "theta" / f"{particle}.npz"
                with as_file(outf) as fpath:
                    # Shift to convert into a density with coeff_shift
                    # this notation is to match the NdBSpline
                    np.savez(fpath,
                             t0=knots[0],
                             t1=knots[1],
                             t2=knots[2],
                             c=theta.reshape((c_a,c_b,c_E)) - coeff_shift,
                             k=deg)

            if np.max(np.abs(delta)) < 1e-6:
                break
