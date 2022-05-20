import numpy as np
from desc_bpz.bpz_tools_py3 import prior_with_dict, prior

def test_old_vs_new_prior():
    zgrid = np.arange(0, 10.01, 0.1)
    mags = np.array([20.5, 21., 22.])
    # copy params needed for hdfn_gen prior
    paramdict = dict(fo_arr=[0.35, 0.5], kt_arr=[0.45, 0.147],
                     alpha_arr=[2.465, 1.806, 0.906],
                     zo_arr=[0.431, 0.390, 0.0626],
                     km_arr=[0.0913, 0.0636, 0.123],
                     m0=20.0,
                     numpertype=[1, 2, 3])

    for mag in mags:
        oldprior = prior(zgrid, mag, 'hdfn_gen')
        newprior = prior_with_dict(zgrid, mag, paramdict)
        assert (oldprior == newprior).all()
