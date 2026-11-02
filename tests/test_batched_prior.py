import numpy as np
import pytest

from desc_bpz.batched._prior import PriorParams, prior_from_params, prior_params_hdfn
from desc_bpz.prior_from_dict import prior_function


@pytest.mark.parametrize("nt_array", [[1, 2, 5], [8], [2, 6]])
def test_vectorised_prior_matches_prior_function(nt_array):
    z = np.linspace(0.0, 3.0, 301)
    params = prior_params_hdfn(nt_array=nt_array)
    if len(nt_array) == 1:
        params.update(fo_arr=np.array([1.0]), kt_arr=np.array([0.3]), a_arr=[2.0], zo_arr=[0.4], km_arr=[0.1])
    elif len(nt_array) == 2:
        params.update(fo_arr=np.array([0.4]), kt_arr=np.array([0.3]), a_arr=[2.0, 1.5], zo_arr=[0.4, 0.2], km_arr=[0.1, 0.05])
    pp = PriorParams(params, z)
    mags = np.array([15.0, 20.0, 22.37, 24.9, 31.0, 40.0, 99.0])
    batch = prior_from_params(np, pp.zt_at_a, pp.a, pp.zo, pp.km, pp.k_t, pp.fo_t, pp.n_most, pp.n_last, pp.dm(mags))
    for i, m in enumerate(mags):
        np.testing.assert_allclose(batch[i], prior_function(z, m, params, pp.nt), rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("mag_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("m0", [20.0, np.float64(20.0), np.float32(20.0)])
def test_prior_dm_follows_scalar_promotion(mag_dtype, m0):
    """dm = m - m0 is computed with the per-galaxy loop's promotion rules
    (float32 for float32 magnitudes and a python float m0, as in rail)."""
    params = dict(prior_params_hdfn(), mo=m0)
    pp = PriorParams(params, np.linspace(0, 3, 31))
    mags = np.array([18.0, 20.0, 21.123456, 24.987654, 31.99, 32.5, 99.0], dtype=mag_dtype)
    expected = []
    for m in mags:  # the scalar code of prior_from_dict.prior_function
        if m > 32.0:
            m = 32.0
        if m < m0:
            m = m0
        expected.append(float(m - m0))
    np.testing.assert_array_equal(pp.dm(mags), expected)
