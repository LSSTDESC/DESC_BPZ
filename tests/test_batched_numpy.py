import numpy as np
import pytest
from conftest import ODDS, assert_matches_reference

from desc_bpz.batched import estimate_batch


@pytest.mark.parametrize("case", ["bpz-test1000", "rail-test1000", "bpz-synthetic", "rail-synthetic"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_numpy_backend_matches_loop(test1000, case, dtype):
    cat, mode, kw, ref = test1000["cases"][case]
    res = estimate_batch(cat["flux"], cat["flux_err"], test1000["f_mod"], test1000["z"], mode=mode, backend="numpy", dtype=dtype, **kw)
    assert_matches_reference(res, ref, dtype, mode)


def test_chunking_does_not_change_results(test1000):
    cat, mode, kw, _ = test1000["cases"]["bpz-test1000"]
    args = (cat["flux"], cat["flux_err"], test1000["f_mod"], test1000["z"])
    a = estimate_batch(*args, mode=mode, chunk_size=97, **kw)
    b = estimate_batch(*args, mode=mode, chunk_size=4096, **kw)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_flat_prior_and_bpz_uniform_prior_agree(test1000):
    """rail no_prior (ones) and bpz useprior=0 (1/(nz nt)) give the same PDF."""
    cat, z, f_mod = test1000["cat"], test1000["z"], test1000["f_mod"]
    nz, nt = f_mod.shape[:2]
    a = estimate_batch(cat["flux"], cat["flux_err"], f_mod, z, mode="bpz", p_min=0.01)
    b = estimate_batch(cat["flux"], cat["flux_err"], f_mod, z, mode="bpz", p_min=0.01, prior_table=np.ones((nz, nt)) / (nz * nt))
    np.testing.assert_allclose(a["pdf"], b["pdf"], rtol=1e-12)
    np.testing.assert_array_equal(a["iz_b"], b["iz_b"])


def test_bad_arguments(test1000):
    cat, z, f_mod = test1000["cat"], test1000["z"], test1000["f_mod"]
    with pytest.raises(ValueError):
        estimate_batch(cat["flux"], cat["flux_err"], f_mod, z, dtype="float16")
    with pytest.raises(ValueError):
        estimate_batch(cat["flux"], cat["flux_err"], f_mod, z, mode="other")
    with pytest.raises(ValueError):
        estimate_batch(cat["flux"], cat["flux_err"], f_mod, z, backend="opencl")
    with pytest.raises(ValueError):
        estimate_batch(cat["flux"], cat["flux_err"], f_mod, z[:-1])
    with pytest.raises(ValueError):
        estimate_batch(cat["flux"], cat["flux_err"], f_mod, z, prior_params=test1000["prior_params"])
    empty = estimate_batch(cat["flux"][:0], cat["flux_err"][:0], f_mod, z, odds=ODDS)
    assert empty["pdf"].shape == (0, len(z))


@pytest.mark.parametrize("mode", ["bpz", "rail"])
def test_float32_stays_float32(test1000, mode):
    """No float64 constant may silently promote the float32 pipeline."""
    cat, _, kw, _ = test1000["cases"][f"{mode}-test1000"]
    res = estimate_batch(cat["flux"], cat["flux_err"], test1000["f_mod"], test1000["z"], mode=mode, dtype="float32", **kw)
    assert res["pdf"].dtype == np.float32
