import os

import numpy as np
import pytest

TESTDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.abspath(os.path.join(TESTDIR, "../src/desc_bpz/data_files"))

# test1000.pars
ZMIN, ZMAX, DZ = 0.005, 3.505, 0.01
ODDS, P_MIN, MIN_RMS = 0.68, 0.01, 0.05


@pytest.fixture(scope="session")
def test1000():
    """test1000 inputs exactly as bpz.py builds them, plus loop references."""
    import desc_bpz.paths
    from desc_bpz.batched import prior_params_hdfn
    from desc_bpz.batched.reference import loop_bpz, loop_rail
    from desc_bpz.batched.testing import (
        bpz_prior_table,
        bpz_zgrid,
        gaussian_kernel,
        load_catalog,
        load_templates,
        resample_catalog,
    )

    desc_bpz.paths.set_data_dir(DATADIR)
    cat = load_catalog(os.path.join(TESTDIR, "test1000.h5"), os.path.join(TESTDIR, "CSDC2114_test.columns"))
    z = bpz_zgrid(ZMIN, ZMAX, DZ)
    f_mod = load_templates("CWWSB4.list", cat["filters"], z)
    kernel = gaussian_kernel(0.03, DZ)
    syn = resample_catalog(cat, 10000, seed=42)
    d = dict(cat=cat, syn=syn, z=z, f_mod=f_mod, kernel=kernel, prior_params=prior_params_hdfn())

    def bpz_case(c):
        table, index = bpz_prior_table(z, c["m_0"], f_mod.shape[1])
        kw = dict(prior_table=table, prior_index=index, p_min=P_MIN, kernel=kernel, odds=ODDS, min_rms=MIN_RMS)
        return kw, loop_bpz(c["flux"], c["flux_err"], f_mod, z, **kw)

    def rail_case(c):
        kw = dict(mag0=c["m_0"], prior_params=d["prior_params"], p_min=0.005, kernel=None)
        return kw, loop_rail(c["flux"], c["flux_err"], f_mod, z, **kw)

    d["cases"] = {
        "bpz-test1000": (cat, "bpz") + bpz_case(cat),
        "rail-test1000": (cat, "rail") + rail_case(cat),
        "bpz-synthetic": (syn, "bpz") + bpz_case(syn),
        "rail-synthetic": (syn, "rail") + rail_case(syn),
    }
    return d


def assert_matches_reference(res, ref, dtype, mode):
    """Acceptance gates for a batched result against the per-galaxy loop.

    The rail loop does not compute ODDS / the confidence interval.
    """
    if np.dtype(dtype) == np.float64:
        np.testing.assert_allclose(res["pdf"], ref["pdf"], rtol=1e-9, atol=1e-12)
        keys = ("iz_b", "t_b", "iz_ml", "it_ml") + (("iz1", "iz2", "t_b_valid") if mode == "bpz" else ())
        for k in keys:
            np.testing.assert_array_equal(res[k], ref[k], err_msg=k)
        np.testing.assert_allclose(res["chi2_min"], ref["chi2_min"], rtol=1e-9, atol=1e-9)
        if mode == "bpz":
            np.testing.assert_allclose(res["odds"], ref["odds"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(res["todds"], ref["todds"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(res["zmean"], ref["zmean"], rtol=1e-9, atol=1e-12)
        return
    # float32: the mode must agree for >= 99.5% of the galaxies and every
    # disagreement must be within 2 grid cells or a near tie in the
    # reference PDF (the other peak >= 0.99 of the maximum).
    diff = res["iz_b"] != ref["iz_b"]
    assert diff.mean() <= 0.005, f"{diff.sum()} zmode mismatches"
    rows = np.flatnonzero(diff)
    close = np.abs(res["iz_b"][rows] - ref["iz_b"][rows]) <= 2
    peak = ref["pdf"][rows].max(axis=1)
    tie = ref["pdf"][rows, res["iz_b"][rows]] >= 0.99 * peak
    assert np.all(close | tie), f"zmode mismatches beyond tolerance: {rows[~(close | tie)]}"
    peak = ref["pdf"].max(axis=1, keepdims=True)
    ok = peak[:, 0] > 0
    rel = np.abs(res["pdf"][ok] - ref["pdf"][ok]).max(axis=1) / peak[ok, 0]
    assert np.median(rel) <= 1e-5, f"median over galaxies of max|dPDF|/peak = {np.median(rel)}"
