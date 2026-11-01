"""Acceptance gates of the GPU backends against the per-galaxy loops."""

import numpy as np
import pytest
from conftest import assert_matches_reference

from desc_bpz.batched import BACKENDS, estimate_batch, get_backend

GPU_BACKENDS = [b for b in BACKENDS if b != "numpy"]


def _require(backend):
    try:
        available = get_backend(backend).is_available()
    except ImportError as err:
        pytest.skip(f"{backend} backend not installed: {err}")
    if not available:
        pytest.skip(f"{backend} backend has no usable device")


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("case", ["bpz-test1000", "rail-test1000", "bpz-synthetic", "rail-synthetic"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_backend_matches_loop(test1000, backend, case, dtype):
    _require(backend)
    cat, mode, kw, ref = test1000["cases"][case]
    res = estimate_batch(cat["flux"], cat["flux_err"], test1000["f_mod"], test1000["z"], mode=mode, backend=backend, dtype=dtype, **kw)
    assert_matches_reference(res, ref, dtype, mode)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_backend_flat_prior_and_chunks(test1000, backend):
    _require(backend)
    cat, z, f_mod = test1000["cat"], test1000["z"], test1000["f_mod"]
    args = (cat["flux"], cat["flux_err"], f_mod, z)
    ref = estimate_batch(*args, mode="rail", p_min=0.005, backend="numpy")
    for chunk in (100, None):
        res = estimate_batch(*args, mode="rail", p_min=0.005, backend=backend, chunk_size=chunk)
        np.testing.assert_allclose(res["pdf"], ref["pdf"], rtol=1e-9, atol=1e-12)
        np.testing.assert_array_equal(res["iz_b"], ref["iz_b"])
