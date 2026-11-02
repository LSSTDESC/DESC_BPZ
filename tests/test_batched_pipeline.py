import numpy as np
import pytest

from desc_bpz.batched._pipeline import convolve_same


@pytest.mark.parametrize("nk", [1, 4, 7, 19, 20])
def test_convolve_same_matches_numpy(nk):
    rng = np.random.default_rng(nk)
    p = rng.random((5, 50))
    kernel = rng.random(nk)
    expected = np.array([np.convolve(row, kernel, 1) for row in p])
    np.testing.assert_allclose(convolve_same(np, p, kernel), expected, rtol=1e-13)


def test_count_less_is_searchsorted():
    rng = np.random.default_rng(1)
    cp = np.cumsum(rng.random(100) * (rng.random(100) > 0.3))
    cp /= cp[-1]
    for q in [0.0, 0.16, 0.5, 0.84, 1.0, cp[10], cp[57]]:
        assert (cp < q).sum() == np.searchsorted(cp, q)
