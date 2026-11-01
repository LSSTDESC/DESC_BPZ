"""Vectorised CPU backend (and the reference for the GPU backends)."""

import numpy as np

from ._pipeline import chi2_grid, posterior_stats
from ._prior import prior_from_params
from ._prep import default_chunk_size

name = "numpy"


def is_available():
    return True


def run(inputs, cfg):
    """Run the batched pipeline.  ``inputs`` is built by ``estimate_batch``."""
    dtype = inputs["dtype"]
    y, s, c = (inputs[k].astype(dtype) for k in ("y", "s", "c"))
    m = inputs["m"].astype(dtype)
    z = inputs["z"].astype(dtype)
    kernel = inputs["kernel"]
    if kernel is not None:
        kernel = kernel.astype(dtype)
    ng = y.shape[0]
    nz, nt, nf = m.shape
    chunk = inputs["chunk_size"] or default_chunk_size(nz, nt, nf, dtype, 512 << 20)

    prior_kind = inputs["prior_kind"]
    if prior_kind == "table":
        table = inputs["prior_table"].astype(dtype)
        index = inputs["prior_index"]
    elif prior_kind == "params":
        pp = inputs["prior_params"]
        pconst = [np.asarray(v, dtype=dtype) for v in (pp.zt_at_a, pp.a, pp.zo, pp.km, pp.k_t, pp.fo_t)]
        dm = inputs["dm"].astype(dtype)

    out = None
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        for start in range(0, ng, chunk):
            sl = slice(start, min(start + chunk, ng))
            chi2 = chi2_grid(np, y[sl], s[sl], c[sl], m, cfg["chi2_form"])
            if prior_kind == "table":
                prior = table[index[sl]]
            elif prior_kind == "params":
                prior = prior_from_params(np, *pconst, pp.n_most, pp.n_last, dm[sl])
            else:
                prior = None
            res = posterior_stats(np, chi2, prior, z, cfg, kernel)
            if out is None:
                out = {k: np.empty((ng,) + v.shape[1:], v.dtype) for k, v in res.items()}
            for k, v in res.items():
                out[k][sl] = v
    return out
