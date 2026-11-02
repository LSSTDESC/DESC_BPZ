"""Host-side preprocessing shared by all batched backends.

Everything here runs in float64 on the host and only the (small) per-galaxy
and per-template arrays are cast to the working precision afterwards.

The chi^2 of ``bpz_tools_py3.p_c_z_t`` is

    chi2 = foo - fot**2 / ftt
    foo  = sum_f where(f/ef < 1e-4, 0, (f/ef)**2)
    fot  = sum_f where(ef > 1, 0, f * m / ef**2)
    ftt  = sum_f where(ef > 1, 0, m**2 / ef**2)

It is invariant under a per-galaxy rescaling of the errors and a global
rescaling of the templates, so we work with whitened, dimensionless inputs

    y[g, f] = where(ef > 1, 0, f / ef)          (signal to noise)
    s[g, f] = where(ef > 1, 0, 1 / ef) / max_f(...)
    x[g, z, t, f] = m[z, t, f] / max(m) * s[g, f]

so that every intermediate stays well inside the float32 range (the
``1e108`` / ``1e28`` "unobserved" errors used by bpz.py and rail would
otherwise overflow ``ef**2`` in float32).  With ``a = fot / ftt`` we then
evaluate the cancellation-free residual form

    chi2 = sum_f (y - a x)**2 + c,      c = foo - sum_f y**2

(``c`` is exactly zero unless the two masks above disagree).
"""

import numpy as np

# Same constants as bpz_tools_py3 (eps = 1e-300, eeps = log(eps))
EPS = 1e-300
EEPS = float(np.log(EPS))  # python float: must not promote float32 arrays
CHI2_CLIP = -2.0 * EEPS


def resolve_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"dtype must be float32 or float64, not {dtype}")
    return dtype


def prepare_galaxies(flux, flux_err):
    """Whitened per-galaxy inputs (float64).

    Returns y, s, c with shapes (ng, nf), (ng, nf), (ng,).
    """
    f = np.asarray(flux, dtype=np.float64)
    ef = np.asarray(flux_err, dtype=np.float64)
    if f.ndim != 2 or f.shape != ef.shape:
        raise ValueError("flux and flux_err must both have shape (ng, nf)")

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = f / ef
        foo_terms = np.where(snr < 1e-4, 0.0, snr**2)
        nonobs = ef > 1.0
        y = np.where(nonobs, 0.0, snr)
        inv_ef = np.where(nonobs, 0.0, 1.0 / ef)
    smax = inv_ef.max(axis=1, keepdims=True)
    s = inv_ef / np.where(smax > 0, smax, 1.0)
    c = (foo_terms - y**2).sum(axis=1)
    return y, s, c


def prepare_templates(f_mod):
    """Templates (nz, nt, nf) normalised to a maximum of 1 (float64)."""
    m = np.asarray(f_mod, dtype=np.float64)
    if m.ndim != 3:
        raise ValueError("f_mod must have shape (nz, nt, nf)")
    mmax = np.abs(m).max()
    return m / (mmax if mmax > 0 else 1.0)


def convolution_offsets(kernel):
    """Index offsets so that np.convolve(p, kernel, 'same') equals
    sum_k kernel[k] * p[i + offset[k]] (zero outside the grid)."""
    nk = len(kernel)
    return (nk - 1) // 2 - np.arange(nk)


def default_chunk_size(nz, nt, nf, dtype, budget_bytes):
    """Number of galaxies per chunk so that the (B, nz, nt[, nf]) work
    arrays fit in ``budget_bytes``."""
    per_gal = nz * nt * (nf + 12) * np.dtype(dtype).itemsize
    return int(max(1, min(1 << 16, budget_bytes // per_gal)))
