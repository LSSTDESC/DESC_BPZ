"""Batched (vectorised / GPU) BPZ estimation.

``estimate_batch`` computes, for many galaxies at once, what the per-galaxy
loops in ``scripts/bpz.py`` (``mode='bpz'``) and rail_bpz's
``BPZliteEstimator`` (``mode='rail'``) compute with ``p_c_z_t`` and the
prior.  Backends:

* ``'numpy'``: vectorised CPU implementation (reference),
* ``'jax'``:   JAX/XLA (GPU if available),
* ``'cuda'``:  CuPy with hand written CUDA C kernels.

Example
-------
>>> res = estimate_batch(flux, flux_err, f_mod, zgrid, mag0=m,
...                      prior_params=modeldict, p_min=0.005,
...                      backend='jax', dtype='float32')
>>> res['pdf'].shape, res['zmode'].shape
"""

import importlib

import numpy as np

from ._prep import prepare_galaxies, prepare_templates, resolve_dtype
from ._prior import PriorParams, build_prior_table, prior_params_hdfn

__all__ = [
    "estimate_batch",
    "available_backends",
    "get_backend",
    "build_prior_table",
    "prior_params_hdfn",
    "BACKENDS",
]

BACKENDS = ("numpy", "jax", "cuda")
_MODULES = {
    "numpy": ".numpy_backend",
    "jax": ".jax_backend",
    "cuda": ".cuda_backend",
}


def get_backend(name):
    """Import and return a backend module (raises ValueError/ImportError)."""
    if name not in _MODULES:
        raise ValueError(f"unknown backend {name!r}, choose from {BACKENDS}")
    return importlib.import_module(_MODULES[name], __name__)


def available_backends():
    """Backends whose dependencies import and that can see a device."""
    found = []
    for name in BACKENDS:
        try:
            if get_backend(name).is_available():
                found.append(name)
        except Exception:  # missing optional dependency or no GPU
            pass
    return found


def estimate_batch(
    flux,
    flux_err,
    f_mod,
    zgrid,
    *,
    mag0=None,
    prior_params=None,
    prior_table=None,
    prior_index=None,
    mode="rail",
    p_min=0.005,
    kernel=None,
    odds=0.68,
    oi=None,
    min_rms=0.05,
    z_thr=0.0,
    zmax=None,
    backend="numpy",
    dtype="float64",
    chunk_size=None,
    chi2_form="residual",
):
    """BPZ posteriors and summary statistics for a batch of galaxies.

    Parameters
    ----------
    flux, flux_err : (ng, nf) arrays of (pseudo-)fluxes, as passed to p_c_z_t
    f_mod : (nz, nt, nf) template fluxes
    zgrid : (nz,) redshift grid
    mag0 : (ng,) reference magnitudes, needed with ``prior_params``
    prior_params : dict in the ``prior_from_dict`` format; prior evaluated
        per galaxy at the exact magnitude (rail_bpz behaviour)
    prior_table, prior_index : (K, nz, nt) table and (ng,) index into it
        (see ``build_prior_table``; bpz.py behaviour).  With neither prior
        given the prior is flat (ones).
    mode : 'rail' or 'bpz', selects the PDF trimming / normalisation rule
    p_min : fraction of the PDF peak below which the PDF is set to zero
    kernel : optional 1-d kernel, the PDF is ``np.convolve(p, kernel, 'same')``
    odds, oi, min_rms, z_thr, zmax : ODDS / confidence-interval parameters
        of bpz.py (``oi`` defaults to ``inv_gauss_int(odds)``, ``zmax`` to
        the last grid point; it is only used when ``z_thr > 0``)
    backend : 'numpy', 'jax' or 'cuda'
    dtype : 'float64' or 'float32' working precision
    chunk_size : galaxies per device chunk (None: automatic)
    chi2_form : 'residual' (default) or 'expanded' (textbook
        foo - fot^2/ftt, kept for numerical comparisons only)

    Returns
    -------
    dict of numpy arrays: pdf (ng, nz) in the working precision (not
    copied to float64, it can be large); float64 / int64 / bool: zmode, iz_b, t_b (0-based best template at the mode),
    t_b_valid (False where bpz.py would report T_B = -1), todds, zmean,
    iz_ml, it_ml, chi2_min, odds, z_b_min, z_b_max (confidence interval
    before bpz.py's PHOTO_ERRORS widening).
    """
    dtype = resolve_dtype(dtype)
    if mode not in ("rail", "bpz"):
        raise ValueError("mode must be 'rail' or 'bpz'")
    zgrid = np.asarray(zgrid, dtype=np.float64)
    y, s, c = prepare_galaxies(flux, flux_err)
    m = prepare_templates(f_mod)
    ng = y.shape[0]
    nz, nt, nf = m.shape
    if len(zgrid) != nz or y.shape[1] != nf:
        raise ValueError("inconsistent shapes between flux, f_mod and zgrid")

    inputs = dict(y=y, s=s, c=c, m=m, z=zgrid, dtype=dtype, chunk_size=chunk_size)
    inputs["kernel"] = None if kernel is None else np.asarray(kernel, dtype=np.float64)
    if prior_params is not None:
        if mag0 is None:
            raise ValueError("prior_params needs mag0")
        pp = PriorParams(prior_params, zgrid)
        if pp.nt != nt:
            raise ValueError(f"prior has {pp.nt} templates, f_mod has {nt}")
        inputs.update(prior_kind="params", prior_params=pp, dm=pp.dm(mag0))
    elif prior_table is not None:
        table = np.asarray(prior_table, dtype=np.float64)
        if table.ndim == 2:
            table = table[None]
        index = np.zeros(ng, dtype=np.int64) if prior_index is None else np.asarray(prior_index, dtype=np.int64)
        if table.shape[1:] != (nz, nt) or len(index) != ng:
            raise ValueError("prior_table must be (K, nz, nt) and prior_index (ng,)")
        inputs.update(prior_kind="table", prior_table=table, prior_index=index)
    else:
        inputs.update(prior_kind="none")

    if oi is None:
        from desc_bpz.useful_py3 import inv_gauss_int

        oi = inv_gauss_int(float(odds))
    cfg = dict(
        mode=mode,
        p_min=float(p_min),
        oi=float(oi),
        min_rms=float(min_rms),
        odds_ci=float(odds),
        z_thr=float(z_thr),
        zmax=float(zgrid[-1] if zmax is None else zmax),
        chi2_form=chi2_form,
    )

    if ng == 0:
        raw = dict(pdf=np.zeros((0, nz)))
        raw.update({k: np.zeros(0, int) for k in ("iz_b", "it_b", "iz_ml", "it_ml", "iz1", "iz2")})
        raw.update({k: np.zeros(0) for k in ("todds", "zmean", "chi2_min", "odds")})
        raw["t_b_valid"] = np.zeros(0, bool)
    else:
        raw = get_backend(backend).run(inputs, cfg)
    return _finalise(raw, zgrid)


def _finalise(raw, zgrid):
    out = {k: np.asarray(v) for k, v in raw.items()}
    for k in ("iz_b", "it_b", "iz_ml", "it_ml", "iz1", "iz2"):
        out[k] = out[k].astype(np.int64, copy=False)
    for k in ("todds", "zmean", "chi2_min", "odds"):
        out[k] = out[k].astype(np.float64, copy=False)
    out["zmode"] = zgrid[out["iz_b"]]
    out["z_b_min"] = zgrid[out["iz1"]]
    out["z_b_max"] = zgrid[out["iz2"]]
    out["t_b"] = out.pop("it_b")
    out["t_b_valid"] = out["t_b_valid"].astype(bool)
    return out
