"""JAX backend: the shared functional pipeline, jitted (GPU if available).

float64 runs inside a scoped ``jax.experimental.enable_x64()`` context, so
the process-wide x64 flag of other JAX code (e.g. elsewhere in a RAIL
pipeline) is left untouched.

Unless the user configured it, JAX is told not to pre-allocate 75% of the
GPU so that it can share the device with CuPy (the 'cuda' backend) and
other libraries; this only has an effect if JAX was not initialised yet.
"""

import contextlib
import os
import warnings
from functools import partial

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from ._pipeline import chi2_grid, posterior_stats  # noqa: E402
from ._prep import default_chunk_size  # noqa: E402
from ._prior import prior_from_params  # noqa: E402

name = "jax"


def is_available():
    return len(jax.devices()) > 0


def _x64(dtype):
    if dtype != np.float64:
        return contextlib.nullcontext()
    from jax.experimental import enable_x64

    @contextlib.contextmanager
    def ctx():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with enable_x64():
                yield

    return ctx()


@partial(jax.jit, static_argnames=("cfg_items", "prior_kind", "n_most", "n_last", "has_kernel"))
def _chunk(y, s, c, prior_arg, m, z, kernel, pconst, *, cfg_items, prior_kind, n_most, n_last, has_kernel):
    cfg = dict(cfg_items)
    chi2 = chi2_grid(jnp, y, s, c, m, cfg["chi2_form"])
    if prior_kind == "table":
        table = pconst[0]
        prior = table[prior_arg]
    elif prior_kind == "params":
        prior = prior_from_params(jnp, *pconst, n_most, n_last, prior_arg)
    else:
        prior = None
    return posterior_stats(jnp, chi2, prior, z, cfg, kernel if has_kernel else None)


def _budget_bytes(device):
    try:
        stats = device.memory_stats()
        free = stats["bytes_limit"] - stats["bytes_in_use"]
        return int(0.3 * free)
    except Exception:  # CPU devices have no memory stats
        return 512 << 20


def _pad(a, n):
    if len(a) == n:
        return a
    pad = np.zeros((n - len(a),) + a.shape[1:], dtype=a.dtype)
    return np.concatenate([a, pad])


def run(inputs, cfg):
    dtype = inputs["dtype"]
    ng = inputs["y"].shape[0]
    nz, nt, nf = inputs["m"].shape
    device = jax.devices()[0]
    with _x64(dtype):
        y, s, c = (inputs[k].astype(dtype) for k in ("y", "s", "c"))
        m = jax.device_put(inputs["m"].astype(dtype), device)
        z = jax.device_put(inputs["z"].astype(dtype), device)
        has_kernel = inputs["kernel"] is not None
        kernel = inputs["kernel"].astype(dtype) if has_kernel else None

        kind = inputs["prior_kind"]
        n_most = n_last = 0
        if kind == "table":
            pconst = (jax.device_put(inputs["prior_table"].astype(dtype), device),)
            per_gal = inputs["prior_index"].astype(np.int32)
        elif kind == "params":
            pp = inputs["prior_params"]
            pconst = tuple(jax.device_put(np.asarray(v, dtype=dtype), device)
                           for v in (pp.zt_at_a, pp.a, pp.zo, pp.km, pp.k_t, pp.fo_t))
            n_most, n_last = pp.n_most, pp.n_last
            per_gal = inputs["dm"].astype(dtype)
        else:
            pconst = ()
            per_gal = np.zeros(ng, dtype=np.int32)

        chunk = inputs["chunk_size"] or default_chunk_size(nz, nt, nf, dtype, _budget_bytes(device))
        # Fixed chunk shape (a power of two for small inputs) => one compilation
        chunk = min(chunk, 1 << int(np.ceil(np.log2(max(ng, 1)))))
        cfg_items = tuple(sorted(cfg.items()))
        static = dict(cfg_items=cfg_items, prior_kind=kind, n_most=n_most, n_last=n_last, has_kernel=has_kernel)

        out = None
        pending = None

        def collect(item):
            nonlocal out
            sl, res = item
            res = jax.device_get(res)
            if out is None:
                out = {k: np.empty((ng,) + v.shape[1:], v.dtype) for k, v in res.items()}
            n = sl.stop - sl.start
            for k, v in res.items():
                out[k][sl] = v[:n]

        for start in range(0, ng, chunk):
            sl = slice(start, min(start + chunk, ng))
            args = [_pad(a[sl], chunk) for a in (y, s, c, per_gal)]
            res = _chunk(*args, m, z, kernel, pconst, **static)  # asynchronous dispatch
            if pending is not None:
                collect(pending)  # overlaps the transfer with the next chunk
            pending = (sl, res)
        collect(pending)
    return out
