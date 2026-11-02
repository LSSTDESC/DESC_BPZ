"""Priors for the batched backends.

Two ways of supplying the prior p(z, t | m):

* a *table* of shape (K, nz, nt) plus a per-galaxy index into it.  bpz.py
  quantises the magnitude to ``'%.1f'`` and caches one prior per quantised
  value, so a table built with the existing ``prior()`` function reproduces
  the per-galaxy loop exactly (for any prior module, cluster spikes or
  template interpolation).
* *parameters* in the ``prior_from_dict`` format (what rail_bpz uses), which
  are evaluated per galaxy on the device without magnitude quantisation.
"""

import numpy as np


def prior_params_hdfn(m0=20.0, nt_array=(1, 2, 5)):
    """The hdfn_gen prior of Benitez (2000) in the ``prior_from_dict`` format."""
    return dict(
        fo_arr=np.array([0.35, 0.5]),
        kt_arr=np.array([0.450, 0.147]),
        a_arr=np.array([2.465, 1.806, 0.906]),
        zo_arr=np.array([0.431, 0.390, 0.0626]),
        km_arr=np.array([0.0913, 0.0636, 0.123]),
        mo=m0,
        nt_array=list(nt_array),
    )


class PriorParams:
    """Per-template constants of ``prior_from_dict.prior_function``.

    The per-galaxy part only depends on ``dm = clip(m, m0, 32) - m0``.
    """

    def __init__(self, paramdict, zgrid):
        numpertype = np.asarray(paramdict["nt_array"], dtype=int)
        self.nt = int(numpertype.sum())
        self.m0_raw = paramdict["mo"]
        self.m0 = float(self.m0_raw)
        self.a = np.repeat(np.asarray(paramdict["a_arr"], float), numpertype)
        self.zo = np.repeat(np.asarray(paramdict["zo_arr"], float), numpertype)
        self.km = np.repeat(np.asarray(paramdict["km_arr"], float), numpertype)
        fo_arr = np.asarray(paramdict["fo_arr"], float)
        kt_arr = np.asarray(paramdict["kt_arr"], float)
        if len(numpertype) == 1:
            self.n_most = self.nt
            self.n_last = 0
            k_t = np.repeat(kt_arr, numpertype)
            fo_t = np.repeat(fo_arr / numpertype[0], numpertype)
        else:
            self.n_most = int(numpertype[:-1].sum())
            self.n_last = int(numpertype[-1])
            k_t = np.repeat(kt_arr, numpertype[:-1])
            fo_t = np.repeat(fo_arr / numpertype[:-1], numpertype[:-1])
        # Pad to nt; the trailing (last broad type) entries are unused.
        self.k_t = np.zeros(self.nt)
        self.fo_t = np.zeros(self.nt)
        self.k_t[: self.n_most] = k_t
        self.fo_t[: self.n_most] = fo_t
        z = np.asarray(zgrid, float)
        self.nz = len(z)
        self.zt_at_a = np.power.outer(z, self.a)  # (nz, nt)

    def dm(self, mag):
        """``m - m0`` exactly as prior_function computes it per galaxy.

        There ``if m>32: m=32; if m<m0: m=m0`` replaces m by python floats
        when clipping, and otherwise ``m - m0`` follows NumPy's promotion
        rules, i.e. it is evaluated in float32 for float32 magnitudes and a
        python float m0 (as in rail_bpz).  NaN passes through.
        """
        m = np.asarray(mag)
        if not np.issubdtype(m.dtype, np.floating):
            m = m.astype(float)
        dm = np.asarray(m - self.m0_raw, dtype=float)
        dm = np.where(m < self.m0, self.m0 - self.m0, dm)
        return np.where(m > 32.0, 32.0 - self.m0, dm)


def prior_from_params(xp, zt_at_a, a, zo, km, k_t, fo_t, n_most, n_last, dm):
    """Vectorised ``prior_function`` for a chunk of galaxies.

    ``xp`` is numpy or jax.numpy; all arrays are already in the working
    dtype.  Returns an array of shape (B, nz, nt).
    """
    nz = zt_at_a.shape[0]
    dm = dm[:, None]
    zmt = xp.clip(zo + km * dm, 0.01, 15.0)
    zmt_at_a = zmt ** a  # (B, nt)
    f_most = fo_t * xp.exp(-k_t * dm)  # (B, nt); only [:n_most] meaningful
    if n_last:
        tmask = xp.arange(zt_at_a.shape[1]) < n_most
        f_most = xp.where(tmask, f_most, 0.0)
        f_rest = (1.0 - f_most.sum(axis=1, keepdims=True)) / float(n_last)
        f_t = xp.where(tmask, f_most, f_rest)
    else:
        f_t = f_most
    zt = zt_at_a[None, :, :]
    p = zt * xp.exp(-1.0 * xp.clip(zt / zmt_at_a[:, None, :], 0.0, 700.0))
    norm = p.sum(axis=1, keepdims=True)
    q = p / norm
    q = xp.where(q < 1e-2 / float(nz), 0.0, q)
    norm = q.sum(axis=1, keepdims=True)
    return q / norm * f_t[:, None, :]


def build_prior_table(mags, prior_fn, fmt="%.1f"):
    """Tabulate ``prior_fn(m)`` on the distinct ``fmt``-quantised magnitudes.

    Returns (table (K, nz, nt), index (ng,)) with ``table[index[g]]`` equal
    to ``prior_fn(mags[g])`` whenever ``prior_fn`` itself only depends on the
    quantised magnitude (true for bpz_tools_py3.prior).
    """
    keys = np.char.mod(fmt, np.asarray(mags, dtype=float))
    ukeys, first, index = np.unique(keys, return_index=True, return_inverse=True)
    mags = np.asarray(mags, dtype=float)
    table = np.stack([np.asarray(prior_fn(mags[i]), dtype=float) for i in first])
    return table, index.reshape(-1)
