"""Per-galaxy reference loops using the original BPZ functions.

``loop_rail`` is a copy of rail_bpz's ``BPZliteEstimator._estimate_pdf``
loop and ``loop_bpz`` of the N_PEAKS=1 branch of the loop in
``scripts/bpz.py``.  They return the same dictionary as ``estimate_batch``
and are the ground truth for the batched backends' tests and benchmarks.
"""

import numpy as np

from desc_bpz.bpz_tools_py3 import interval, odds as bpz_odds, p_c_z_t
from desc_bpz.prior_from_dict import prior_function
from desc_bpz.useful_py3 import inv_gauss_int


def _empty(ng, nz):
    out = dict(pdf=np.zeros((ng, nz)))
    for k in ("iz_b", "t_b", "iz_ml", "it_ml", "iz1", "iz2"):
        out[k] = np.zeros(ng, dtype=np.int64)
    for k in ("todds", "zmean", "chi2_min", "odds", "z_b_min", "z_b_max", "zmode"):
        out[k] = np.zeros(ng)
    out["t_b_valid"] = np.ones(ng, dtype=bool)
    return out


def loop_rail(flux, flux_err, f_mod, zgrid, mag0=None, prior_params=None, p_min=0.005, kernel=None):
    """rail_bpz semantics (no prior if ``prior_params`` is None)."""
    ng, nz = len(flux), len(zgrid)
    nt = f_mod.shape[1]
    out = _empty(ng, nz)
    for i in range(ng):
        pczt = p_c_z_t(flux[i], flux_err[i], f_mod)
        L = pczt.likelihood
        if prior_params is None:
            P = np.ones(L.shape)
        else:
            P = prior_function(zgrid, mag0[i], prior_params, nt)
        post = L * P
        post_z = post.sum(axis=1)
        if kernel is not None:
            post_z = np.convolve(post_z, kernel, 1)
        zpos = np.argmax(post_z)
        p_max = post_z.max()
        post_z[post_z < (p_max * p_min)] = 0
        if not np.isclose(post_z.sum(), 0.0):
            post_z /= post_z.sum()
        tmode = post[zpos, :]
        t_b = np.argmax(tmode)
        tmarg = post.sum(axis=0)
        out["pdf"][i] = post_z
        out["iz_b"][i] = zpos
        out["zmode"][i] = zgrid[zpos]
        out["t_b"][i] = t_b
        out["todds"][i] = tmarg[t_b] / np.sum(tmarg)
        out["zmean"][i] = (zgrid * post_z).sum() / post_z.sum()
        out["iz_ml"][i] = pczt.i_z_ml
        out["it_ml"][i] = pczt.i_t_ml
        out["chi2_min"][i] = pczt.min_chi2
    return out


def loop_bpz(
    flux,
    flux_err,
    f_mod,
    zgrid,
    prior_table=None,
    prior_index=None,
    p_min=0.01,
    kernel=None,
    odds=0.68,
    min_rms=0.05,
    z_thr=0.0,
):
    """bpz.py semantics (N_PEAKS=1, COLOR=no, ONLY_TYPE=no)."""
    ng, nz = len(flux), len(zgrid)
    nt = f_mod.shape[1]
    z = zgrid
    oi = inv_gauss_int(odds)
    out = _empty(ng, nz)
    if prior_table is None:
        prior_table = np.ones((1, nz, nt)) / float(nz * nt)
        prior_index = np.zeros(ng, dtype=int)
    for ig in range(ng):
        likelihood = p_c_z_t(flux[ig], flux_err[ig], f_mod)
        p = likelihood.likelihood
        p_i = prior_table[prior_index[ig]]
        pb = p_i[:nz, :nt] * p[:nz, :nt]
        p_bayes = np.add.reduce(pb[:nz, :nt], -1)
        if kernel is not None:
            p_bayes = np.convolve(p_bayes, kernel, 1)
        pmax = max(p_bayes)
        p_bayes = np.where(np.greater(p_bayes, pmax * float(p_min)), p_bayes, 0.0)
        norm = np.add.reduce(p_bayes)
        p_bayes = p_bayes / norm
        iz_b = np.argmax(p_bayes)
        zb = z[iz_b]
        zo1 = zb - oi * min_rms * (1.0 + zb)
        zo2 = zb + oi * min_rms * (1.0 + zb)
        if z_thr > 0:
            zo1 = float(z_thr)
            zo2 = float(z[-1])
        o = bpz_odds(p_bayes[:nz], z, zo1, zo2)
        it_b = np.argmax(pb[iz_b, :nt])
        z1, z2 = interval(p_bayes[:nz], z, odds)
        tmarg = pb.sum(axis=0)
        out["pdf"][ig] = p_bayes
        out["iz_b"][ig] = iz_b
        out["zmode"][ig] = zb
        out["t_b"][ig] = it_b
        out["t_b_valid"][ig] = not (max(pb[iz_b, :]) < 1e-300)
        out["todds"][ig] = tmarg[it_b] / tmarg.sum()
        out["zmean"][ig] = (z * p_bayes).sum() / p_bayes.sum()
        out["iz_ml"][ig] = likelihood.i_z_ml
        out["it_ml"][ig] = likelihood.i_t_ml
        out["chi2_min"][ig] = likelihood.min_chi2
        out["odds"][ig] = o
        out["z_b_min"][ig] = z1
        out["z_b_max"][ig] = z2
        out["iz1"][ig] = np.searchsorted(z, z1)
        out["iz2"][ig] = np.searchsorted(z, z2)
    return out
