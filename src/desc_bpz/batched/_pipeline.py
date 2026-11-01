"""Array-module agnostic BPZ pipeline for a chunk of galaxies.

Written in a purely functional style so the same code runs with
``xp = numpy`` (the CPU reference) and ``xp = jax.numpy`` (traced and
jitted).  Shapes: B galaxies, nz redshifts, nt templates, nf filters.
"""

import math

from ._prep import CHI2_CLIP

LOG_1E_8 = math.log(1e-8)
LOG_1E_300 = math.log(1e-300)


def chi2_grid(xp, y, s, c, m, chi2_form="residual"):
    """chi^2 (B, nz, nt) from whitened inputs, see ``_prep``.

    Loops over the (few) filters so that only (B, nz, nt) temporaries are
    created, never (B, nz, nt, nf).
    """
    if chi2_form not in ("residual", "expanded"):
        raise ValueError(f"unknown chi2_form {chi2_form!r}")
    nf = m.shape[-1]

    def x_(f):
        return m[None, :, :, f] * s[:, f, None, None]

    def y_(f):
        return y[:, f, None, None]

    fot = ftt = 0.0
    for f in range(nf):
        xf = x_(f)
        fot = fot + xf * y_(f)
        ftt = ftt + xf * xf
    pos = ftt > 0
    safe_ftt = xp.where(pos, ftt, 1.0)
    if chi2_form == "expanded":
        foo = ((y * y).sum(axis=-1) + c)[:, None, None]
        return xp.where(pos, foo - fot**2 / safe_ftt, foo)
    a = xp.where(pos, fot / safe_ftt, 0.0)
    chi2 = c[:, None, None]
    for f in range(nf):
        r = y_(f) - a * x_(f)
        chi2 = chi2 + r * r
    return chi2


def convolve_same(xp, p, kernel):
    """``np.convolve(row, kernel, 'same')`` for every row of p (B, nz)."""
    nk = len(kernel)
    nz = p.shape[1]
    if nk > nz:
        raise ValueError("convolution kernel longer than the redshift grid")
    right = (nk - 1) // 2
    left = nk - 1 - right
    pp = xp.pad(p, ((0, 0), (left, right)))
    out = None
    for k in range(nk):
        start = left + right - k
        term = kernel[k] * pp[:, start : start + nz]
        out = term if out is None else out + term
    return out


def take_rows(xp, a, idx):
    """a[b, idx[b]] for a of shape (B, n)."""
    return xp.take_along_axis(a, idx[:, None], axis=1)[:, 0]


def posterior_stats(xp, chi2, prior, z, cfg, kernel=None):
    """Everything downstream of the chi^2 grid.

    ``prior`` is (B, nz, nt) or None (flat, i.e. ones).  ``cfg`` holds the
    python scalars mode, p_min, oi, min_rms, odds_ci, z_thr, zmax.
    Returns a dict of (B,)/(B, nz) arrays.
    """
    B, nz, nt = chi2.shape
    flat = chi2.reshape(B, nz * nt)
    imin = xp.argmin(flat, axis=1)
    chi2_min = take_rows(xp, flat, imin)

    # post = likelihood * prior, formed in log space and rescaled per galaxy
    # by exp(-shift) so that its maximum is 1.  Every statistic below is
    # invariant to that scale except two absolute thresholds of the original
    # code, which are evaluated in log space.  In float64 this is the loop's
    # arithmetic to rounding; in float32 it avoids products of two small
    # numbers underflowing (GPUs flush float32 subnormals to zero).
    logpost = -0.5 * xp.clip(chi2 - chi2_min[:, None, None], 0.0, CHI2_CLIP)
    if prior is not None:
        logpost = logpost + xp.log(prior)
    shift = logpost.reshape(B, nz * nt).max(axis=1)
    shift = xp.where(xp.isfinite(shift), shift, 0.0)
    post = xp.exp(logpost - shift[:, None, None])
    pz = post.sum(axis=-1)
    if kernel is not None:
        pz = convolve_same(xp, pz, kernel)

    iz_b = xp.argmax(pz, axis=1)
    pmax = take_rows(xp, pz, iz_b)
    thr = (pmax * cfg["p_min"])[:, None]
    if cfg["mode"] == "bpz":
        # bpz.py: where(p > pmax*P_MIN, p, 0); p /= sum(p)
        pz = xp.where(pz > thr, pz, 0.0)
        pz = pz / pz.sum(axis=1, keepdims=True)
        pdf = pz
    else:
        # rail: p[p < pmax*p_min] = 0; normalise unless isclose(sum, 0),
        # i.e. unless the unscaled sum tot * exp(shift) <= 1e-8
        pz = xp.where(pz < thr, 0.0, pz)
        tot = pz.sum(axis=1, keepdims=True)
        log_tot = xp.log(xp.where(tot > 0, tot, 1.0)) + shift[:, None]
        norm = (tot > 0) & (log_tot > LOG_1E_8)
        pdf = xp.where(norm, pz / xp.where(tot > 0, tot, 1.0), pz * xp.exp(shift)[:, None])

    # Best template at the mode, and its marginal "odds"
    post_at = xp.take_along_axis(post, iz_b[:, None, None], axis=1)[:, 0, :]
    it_b = xp.argmax(post_at, axis=1)
    post_at_max = take_rows(xp, post_at, it_b)
    # bpz.py reports T_B = -1 if max(pb[iz_b, :]) < 1e-300
    t_b_valid = ~(xp.log(post_at_max) + shift < LOG_1E_300)
    tmarg = post.sum(axis=1)
    todds = take_rows(xp, tmarg, it_b) / tmarg.sum(axis=1)

    zmean = (z[None, :] * pz).sum(axis=1) / pz.sum(axis=1)

    # bpz_tools_py3.odds and .interval; searchsorted(a, v) == sum(a < v)
    cp = xp.cumsum(pz, axis=1)
    ctot = cp[:, -1]
    zb = z[iz_b]
    if cfg["z_thr"] > 0:
        zo1 = xp.full_like(zb, cfg["z_thr"])
        zo2 = xp.full_like(zb, cfg["zmax"])
    else:
        zo1 = zb - cfg["oi"] * cfg["min_rms"] * (1.0 + zb)
        zo2 = zb + cfg["oi"] * cfg["min_rms"] * (1.0 + zb)
    i1 = (z[None, :] < zo1[:, None]).sum(axis=1) - 1
    i2 = (z[None, :] < zo2[:, None]).sum(axis=1)
    lower = xp.where(i1 < 0, 0.0, take_rows(xp, cp, xp.clip(i1, 0, nz - 1)))
    upper = xp.where(i2 > nz - 1, ctot, take_rows(xp, cp, xp.clip(i2, 0, nz - 1)))
    odds = (upper - lower) / ctot

    q1 = (1.0 - cfg["odds_ci"]) / 2.0
    q2 = 1.0 - q1
    cpn = cp / ctot[:, None]
    j1 = xp.maximum((cpn < q1).sum(axis=1) - 1, 0)
    j2 = xp.minimum((cpn < q2).sum(axis=1), nz - 1)

    return dict(
        pdf=pdf,
        iz_b=iz_b,
        it_b=it_b,
        t_b_valid=t_b_valid,
        todds=todds,
        zmean=zmean,
        iz_ml=imin // nt,
        it_ml=imin % nt,
        chi2_min=chi2_min,
        odds=odds,
        iz1=j1,
        iz2=j2,
    )
