"""Build BPZ inputs the way ``scripts/bpz.py`` does, for tests and benchmarks.

Only the common path is supported: magnitudes (MAG=yes) in an hdf5
catalogue, AB calibration, no zero point offsets from the parameter file.
"""

import os

import numpy as np

from desc_bpz.bpz_tools_py3 import e_mag2frac
from desc_bpz.paths import get_ab_file, get_sed_file
from desc_bpz.useful_py3 import get_2Darray_hdf5, get_data, get_str, match_resol, params


def bpz_zgrid(zmin, zmax, dz):
    """The redshift grid of bpz.py."""
    return np.arange(zmin, zmax + dz, dz)


def load_templates(spectra_file, filters, zgrid, madau="no"):
    """f_mod (nz, nt, nf) from existing AB files (generating them if needed)."""
    from desc_bpz.bpz_tools_py3 import ABflux

    if not os.path.isabs(spectra_file):
        spectra_file = get_sed_file(spectra_file)
    spectra = [s[:-4] if s.endswith(".sed") else s for s in get_str(spectra_file, 0)]
    mflag = "nomadau" if madau == "no" else "withmadau"
    f_mod = np.zeros((len(zgrid), len(spectra), len(filters)))
    for it, sed in enumerate(spectra):
        for jf, filt in enumerate(filters):
            path = get_ab_file(f"{sed}.{filt}.{mflag}.AB")
            if not os.path.exists(path):  # pragma: no cover
                ABflux(sed, filt, madau=madau)
            zo, f0 = get_data(path, (0, 1))
            f_mod[:, it, jf] = match_resol(zo, f0, zgrid)
            if (f_mod[:, it, jf] < 0).any():
                f_mod[:, it, jf] = np.clip(f_mod[:, it, jf], 0.0, 1e300)
    return f_mod


def load_catalog(h5file, columns_file, min_magerr=0.001):
    """Fluxes, errors, m_0 and ids from a bpz.py style (.h5, .columns) pair.

    Returns dict(filters, flux, flux_err, m_0, id, z_s).
    """
    col = params()
    col.fromfile(columns_file)
    filters = [f for f in get_str(columns_file, 0) if f not in ("M_0", "OTHER", "ID", "Z_S")]
    fcols = tuple(col.d[f][0] for f in filters)
    ecols = tuple(col.d[f][1] for f in filters)
    zp_err = np.array([float(col.d[f][3]) for f in filters])
    zp_off = np.array([float(col.d[f][4]) for f in filters])
    # Keep the catalogue dtype: bpz.py does this preprocessing in the input
    # precision (float32 for test1000), and we want its exact inputs.
    f = np.array(get_2Darray_hdf5(h5file, fcols))
    ef = np.array(get_2Darray_hdf5(h5file, ecols))
    unobs, undet = -99.0, 99.0
    seen = (f > 0) & (f < undet)
    no_seen = f == undet
    no_observed = f == unobs
    ef = np.where(ef >= 0, np.clip(ef, min_magerr, 1e10), ef)
    f = np.where(seen, 10.0 ** (-0.4 * f), f)
    ef = np.where(seen, (10.0 ** (0.4 * ef) - 1.0) * f, ef)
    f = np.where(no_seen, 0.0, f)
    ef = np.where(no_seen, 10.0 ** (-0.4 * np.abs(ef)), ef)
    f = np.where(no_observed, 0.0, f)
    ef = np.where(no_observed, 0.0, ef)

    f = np.where(f < 0, 0.0, f)
    seen = (f > 0) & (ef > 0)
    no_seen = (f == 0) & (ef > 0)
    no_observed = (f == 0) & (ef == 0)
    with np.errstate(over="ignore"):  # 1e108 is inf in float32, as in bpz.py
        ef = np.where(no_observed, 1e108, ef)
    zp_frac = e_mag2frac(zp_err)
    ef = np.where(seen, np.sqrt(ef * ef + (zp_frac * f) ** 2), ef)
    ef = np.where(no_seen, np.sqrt(ef * ef + (zp_frac * (ef / 2.0)) ** 2), ef)
    zp_off = np.where(zp_off != 0, 10.0 ** (-0.4 * zp_off), 1.0)
    f = f * zp_off
    ef = ef * zp_off

    out = dict(filters=filters, flux=f, flux_err=ef)
    out["m_0"] = np.array(get_2Darray_hdf5(h5file, col.d["M_0"])).reshape(-1)
    out["id"] = np.array(get_2Darray_hdf5(h5file, col.d["ID"])).reshape(-1) if "ID" in col.d else None
    out["z_s"] = np.array(get_2Darray_hdf5(h5file, col.d["Z_S"]), dtype=float).reshape(-1) if "Z_S" in col.d else None
    return out


def bpz_prior_table(zgrid, m_0, nt, info="hdfn_gen", ninterp=0):
    """Prior table / index exactly as bpz.py's per-galaxy prior() calls."""
    from desc_bpz.bpz_tools_py3 import prior

    from ._prior import build_prior_table

    nz = len(zgrid)
    return build_prior_table(m_0, lambda m: prior(zgrid, m, info, nt, ninterp)[:nz, :nt])


def gaussian_kernel(sigma, dz):
    """bpz.py's CONVOLVE_P kernel (rail uses the same form)."""
    x = np.arange(-3.0 * sigma, 3.0 * sigma + dz / 10.0, dz)
    return np.exp(-((x / sigma) ** 2))


def resample_catalog(cat, n, seed=0, noise=1.0):
    """Synthetic catalogue of n galaxies: draw rows of ``cat`` with
    replacement and perturb the detected fluxes by ``noise`` x their errors."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(cat["flux"]), n)
    f = cat["flux"][idx].copy()
    ef = cat["flux_err"][idx].copy()
    det = (f > 0) & (ef < 1.0)
    f = np.where(det, np.abs(f + noise * ef * rng.standard_normal(f.shape)), f)
    m_0 = cat["m_0"][idx] + np.where(cat["m_0"][idx] < 90, 0.02 * rng.standard_normal(n), 0.0)
    return dict(flux=f, flux_err=ef, m_0=m_0)
