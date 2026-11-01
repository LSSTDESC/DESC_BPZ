"""CUDA backend: hand written kernels (kernels.cu) driven through CuPy.

The kernels are compiled at first use with NVRTC (``cupy.RawModule``) for
float and double.  Galaxies are processed in chunks on two CUDA streams so
that the host<->device copies of one chunk overlap with the kernels of the
other; results are staged through pinned host memory.
"""

import math
import os

import numpy as np

import cupy as cp

from ._prep import CHI2_CLIP

name = "cuda"

BLOCK = 256
_N_IOUT, _N_FOUT = 7, 4
_PRIOR_KIND = {"none": 0, "table": 1, "params": 2}
_module = None


def is_available():
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def _kernels(dtype):
    global _module
    if _module is None:
        with open(os.path.join(os.path.dirname(__file__), "kernels.cu")) as f:
            code = f.read()
        names = [f"{k}<{t}>" for k in ("bpz_chi2", "bpz_posterior") for t in ("float", "double")]
        _module = cp.RawModule(code=code, options=("-std=c++17",), name_expressions=names)
    t = "float" if dtype == np.float32 else "double"
    return _module.get_function(f"bpz_chi2<{t}>"), _module.get_function(f"bpz_posterior<{t}>")


def _chunk_size(nz, nt, nf, dtype, params_prior):
    free, _ = cp.cuda.runtime.memGetInfo()
    item = np.dtype(dtype).itemsize
    per_gal = item * (nz * nt * (2 if params_prior else 1) + nz + 2 * nf + 8) + 4 * (_N_IOUT + 1)
    # two streams, keep a safety margin
    return int(max(1, min(1 << 16, 0.35 * free / (2 * per_gal))))


class _Slot:
    """Device buffers + pinned staging area for one in-flight chunk."""

    def __init__(self, B, nz, nt, nf, dtype, params_prior):
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.work = cp.empty((B, nz, nt), dtype)
        self.pbuf = cp.empty((B, nz, nt), dtype) if params_prior else cp.empty(1, dtype)
        self.pdf = cp.empty((B, nz), dtype)
        self.iout = cp.empty((B, _N_IOUT), np.int32)
        self.fout = cp.empty((B, _N_FOUT), dtype)
        self.h_pdf = _pinned((B, nz), dtype)
        self.h_iout = _pinned((B, _N_IOUT), np.int32)
        self.h_fout = _pinned((B, _N_FOUT), dtype)
        self.pending = None  # slice of the chunk whose results are in flight


def _pinned(shape, dtype):
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    mem = cp.cuda.alloc_pinned_memory(n)
    return np.frombuffer(mem, dtype, int(np.prod(shape))).reshape(shape)


def run(inputs, cfg):
    dtype = np.dtype(inputs["dtype"])
    T = dtype.type
    ng = inputs["y"].shape[0]
    nz, nt, nf = inputs["m"].shape
    k_chi2, k_post = _kernels(dtype)
    kind = inputs["prior_kind"]
    params_prior = kind == "params"

    dev = lambda a, dt=dtype: cp.asarray(np.ascontiguousarray(a, dtype=dt))  # noqa: E731
    m, z = dev(inputs["m"]), dev(inputs["z"])
    y, s, c = (np.ascontiguousarray(inputs[k], dtype=dtype) for k in ("y", "s", "c"))
    ker = inputs["kernel"]
    nk = 0 if ker is None else len(ker)
    d_ker = dev(ker if ker is not None else np.zeros(1))
    dummy = cp.zeros(1, dtype)
    d_table = d_zt = d_a = d_zo = d_km = d_kt = d_fo = dummy
    n_most = n_last = 0
    if kind == "table":
        d_table = dev(inputs["prior_table"])
        per_gal = np.ascontiguousarray(inputs["prior_index"], dtype=np.int32)
    elif kind == "params":
        pp = inputs["prior_params"]
        d_zt, d_a, d_zo, d_km, d_kt, d_fo = (dev(v) for v in (pp.zt_at_a, pp.a, pp.zo, pp.km, pp.k_t, pp.fo_t))
        n_most, n_last = pp.n_most, pp.n_last
        per_gal = np.ascontiguousarray(inputs["dm"], dtype=dtype)
    else:
        per_gal = np.zeros(ng, dtype=np.int32)

    B = inputs["chunk_size"] or _chunk_size(nz, nt, nf, dtype, params_prior)
    B = min(B, ng)
    slots = [_Slot(B, nz, nt, nf, dtype, params_prior) for _ in range(2 if ng > B else 1)]
    smem = (2 * nz + 2 * nt) * dtype.itemsize
    q1 = (1.0 - cfg["odds_ci"]) / 2.0
    q2 = 1.0 - q1
    scalars = dict(
        tail_thr=T(1e-2 / float(nz)),
        mode=np.int32(1 if cfg["mode"] == "bpz" else 0),
        chi2_clip=T(CHI2_CLIP),
        log_1e8=T(math.log(1e-8)),
        log_1e300=T(math.log(1e-300)),
        p_min=T(cfg["p_min"]),
        oi_rms=T(cfg["oi"] * cfg["min_rms"]),
        q1=T(q1),
        q2=T(q2),
        use_zthr=np.int32(cfg["z_thr"] > 0),
        z_thr=T(cfg["z_thr"]),
        zmax=T(cfg["zmax"]),
    )
    expanded = np.int32(cfg["chi2_form"] == "expanded")
    if cfg["chi2_form"] not in ("residual", "expanded"):
        raise ValueError(f"unknown chi2_form {cfg['chi2_form']!r}")

    out = dict(
        pdf=np.empty((ng, nz), dtype),
        iout=np.empty((ng, _N_IOUT), np.int32),
        fout=np.empty((ng, _N_FOUT), dtype),
    )

    def drain(slot):
        if slot.pending is None:
            return
        slot.stream.synchronize()
        sl = slot.pending
        n = sl.stop - sl.start
        out["pdf"][sl] = slot.h_pdf[:n]
        out["iout"][sl] = slot.h_iout[:n]
        out["fout"][sl] = slot.h_fout[:n]
        slot.pending = None

    for i, start in enumerate(range(0, ng, B)):
        slot = slots[i % len(slots)]
        drain(slot)
        sl = slice(start, min(start + B, ng))
        n = sl.stop - sl.start
        with slot.stream:
            d_y, d_s, d_c = cp.asarray(y[sl]), cp.asarray(s[sl]), cp.asarray(c[sl])
            d_pg = cp.asarray(per_gal[sl])
            nel = n * nz * nt
            grid = min((nel + BLOCK - 1) // BLOCK, 65535 * 8)
            k_chi2((grid,), (BLOCK,), (m, d_y, d_s, d_c, np.int32(n), np.int32(nz), np.int32(nt),
                                       np.int32(nf), expanded, slot.work))
            index = d_pg if kind == "table" else cp.zeros(1, np.int32)
            dm = d_pg if kind == "params" else dummy
            k_post((n,), (BLOCK,), (
                slot.work, slot.pbuf, np.int32(n), np.int32(nz), np.int32(nt),
                np.int32(_PRIOR_KIND[kind]), d_table, index,
                d_zt, d_a, d_zo, d_km, d_kt, d_fo, np.int32(n_most), np.int32(n_last), dm,
                scalars["tail_thr"], z, d_ker, np.int32(nk),
                scalars["mode"], scalars["chi2_clip"], scalars["log_1e8"], scalars["log_1e300"],
                scalars["p_min"], scalars["oi_rms"], scalars["q1"], scalars["q2"],
                scalars["use_zthr"], scalars["z_thr"], scalars["zmax"],
                slot.pdf, slot.iout, slot.fout,
            ), shared_mem=smem)
            slot.pdf[:n].get(out=slot.h_pdf[:n], stream=slot.stream, blocking=False)
            slot.iout[:n].get(out=slot.h_iout[:n], stream=slot.stream, blocking=False)
            slot.fout[:n].get(out=slot.h_fout[:n], stream=slot.stream, blocking=False)
        slot.pending = sl
    for slot in slots:
        drain(slot)

    io, fo = out["iout"], out["fout"]
    return dict(
        pdf=out["pdf"],
        iz_b=io[:, 0],
        it_b=io[:, 1],
        t_b_valid=io[:, 2].astype(bool),
        iz_ml=io[:, 3],
        it_ml=io[:, 4],
        iz1=io[:, 5],
        iz2=io[:, 6],
        chi2_min=fo[:, 0],
        todds=fo[:, 1],
        zmean=fo[:, 2],
        odds=fo[:, 3],
    )
