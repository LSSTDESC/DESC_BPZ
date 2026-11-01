"""Speed and accuracy benchmark of the batched BPZ backends.

Synthetic catalogues are built by resampling test1000 (with photometric
noise); templates, grid and prior follow test1000.pars (nz=351, nt=8, nf=6).

* timing: wall time of ``estimate_batch`` (host preparation, host<->device
  transfers and compute; results are back on the host when it returns).
  The first call (JIT / kernel compilation) is reported separately and the
  steady state is the median of ``--repeats`` calls.  The per-galaxy
  ``loop`` (p_c_z_t + prior, as in bpz.py / rail_bpz) is timed up to
  ``--loop-max`` galaxies and extrapolated linearly beyond (flagged).
* accuracy: every backend / precision against the float64 loop on
  ``--acc-size`` galaxies.

Usage:
    python benchmarks/bench_bpz_gpu.py --backends numpy jax cuda \
        --dtypes float64 float32 --sizes 1000 10000 100000 1000000
"""

import argparse
import json
import os
import time
from functools import partial

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TESTS = os.path.join(REPO, "tests")
DATADIR = os.path.join(REPO, "src", "desc_bpz", "data_files")

MODE_KW = {
    # test1000.pars: P_MIN 0.01, CONVOLVE_P yes (sigma 0.03), ODDS 0.68
    "bpz": dict(p_min=0.01, odds=0.68, min_rms=0.05),
    # rail_bpz defaults: p_min 0.005, no kernel
    "rail": dict(p_min=0.005),
}


def load_inputs():
    import desc_bpz.paths
    from desc_bpz.batched import prior_params_hdfn
    from desc_bpz.batched.testing import bpz_zgrid, gaussian_kernel, load_catalog, load_templates

    desc_bpz.paths.set_data_dir(DATADIR)
    cat = load_catalog(os.path.join(TESTS, "test1000.h5"), os.path.join(TESTS, "CSDC2114_test.columns"))
    z = bpz_zgrid(0.005, 3.505, 0.01)
    f_mod = load_templates("CWWSB4.list", cat["filters"], z)
    return dict(cat=cat, z=z, f_mod=f_mod, kernel=gaussian_kernel(0.03, 0.01), prior_params=prior_params_hdfn())


def mode_kwargs(inp, mode, sample):
    from desc_bpz.batched.testing import bpz_prior_table

    if mode == "bpz":
        table, index = bpz_prior_table(inp["z"], sample["m_0"], inp["f_mod"].shape[1])
        return dict(MODE_KW["bpz"], prior_table=table, prior_index=index, kernel=inp["kernel"])
    return dict(MODE_KW["rail"], mag0=sample["m_0"], prior_params=inp["prior_params"])


def run_loop(inp, mode, sample, kw):
    from desc_bpz.batched.reference import loop_bpz, loop_rail

    args = (sample["flux"], sample["flux_err"], inp["f_mod"], inp["z"])
    if mode == "bpz":
        return loop_bpz(*args, **{k: v for k, v in kw.items()})
    return loop_rail(*args, mag0=kw["mag0"], prior_params=kw["prior_params"], p_min=kw["p_min"])


def run_batched(inp, mode, sample, kw, backend, dtype, form="residual", chunk=None):
    from desc_bpz.batched import estimate_batch

    return estimate_batch(sample["flux"], sample["flux_err"], inp["f_mod"], inp["z"], mode=mode,
                          backend=backend, dtype=dtype, chi2_form=form, chunk_size=chunk, **kw)


def accuracy(res, ref, mode):
    """Errors against the float64 loop.

    PDF errors are max_z |dPDF| / peak per galaxy over galaxies with a
    normalised reference PDF.  rail leaves the PDF unnormalised (absolute
    values, typically 1e-70) when the posterior mass is <= 1e-8; those
    galaxies are reported separately with their absolute error, since such
    values are not representable in float32.
    """
    peak = ref["pdf"].max(axis=1)
    tot = ref["pdf"].sum(axis=1)
    unnorm = ~(np.abs(tot - 1.0) < 1e-6)
    ok = (peak > 0) & ~unnorm
    diff = np.abs(res["pdf"].astype(np.float64) - ref["pdf"])
    rel = diff[ok].max(axis=1) / peak[ok]
    chi_rel = np.abs(res["chi2_min"] - ref["chi2_min"]) / np.maximum(np.abs(ref["chi2_min"]), 1e-300)
    out = dict(
        pdf_rel_max=float(rel.max()),
        pdf_rel_p99=float(np.percentile(rel, 99)),
        pdf_rel_median=float(np.median(rel)),
        n_unnormalised=int(unnorm.sum()),
        unnormalised_absmax=float(diff[unnorm].max()) if unnorm.any() else 0.0,
        zmode_equal=float(np.mean(res["iz_b"] == ref["iz_b"])),
        zmode_maxdiff=float(np.abs(res["zmode"] - ref["zmode"]).max()),
        tb_equal=float(np.mean(res["t_b"] == ref["t_b"])),
        zml_equal=float(np.mean((res["iz_ml"] == ref["iz_ml"]) & (res["it_ml"] == ref["it_ml"]))),
        chi2_rel_max=float(np.nanmax(chi_rel)),
        todds_absmax=float(np.nanmax(np.abs(res["todds"] - ref["todds"]))),
        zmean_absmax=float(np.nanmax(np.abs(res["zmean"] - ref["zmean"]))),
    )
    if mode == "bpz":
        out["odds_absmax"] = float(np.abs(res["odds"] - ref["odds"]).max())
        out["interval_equal"] = float(np.mean((res["iz1"] == ref["iz1"]) & (res["iz2"] == ref["iz2"])))
    return out


def timeit(fn, repeats):
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return first, times


def configs(backends, dtypes, foil):
    out = [(b, d, "residual") for b in backends for d in dtypes]
    if foil:
        out += [(b, "float32", "expanded") for b in backends if "float32" in dtypes]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", nargs="+", default=["numpy", "jax", "cuda"])
    ap.add_argument("--dtypes", nargs="+", default=["float64", "float32"])
    ap.add_argument("--modes", nargs="+", default=["bpz", "rail"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 100000, 1000000])
    ap.add_argument("--numpy-max", type=int, default=100000, help="largest N timed with the numpy backend")
    ap.add_argument("--loop-max", type=int, default=10000)
    ap.add_argument("--acc-size", type=int, default=10000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-foil", action="store_true", help="skip the float32 'expanded' chi2 foil")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--replot", action="store_true", help="only redraw the figures from <out>/results.json")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.replot:
        with open(os.path.join(args.out, "results.json")) as f:
            res = json.load(f)
        plot(res["timing"], res["accuracy"], args.out)
        return

    from desc_bpz.batched import get_backend
    from desc_bpz.batched.testing import resample_catalog

    backends = []
    for b in args.backends:
        try:
            if get_backend(b).is_available():
                backends.append(b)
                continue
        except ImportError as err:
            print(f"skipping {b}: {err}")
            continue
        print(f"skipping {b}: no device")

    inp = load_inputs()
    env = environment(backends)
    print(json.dumps(env, indent=1))
    timing, acc = [], []
    for mode in args.modes:
        # Accuracy against the float64 loop
        sample = resample_catalog(inp["cat"], args.acc_size, seed=1)
        kw = mode_kwargs(inp, mode, sample)
        ref = run_loop(inp, mode, sample, kw)
        for backend, dtype, form in configs(backends, args.dtypes, not args.no_foil):
            res = run_batched(inp, mode, sample, kw, backend, dtype, form)
            row = dict(mode=mode, backend=backend, dtype=dtype, chi2_form=form, n=args.acc_size)
            row.update(accuracy(res, ref, mode))
            acc.append(row)
            print("ACC", json.dumps(row))

        # Timing
        loop_rate = None
        for n in sorted(args.sizes):
            sample = resample_catalog(inp["cat"], n, seed=2)
            kw = mode_kwargs(inp, mode, sample)
            if n <= args.loop_max:
                t0 = time.perf_counter()
                run_loop(inp, mode, sample, kw)
                dt = time.perf_counter() - t0
                loop_rate = n / dt
                timing.append(dict(mode=mode, backend="loop", dtype="float64", chi2_form="-", n=n,
                                   first_s=dt, median_s=dt, min_s=dt, extrapolated=False))
            elif loop_rate is not None:
                timing.append(dict(mode=mode, backend="loop", dtype="float64", chi2_form="-", n=n,
                                   first_s=None, median_s=n / loop_rate, min_s=n / loop_rate, extrapolated=True))
            print("TIME", json.dumps(timing[-1]))
            for backend, dtype, form in configs(backends, args.dtypes, False):
                if backend == "numpy" and n > args.numpy_max:
                    continue
                reps = args.repeats if (backend != "numpy" or n <= 10000) else 1
                first, times = timeit(partial(run_batched, inp, mode, sample, kw, backend, dtype, form), reps)
                timing.append(dict(mode=mode, backend=backend, dtype=dtype, chi2_form=form, n=n,
                                   first_s=first, median_s=float(np.median(times)), min_s=float(np.min(times)),
                                   extrapolated=False))
                print("TIME", json.dumps(timing[-1]))

    for row in timing:
        loop = [r for r in timing if r["mode"] == row["mode"] and r["backend"] == "loop" and r["n"] == row["n"]]
        row["gal_per_s"] = row["n"] / row["median_s"]
        row["speedup_vs_loop"] = loop[0]["median_s"] / row["median_s"] if loop else None

    write_csv(os.path.join(args.out, "timing.csv"), timing)
    write_csv(os.path.join(args.out, "accuracy.csv"), acc)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(dict(environment=env, timing=timing, accuracy=acc), f, indent=1)
    plot(timing, acc, args.out)
    print("wrote", args.out)


def environment(backends):
    import platform

    env = dict(python=platform.python_version(), numpy=np.__version__, cpu=platform.processor() or platform.machine())
    try:
        with open("/proc/cpuinfo") as f:
            env["cpu"] = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    if "jax" in backends:
        import jax

        env["jax"] = jax.__version__
        env["gpu"] = str(jax.devices()[0].device_kind)
    if "cuda" in backends:
        import cupy

        env["cupy"] = cupy.__version__
        env["cuda_runtime"] = cupy.cuda.runtime.runtimeGetVersion()
        env["gpu"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    return env


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + "\n")


# Reference categorical palette (dataviz skill), fixed slot per backend
COLORS = {"loop": "#2a78d6", "numpy": "#eb6834", "jax": "#1baf7a", "cuda": "#eda100"}
STYLE = {"float64": dict(ls="-", marker="o"), "float32": dict(ls="--", marker="s")}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot(timing, acc, outdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(dict.fromkeys(r["mode"] for r in timing))
    fig, axes = plt.subplots(1, len(modes), figsize=(6.2 * len(modes), 4.6), squeeze=False, facecolor=SURFACE)
    for ax, mode in zip(axes[0], modes, strict=True):
        _style_axes(ax)
        rows = [r for r in timing if r["mode"] == mode and r["chi2_form"] in ("residual", "-")]
        for backend in COLORS:
            for dtype in STYLE:
                pts = sorted((r for r in rows if r["backend"] == backend and r["dtype"] == dtype), key=lambda r: r["n"])
                if not pts:
                    continue
                label = "loop (per galaxy)" if backend == "loop" else f"{backend} {dtype[-2:]}-bit"
                st = dict(STYLE[dtype])
                ax.plot([r["n"] for r in pts], [r["gal_per_s"] for r in pts], color=COLORS[backend], lw=2,
                        ms=8, mec=SURFACE, mew=1.5, label=label, **st)
                ext = [r for r in pts if r["extrapolated"]]
                if ext:
                    ax.plot([r["n"] for r in ext], [r["gal_per_s"] for r in ext], ls="none", marker="o", ms=8,
                            mfc=SURFACE, mec=COLORS[backend], mew=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("galaxies per call", color=INK2)
        ax.set_ylabel("throughput (galaxies / s)", color=INK2)
        ax.set_title(f"{mode} mode: throughput (open markers = extrapolated loop)", color=INK, fontsize=10, loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK2, ncol=7, loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(outdir, "throughput.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    modes = list(dict.fromkeys(r["mode"] for r in acc))
    if not modes:
        return
    nrow = max(len([r for r in acc if r["mode"] == m]) for m in modes)
    fig, axes = plt.subplots(len(modes), 1, figsize=(8.5, (0.42 * nrow + 1.3) * len(modes)),
                             squeeze=False, facecolor=SURFACE)
    for ax, mode in zip(axes[:, 0], modes, strict=True):
        _style_axes(ax)
        rows = [r for r in acc if r["mode"] == mode]
        labels = [f"{r['backend']} {r['dtype'][-2:]}-bit" + (", expanded chi2" if r["chi2_form"] == "expanded" else "")
                  for r in rows]
        ypos = np.arange(len(rows))[::-1]
        for y, r in zip(ypos, rows, strict=True):
            lo, hi, top = (max(r[k], 1e-17) for k in ("pdf_rel_median", "pdf_rel_p99", "pdf_rel_max"))
            st, col = STYLE[r["dtype"]], COLORS[r["backend"]]
            ax.plot([lo, hi], [y, y], color=col, lw=2)
            ax.plot([lo], [y], ls="none", marker=st["marker"], ms=8, mfc=SURFACE, mec=col, mew=1.5)
            ax.plot([hi], [y], ls="none", marker=st["marker"], ms=8, color=col, mec=SURFACE, mew=1.5)
            ax.plot([top], [y], ls="none", marker="|", ms=12, mew=2, color=INK2)
            ax.annotate(f"zmode equal {100 * r['zmode_equal']:.2f}%", (1e-1, y), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, color=INK, fontsize=9)
        ax.set_xscale("log")
        ax.set_xlim(1e-15, 1e3)
        ax.set_title(f"{mode} mode: PDF error vs the float64 loop, N={rows[0]['n']}", color=INK, fontsize=10, loc="left")
    axes[-1, 0].set_xlabel("max_z |dPDF| / PDF peak per galaxy   (open: median, filled: 99th percentile, bar: worst)",
                           color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "accuracy.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
