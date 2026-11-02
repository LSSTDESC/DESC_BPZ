# Batched / GPU BPZ: speed and accuracy

Benchmark of `desc_bpz.batched.estimate_batch` against the original per-galaxy loop
(`p_c_z_t` + prior, as in `scripts/bpz.py` and rail_bpz's `BPZliteEstimator`).
Produced with [bench_bpz_gpu.py](bench_bpz_gpu.py); raw numbers are in
[figures/timing.csv](figures/timing.csv) and [figures/accuracy.csv](figures/accuracy.csv).

- **Machine:** NVIDIA GeForce RTX 5070 Laptop GPU (8 GB, WSL2) and an AMD Ryzen AI 9 365.
- **Software:** Python 3.13.7, NumPy 2.3.3, JAX 0.7.1, CuPy 14.1.1 (CUDA runtime 12.9).
- **Problem:** test1000 templates and grid (CWWSB4, nz = 351, nt = 8, nf = 6).
- **Catalogues:** synthetic, made by resampling test1000 with photometric noise.
- **`bpz` mode:** test1000.pars settings: P_MIN 0.01, Gaussian convolution with σ = 0.03, ODDS 0.68, and the hdfn_gen prior tabulated per 0.1 mag.
- **`rail` mode:** rail defaults: p_min 0.005, no kernel, and the HDFN prior evaluated per galaxy.

## Throughput (galaxies / s)

Wall time of one `estimate_batch` call, median of 5. It includes host preparation, the host↔device
transfers, and returning every output to the host. The per-galaxy loop was timed up to 10⁴
galaxies; `*` marks values extrapolated linearly from that. The NumPy backend was not run at 10⁶.

The benchmark plots this table as `throughput.png` in its output directory (not committed).

**bpz mode**

| backend | bits | 10³ | 10⁴ | 10⁵ | 10⁶ | speed-up (largest N) |
|---|---|---|---|---|---|---|
| loop | 64 | 4,497 | 4,353 | 4,353* | 4,353* | 1× |
| numpy | 64 | 7,772 | 7,397 | 7,863 | — | 2× |
| numpy | 32 | 13,610 | 13,298 | 13,864 | — | 3× |
| jax | 64 | 57,629 | 216,555 | 356,668 | 355,490 | 82× |
| jax | 32 | 117,153 | 491,691 | 960,350 | 1,006,995 | 231× |
| cuda | 64 | 120,017 | 252,623 | 333,019 | 334,891 | 77× |
| cuda | 32 | 204,311 | 1,112,039 | 1,683,659 | 2,075,115 | 477× |

**rail mode**

| backend | bits | 10³ | 10⁴ | 10⁵ | 10⁶ | speed-up (largest N) |
|---|---|---|---|---|---|---|
| loop | 64 | 3,539 | 3,593 | 3,593* | 3,593* | 1× |
| numpy | 64 | 6,070 | 6,941 | 6,950 | — | 2× |
| numpy | 32 | 10,586 | 11,095 | 11,249 | — | 3× |
| jax | 64 | 36,185 | 169,286 | 266,367 | 283,870 | 79× |
| jax | 32 | 56,754 | 532,069 | 986,144 | 903,871 | 252× |
| cuda | 64 | 89,017 | 172,224 | 196,187 | 196,501 | 55× |
| cuda | 32 | 210,921 | 915,722 | 1,422,499 | 1,250,896 | 348× |

What the timings show:

- **FP32 vs FP64.** This laptop GPU runs FP64 arithmetic at about 1/64 of the FP32 rate. Both GPU backends are 3–6× faster in FP32 than in FP64.
- **CUDA vs JAX.** The hand-written CUDA kernels beat JAX in FP32 by 1.3–2×. In rail-mode FP64, JAX/XLA is faster, by about 1.4×. The per-galaxy prior normalisation in the CUDA kernel runs nt sequential block reductions with `pow`/`exp`, which is expensive in FP64 on this card.
- **Start-up cost.** JAX compiles once per chunk shape, which takes about 0.8–0.9 s; chunks are padded to a fixed size, so this happens once per run. CuPy compiles the CUDA kernels with NVRTC on first use and caches them on disk.
- **Small catalogues.** At 10³ galaxies, launch and transfer latency dominate. The GPU gains grow until about 10⁵ galaxies per call.

### End to end through rail_bpz

These are `BPZliteEstimator.estimate` timings on 10⁵ galaxies (`chunk_size=100000`). They include rail's preprocessing, building the qp ensemble, and writing the HDF5 output:

| backend | loop | numpy 64 | jax 64 | jax 32 | cuda 64 | cuda 32 |
|---|---|---|---|---|---|---|
| galaxies / s | 3,976 (10⁴) | 7,088 (10⁴) | 52,414 | 60,732 | 46,251 | 62,549 |

The estimation itself drops to about 5% of the stage time. What remains is rail overhead, mostly the qp ensemble and its output, so the end-to-end gain is about 12–15×.

## Accuracy

Each backend is compared with the float64 loop on 10⁴ synthetic galaxies.

The benchmark plots these errors as `accuracy.png` in its output directory (not committed).

The PDF error is `max_z |ΔPDF| / PDF peak` per galaxy.

| mode | backend | bits | PDF err median | p99 | worst | zmode equal | T_B equal | χ²_min rel. err | ODDS abs. err |
|---|---|---|---|---|---|---|---|---|---|
| bpz | numpy / jax / cuda | 64 | 1e-13 | 1.3e-12 | 5.4e-12 | 100 % | 100 % | 2.9e-11 | 2.8e-12 |
| bpz | numpy / jax / cuda | 32 | 8–10e-7 | 1.0–1.3e-5 | 1.0e-2 † | 100 % | 100 % | 1.9–2.2e-5 | 1.3e-3 † |
| bpz | " " (expanded χ²) | 32 | 4.5–5e-5 | 6.4–7.2e-4 | 1.0e-2 | 99.97 % | 100 % | 1.1–1.9e-2 | 2.8–3.4e-3 |
| rail | numpy / jax / cuda | 64 | 5e-13 | 7.0e-12 | 2.0e-11 | 100 % | 100 % | 2.9e-11 | — |
| rail | numpy / jax / cuda | 32 | 3.5–4.3e-6 | 5.6–7.1e-5 | 2.3e-4 | 100 % | 100 % | 1.9–2.2e-5 | — |
| rail | " " (expanded χ²) | 32 | 2.3–2.6e-4 | 3.4–3.8e-3 | 1.2e-2 | 99.84–99.87 % | 100 % | 1.1–1.9e-2 | — |

What the accuracy figures show:

- **FP64 matches the loop to rounding.** The batched backends reproduce the loop to about 1e-11 or better, with identical modes, templates, ML redshifts and confidence intervals.
- **FP32 is accurate to about 1e-5 of the PDF peak.** Every zmode and T_B is identical at N = 10⁴, and that holds for all three backends because they run the same algorithm.
  - † The bpz-mode worst case, ≈ P_MIN = 0.01, comes from single bins that sit exactly at the `P_MIN × peak` trimming threshold and land on the other side of it in FP32. The ODDS outliers have the same cause.
- **The χ² form is what makes FP32 work.** The textbook `foo − fot²/ftt` loses most of its digits in FP32 for high-S/N galaxies. That costs about 1e-2 relative error in χ²_min and changes 0.03–0.16 % of the modes. The backends instead evaluate the cancellation-free residual form `Σ (y − a·x)²` on whitened fluxes. Those whitened inputs also avoid the FP32 overflow of the `1e108`/`1e28` "unobserved" errors.
- **The posterior is formed in log space.** It is built as `exp(log L + log P − max)`. Otherwise the products of two small numbers underflow, and GPUs flush FP32 subnormals to zero. rail's `isclose(sum, 0)` test and bpz.py's `T_B = −1` test (max < 1e-300) are evaluated in log space, so they behave the same in both precisions.
- **rail leaves some PDFs unnormalised.** For 122 of the 10⁴ galaxies the posterior mass is ≤ 1e-8, so rail returns the absolute values. These are not representable in FP32 and come out as 0, with absolute differences ≤ 1e-12. The PDF errors above exclude these galaxies.
- **rail's float32 catalogues.** rail's `_preprocess_magnitudes` keeps float32 inputs as float32, so the loop itself evaluates `f/ef` and `ef**2` in float32. On the native float32 test catalogue, the float64 backends therefore differ from the loop by up to about 4e-6, while matching to 1e-10 when the same catalogue is cast to float64. Neither case changes any zmode. The batched path whitens in float64 and is the more accurate of the two.

## Validation gates

All of these pass, and are checked by `tests/test_backends.py`, `tests/test_bpz_script.py` and
`tests/test_batched_numpy.py`.

**FP64** (every backend; `bpz` and `rail` modes; test1000 and 10⁴ synthetic galaxies):
- PDFs agree with rtol 1e-9 and atol 1e-12.
- zmode, T_B, ML (z, t), interval indices and the `T_B = −1` flag are identical.
- χ²_min, ODDS, todds and zmean agree with rtol 1e-9.

**FP32:**
- ≥ 99.5 % of zmodes are identical, and every mismatch is within 2 grid cells or a near-tie.
- The median over galaxies of the PDF error is ≤ 1e-5 of the peak.

**`bpz.py` on test1000:**
- With `-BACKEND jax|cuda|numpy -PRECISION float64` it reproduces the golden outputs regenerated from `main`. Z_B, Z_B_MIN/MAX, T_B, Z_ML and T_ML are bit-identical; CHI_SQ, ODDS and PDF match to float32 storage precision.
- With `-PRECISION float32`: ≤ 0.5 % changed Z_B and T_B, |ΔODDS| ≤ 1e-3, |ΔPDF| ≤ 1e-4.
- The committed `tests/test1000.bpz.h5` and `tests/test1000_probs.h5` predate the current code. The loop on `main` no longer reproduces them (|ΔCHI_SQ| ≤ 4e-3, one Z_ML shift), so new golden files were written to [../tests/golden/](../tests/golden).

**rail_bpz** (branch `gpu-backends`): the numeric regression tests pass with every backend, and each backend matches the loop (see the rail_bpz test suite).

## Reproduce

```bash
conda activate rail
python benchmarks/bench_bpz_gpu.py --backends numpy jax cuda --dtypes float64 float32 \
    --modes bpz rail --sizes 1000 10000 100000 1000000 --loop-max 10000 --acc-size 10000
python benchmarks/bench_bpz_gpu.py --replot   # redraw figures from benchmarks/outputs/results.json
```
