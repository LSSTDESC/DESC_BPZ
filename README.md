# DESC_BPZ: 

This repo contains a python3 port of the popular BPZ software package (https://www.stsci.edu/~dcoe/BPZ/).  The code was modified from the original bpz-1.99.3 version available at the given URL, upgraded to be python3 compatible, and modified to read and write hdf5 files rather than straight ascii.  The code is messy in many places, having gone through multiple iterations by multiple authors.  No promises are made as to functionality of multiple subsystems (e.g. plotting), so use with caution!

If you use this code, please cite [Benitez (2000)](https://ui.adsabs.harvard.edu/abs/2000ApJ...536..571B/abstract) and [Coe et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006AJ....132..926C/abstract).

In order to run, you should set the environment variable


`BPZDATAPATH` to point to the directory housing the SED, FILTER, and AB files.  Default SED and FILTER files are available in the
`src/desc_bpz/data_files/FILTER` and `src/desc_bpz/data_files/SED` directories


To run traditional command-line `BPZ` use the `bpz.py` file in `src/desc_bpz/scripts`, e.g.:


`python src/desc_bpz/scripts/bpz.py yourasciicatalog.cat -P yourparameterfile.pars`


Dan Coe's STScI webpage (listed above) contains a nice BPZ user guide, for more detail on running the code, please check that website.


However, this package will mainly be used as a library for running "bpz_lite" in the DESC PZ RAIL Package, see:

https://github.com/LSSTDESC/RAIL


## Batched and GPU estimation

`desc_bpz.batched.estimate_batch` evaluates many galaxies at once instead of looping over them.
It reproduces both the `bpz.py` loop (`mode="bpz"`) and the loop of rail_bpz's `BPZliteEstimator` (`mode="rail"`).
Three backends are available:

- `numpy`: vectorised CPU implementation, no extra dependencies;
- `jax`: JAX/XLA, install with `pip install 'desc-bpz[jax]'`;
- `cuda`: hand-written CUDA kernels driven by CuPy, install with `pip install 'desc-bpz[cuda]'`.

Each backend runs in `float64` (matches the loop to about 1e-11) or `float32` (about 1e-5 of the PDF peak, much faster on consumer GPUs).
In `bpz.py` select them with the new parameters, e.g.

`python src/desc_bpz/scripts/bpz.py test1000.h5 -P test1000.pars -BACKEND cuda -PRECISION float32`

`BACKEND` defaults to `loop`, the original per-galaxy code.
Options the batched path does not support (`N_PEAKS>1`, `COLOR`, `ONLY_TYPE`, the lensing prior, `PROBS`, `PROBS2`, `ADD_SPEC_PROB`) fall back to the loop with a notice.
Speed and accuracy measurements are in [benchmarks/RESULTS.md](benchmarks/RESULTS.md).
