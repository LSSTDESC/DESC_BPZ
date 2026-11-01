"""Run scripts/bpz.py on test1000 and compare with the golden outputs.

The golden files in tests/golden/ were produced by bpz.py on main (the
original per-galaxy loop); the committed tests/test1000*.h5 outputs predate
the current code and differ from it at the 1e-4 level, so they are not used.
"""

import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest
from conftest import DATADIR, TESTDIR

SCRIPT = os.path.join(TESTDIR, "../src/desc_bpz/scripts/bpz.py")
GOLDEN = os.path.join(TESTDIR, "golden", "test1000_main.bpz.h5")
GOLDEN_PROBS = os.path.join(TESTDIR, "golden", "test1000_main_probs.h5")


def run_bpz(tmp_path, *extra):
    for name in ("test1000.h5", "test1000.pars", "CSDC2114_test.columns"):
        shutil.copy(os.path.join(TESTDIR, name), tmp_path)
    out, probs = tmp_path / "out.bpz.h5", tmp_path / "out_probs.h5"
    cmd = [sys.executable, SCRIPT, "test1000.h5", "-P", "test1000.pars",
           "-OUTPUT", str(out), "-PROBS_LITE", str(probs)] + list(extra)
    env = dict(os.environ, BPZDATAPATH=DATADIR)
    proc = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    with h5py.File(out) as f:
        res = {k: f[k][:] for k in f}
    with h5py.File(probs) as f:
        res["PDF"] = f["PDF"][:]
    return res, proc.stdout


@pytest.fixture(scope="module")
def golden():
    with h5py.File(GOLDEN) as f:
        res = {k: f[k][:] for k in f}
    with h5py.File(GOLDEN_PROBS) as f:
        res["PDF"] = f["PDF"][:]
    return res


def test_loop_reproduces_golden(tmp_path, golden):
    res, _ = run_bpz(tmp_path)
    for k in golden:
        np.testing.assert_array_equal(res[k], golden[k], err_msg=k)
