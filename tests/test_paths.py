import os
import pytest
dirname, _ = os.path.split(__file__)
parent = os.path.join(dirname, '../src/desc_bpz/data_files/')


# This test has to go first 
def test_no_path():
    tmp = os.environ.get("BPZDATAPATH")
    if tmp is not None:
        del os.environ["BPZDATAPATH"]

    import desc_bpz.paths

    # Future proof in case a later test import the
    # module and are called first
    if desc_bpz.paths.data_dir is not None:
        # In this case first restore the env var
        if tmp is not None:
            os.environ["BPZDATAPATH"] = tmp
        # then mark the test as skipped
        pytest.skip("desc_bpz already imported by another test and path set")

    # should fail if BPZDATAPATH not set and we have not set manually
    with pytest.raises(RuntimeError):
        print(desc_bpz.paths.get_ab_file("El_B2004a.DC2LSST_g.AB"))

    # restore env var
    if tmp is not None:
        os.environ["BPZDATAPATH"] = tmp


def test_path():
    import desc_bpz.paths
    desc_bpz.paths.set_data_dir(parent)
    assert os.path.exists(desc_bpz.paths.get_ab_file("El_B2004a.DC2LSST_g.AB"))
    assert os.path.exists(desc_bpz.paths.get_fil_file("DC2LSST_g.res"))
    assert os.path.exists(desc_bpz.paths.get_sed_file("CWWSB4.list"))
