import os

data_dir = os.environ.get("BPZDATAPATH", None)
fil_dir = None if data_dir is None else data_dir + "/FILTER/"
ab_dir = None if data_dir is None else data_dir + "/AB/"
sed_dir = None if data_dir is None else data_dir + "/SED/"


def set_data_dir(dirpath):
    global data_dir
    data_dir = dirpath
    set_ab_dir(dirpath + "/AB/")
    set_sed_dir(dirpath + "/SED/")
    set_fil_dir(dirpath + "/FILTER/")


def set_sed_dir(dirpath):
    global sed_dir
    sed_dir = dirpath
    if not sed_dir.endswith("/"):
        sed_dir = sed_dir + "/"


def set_ab_dir(dirpath):
    global ab_dir
    ab_dir = dirpath
    if not ab_dir.endswith("/"):
        ab_dir = ab_dir + "/"


def set_fil_dir(dirpath):
    global fil_dir
    fil_dir = dirpath
    if not fil_dir.endswith("/"):
        fil_dir = fil_dir + "/"


def get_fil_file(filename):
    if fil_dir is None:
        raise RuntimeError(
            "Either set BPZDATAPATH or call set_sed_dir or " "set_fil_dir"
        )
    return fil_dir + filename


def get_sed_file(filename):
    if sed_dir is None:
        raise RuntimeError(
            "Either set BPZDATAPATH or call set_sed_dir with a new " "path"
        )
    return sed_dir + filename


def get_ab_file(filename):
    if ab_dir is None:
        raise RuntimeError(
            "Either set BPZDATAPATH or call set_sed_dir with a new " "path"
        )
    return ab_dir + filename
