# DESC_BPZ: 

This repo contains a python3 port of the popular BPZ softwrae package (https://www.stsci.edu/~dcoe/BPZ/).  The code was modified from the original bpz-1.99.3 version available at the given URL, upgraded to be python3 compatible, and modified to read and write hdf5 files rather than straight ascii.  The code is messy in many places, having gone through multiple iterations by multiple authors.  No promises are made as to functionality of multiple subsystems (e.g. plotting), so use with caution!

In order to run, you should set environment variables <br>

`BPZPY3PATH` to point to the location of the DESC_BPZ directory <br>
and <br>
`BPZDATAPATH` to point to the location of the SED, FILTER, and AB files.
<br>

To run traditional command-line `BPZ` use the `bpz.py` file in `scripts`.  However, this package will mainly be used as a library for running "bpz_lite" in the DESC PZ RAIL Package, see:<br>
https://github.com/LSSTDESC/RAIL
