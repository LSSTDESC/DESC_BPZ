# DESC_BPZ: 

This repo contains a python3 port of the popular BPZ softwrae package (https://www.stsci.edu/~dcoe/BPZ/).  The code was modified from the bpz-1.99.3 version available at the given URL, upgraded to be python3 compatible, and modified to read and write hdf5 files rather than straight ascii.  The code is messy in many places, having gone through multiple iterations by multiple authors.  No promises are made as to functionality of multiple subsystems (e.g. plotting)

In order to run, you should set environment variables
`BPZPY3PATH` to point to the location of the DESC_BPZ directory
and
`BPZDATAPATH` to point to the location of the SED, FILTER, and AB files.
