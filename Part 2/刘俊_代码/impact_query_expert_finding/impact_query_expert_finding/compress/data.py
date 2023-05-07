from os.path import join
import numpy as np
from struct import pack, unpack
from struct import pack


#################################################################
# Small I/O functions
#################################################################

def write_fvecs(filename, vecs):
    with open(filename, "wb") as f:
        for vec in vecs:
            dim = len(vec)
            f.write(pack('<i', dim))
            f.write(pack('f' * dim, *list(vec)))


def write_ivecs(filename, vecs):
    with open(filename, "wb") as f:
        for vec in vecs:
            dim = len(vec)
            f.write(pack('<i', dim))
            f.write(pack('i' * dim, *list(vec)))


def read_ivecs(fname):
    print(fname)
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def read_fvecs(fname):
    print(fname)
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]


#################################################################
# Dataset
#################################################################

def getBasedir(s):
    start = "/home/zjlab/ANNS/dataset/sample/dataset/"
    paths = {
        "sift": start + "sift1M/sift",
        "gist": start + "gist/gist",
    }

    return paths[s]


def load_simple(device, database):
    basedir = getBasedir(database)

    xt = read_fvecs("/home/zjlab/ANNS/dataset/" + database + "/" + database + '_learn.fvecs')
    xb = read_fvecs(basedir + '_sample_base.fvecs')
    xq = read_fvecs(basedir + '_sample_query.fvecs')
    gt = read_ivecs(basedir + '_sample_groundtruth.ivecs')

    xb, xq, xt = np.ascontiguousarray(xb), np.ascontiguousarray(xq), np.ascontiguousarray(xt)
    return xt, xb, xq, gt


def load_embedding(device, database):
    print("hello world")

def load_dataset(name, device):
    return load_simple(device, name)


def load_compress_dataset(name, device):
    basedir = "/home/zjlab/ANNS/yq/paper/BREWESS/results/data/" + name + "/" + name

    xb = read_fvecs(basedir + "_base_triplet.fvecs")
    xq = read_fvecs(basedir + "_query_triplet.fvecs")
    gt = read_ivecs(basedir + "_groundtruth_triplet.ivecs")

    xb, xq = np.ascontiguousarray(xb), np.ascontiguousarray(xq)

    return xb, xq, gt
