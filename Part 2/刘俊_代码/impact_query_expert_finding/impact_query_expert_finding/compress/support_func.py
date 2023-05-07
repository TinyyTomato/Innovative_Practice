import time
import torch
import random
import ctypes
import itertools
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
# import graph.interface.nang as nang
from data import write_fvecs, write_ivecs
import impact_query_expert_finding.data.io as io

# swig_ptr = nang.swig_ptr

try:
    import faiss

    hasfaiss = True
except:
    hasfaiss = False

embedding_auth = np.array(
    io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "AE"))

A_A = np.array(io.load_as_json("/ddisk/lj/DBLP/data/", "A_A"))

xt = embedding_auth
xb = xt
xq = xt
gt = xt


def get_nearestneighbors_faiss(xq, xb, k, device, needs_exact=True, verbose=False):
    assert device in ["cpu", "cuda"]

    if verbose:
        print("Computing nearest neighbors (Faiss)")

    if needs_exact or device == 'cuda':
        index = faiss.IndexFlatL2(xq.shape[1])
    else:
        index = faiss.index_factory(xq.shape[1], "HNSW32")
        index.hnsw.efSearch = 64
    if device == 'cuda':
        index = faiss.index_cpu_to_all_gpus(index)

    start = time.time()
    index.add(xb)
    _, I = index.search(xq, k)
    if verbose:
        print("  NN search (%s) done in %.2f s" % (
            device, time.time() - start))
    return I


# def get_nearestneighbors_graph(xq, xb, k, device, graph, needs_exact=True, verbose=False):
#     assert device in ["cpu", "cuda"]
#     start = time.time()
#     _, I = index.search(xq, k)
#     if verbose:
#         print("  NN search (%s) done in %.2f s" % (
#             device, time.time() - start))
#     return I


def cdist2(A, B):
    return (A.pow(2).sum(1, keepdim=True)
            - 2 * torch.mm(A, B.t())
            + B.pow(2).sum(1, keepdim=True).t())


def top_dist(A, B, k):
    return cdist2(A, B).topk(k, dim=1, largest=False, sorted=True)[1]


def get_nearestneighbors_torch(xq, xb, k, device, needs_exact=False, verbose=False):
    if verbose:
        print("Computing nearest neighbors (torch)")

    assert device in ["cpu", "cuda"]
    start = time.time()
    xb, xq = torch.from_numpy(xb), torch.from_numpy(xq)
    xb, xq = xb.to(device), xq.to(device)
    bs = 500
    I = torch.cat([top_dist(xq[i * bs:(i + 1) * bs], xb, k)
                   for i in range(xq.size(0) // bs)], dim=0)
    if verbose:
        print("  NN search done in %.2f s" % (time.time() - start))
    I = I.cpu()
    return I.numpy()


# if hasfaiss:
#     get_nearestneighbors = get_nearestneighbors_faiss
# else:
#     get_nearestneighbors = get_nearestneighbors_torch


def get_nearestneighbors_graph(xq, xb, graph, k, device, needs_exact=True, verbose=False):
    # assert device in ["cpu", "cuda"]
    start = time.time()
    I = [[] for i in range(len(graph))]
    for i, a in enumerate(A_A):
        author_index = np.flatnonzero(A_A[i])[0:5]
        a_list = [i for _ in range(5 + 1)]
        for j, a in enumerate(author_index):
            a_list[j] = a
        I[i] = a_list
    return np.array(I, dtype="float32")


get_nearestneighbors = get_nearestneighbors_graph


def get_nearestneighbors_partly(xq, xb, k, device, bs=10 ** 5, needs_exact=True, path=""):
    knn = []

    for i0 in range(0, xq.shape[0], bs):
        xq_p = xq[i0:i0 + bs]
        res = get_nearestneighbors(xq_p, xb, k, device, needs_exact)
        knn.append(res)
    if path != "":
        write_ivecs(path, np.vstack(knn))
    return np.vstack(knn)


def calc_permutation(x, y, k):
    ans = 0
    for i in range(x.shape[0]):
        ans += len(list(set(x[i]) & set(y[i]))) / k
    return ans / x.shape[0]


def loss_permutation(x, y, A_A, args, k, size=10 ** 4):
    perm = np.random.permutation(x.shape[0])
    k_nn_x = get_nearestneighbors(x[perm[:size]], x, A_A, k, args.device, needs_exact=True)
    k_nn_y = get_nearestneighbors(y[perm[:size]], y, A_A, k, args.device, needs_exact=True)
    perm_coeff = calc_permutation(k_nn_x, k_nn_y, k)
    print('top %d permutation is %.3f' % (k, perm_coeff))
    return perm_coeff


def loss_top_1_in_lat_top_k(xs, x, ys, y, args, kx, ky, size, name, fake_args=False):
    if xs.shape[0] != ys.shape[0]:
        print("wrong data")
    perm = np.random.permutation(xs.shape[0])
    top1_x = get_nearestneighbors(xs[perm[:size]], x, A_A, kx, args.device, needs_exact=True)
    top_neg_y = get_nearestneighbors(ys[perm[:size]], y, A_A, ky, args.device, needs_exact=True)
    ans_in_top_neg = 0
    for i in range(top1_x.shape[0]):
        if top1_x[i, -1] in top_neg_y[i]:
            ans_in_top_neg += 1
    print('%s: Part of top1_x in gt_lat_ %d = %.4f' % (name, ky, ans_in_top_neg / len(top1_x)))


def sanitize(x):
    return np.ascontiguousarray(x, dtype='float32')


def batch_recall(res, gt, K):
    cnt = 0
    for i in range(gt.shape[0]):
        for a in res[i * K: i * K + K]:
            for b in gt[i][0: K]:
                if a == b:
                    cnt += 1
                    break
    return float(cnt / (gt.shape[0] * K))


def eval(xb, xq, gt, K):
    nang.build(swig_ptr(xb), xb.shape[0], xb.shape[1])

    L1, L2 = 5, 8
    for i in range(10):
        L = L1 + L2
        L1 = L2
        L2 = L

        res_len = K * xq.shape[0]
        res = np.array([0 for i in range(res_len)], dtype='int32')
        t0 = time.time()
        addr = nang.batch_search(swig_ptr(xb), swig_ptr(xq), swig_ptr(res), xb.shape[0], xb.shape[1], xq.shape[0], L, K)
        t1 = time.time()
        res = np.frombuffer((ctypes.c_int * res_len).from_address(int(addr)), np.int32)
        print("Recall@" + str(K), ":", batch_recall(res, gt, K))


def repeat(l, r):
    return list(itertools.chain.from_iterable(itertools.repeat(x, r) for x in l))


def forward_pass(net, xall, bs=128, device=None):
    if device is None:
        device = next(net.parameters()).device
    xl_net = []
    net.eval()
    for i0 in range(0, xall.shape[0], bs):
        x = torch.from_numpy(xall[i0:i0 + bs])
        x = x.to(device)
        x = x.type(torch.float32)
        res = net(x)
        xl_net.append(res.data.cpu().numpy())

    return np.vstack(xl_net)


def forward_pass_enc(enc, xall, bs=128, device=None):
    if device is None:
        device = next(enc.parameters()).device
    xl_net = []
    enc.eval()
    for i0 in range(0, xall.shape[0], bs):
        x = torch.from_numpy(xall[i0:i0 + bs])
        x = x.to(device)
        res, _ = enc(x)
        xl_net.append(res.data.cpu().numpy())

    return np.vstack(xl_net)


def save_transformed_data(ds, model, path, device, enc=False):
    # ds = torch.from_numpy(ds).to(device)
    if enc:
        xb_var = torch.from_numpy(ds).to(device)
        xb_var = xb_var / xb_var.norm(dim=-1, keepdim=True)
        ds = xb_var.detach().cpu().numpy()
        del xb_var

    # ds = forward_pass_model(model, ds, 1024, lat=True)
    if enc:
        ds = forward_pass_enc(model, ds, 1024)
    else:
        ds = forward_pass(model, ds, 1024)
    # file_for_write_base = "data/" + path
    file_for_write_base = "/home/zjlab/ANNS/yq/paper/BREWESS/results/data/" + path
    write_fvecs(file_for_write_base, ds)


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2., dim=1)
