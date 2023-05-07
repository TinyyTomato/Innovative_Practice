# from __future__ import division
import torch
import argparse
import numpy as np
# import triplet
# import angular
import torch
import argparse
import torch.nn.functional as F
from support_func import sanitize, eval
from data import load_dataset, load_compress_dataset
import impact_query_expert_finding.data.io as io
import time
from torch import nn, optim
from net import Normalize, forward_pass
from support_func import repeat
from triplet import validation_vanilla
import impact_query_expert_finding.data.sets as sets
import os
from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser()


def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)


group = parser.add_argument_group('dataset options')
aa("--database", default="V4")  # can be "sift", "gist"

group = parser.add_argument_group('Model hyperparameters')
aa("--dout", type=int, default=768,
   help="output dimension")
aa("--dint", type=int, default=768 * 2,
   help="size of hidden states")
aa("--method", type=str, default="triplet")  # can be "triplet" or "angular"
aa("--lambda_uniform", type=float, default=0.05,
   help="weight of the uniformity loss")
aa("--lambda_angular", type=float, default=0.05)

group = parser.add_argument_group('Training hyperparameters')
aa("--batch_size", type=int, default=2048)
aa("--epochs", type=int, default=64)
aa("--momentum", type=float, default=0.9)
aa("--rank_positive", type=int, default=5,
   help="this number of vectors are considered positives")
aa("--rank_negative", type=int, default=20,
   help="these are considered negatives")

group = parser.add_argument_group('Computation params')
aa("--seed", type=int, default=1234)
aa("--device", choices=["cuda", "cpu", "auto"], default="cuda")
aa("--lr_schedule", type=str, default="0.01,0.001,0.001,0.001")
aa("--val_freq", type=int, default=10,
   help="frequency of validation calls")

args = parser.parse_args()

# if args.device == "auto":
# args.device = "cuda" if torch.cuda.is_available() else "cpu"

print(args)
embedding_AECM = np.array(
    io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "AECM"))
embedding_docs = np.array(io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "V4_0.8_PAP_15core"))
# A_A = np.array(io.load_as_json("/ddisk/lj/DBLP/data/", "A_A"))
# embedding_AE = np.array(
#     io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "AE"))

# xt = np.vstack((embedding_docs, embedding_AEC))  # // with  document embedding
# xt = embedding_AEC  # w/o  document embedding

author_emebedings = embedding_AECM
# embedding_docs_vectors = normalize(embedding_AEC, norm='l2', axis=1)
pad = 0
for i, _ in enumerate(author_emebedings):
    author_emebedings[i] = np.array(author_emebedings[i], dtype="float32")
    pad = max(len(author_emebedings[i]), pad)

zeros = [0 for x in range(0, 768)]
arrays = []
for i, _ in enumerate(author_emebedings):
    if len(author_emebedings[i]) <= pad:
        a = author_emebedings[i]
        b = [zeros for y in range(len(author_emebedings[i]), pad + 1)]
        author_emebedings[i] = np.array(np.vstack((author_emebedings[i], b)), dtype="float32")
        arrays.append(author_emebedings[i])

embedding_AEC = np.reshape(arrays, (-1, 768))
xt = np.array(arrays)
# author_emebedings.astype(np.float32)
# xt = np.stack(author_emebedings.ravel()).reshape(4232, 49, 768)
print(xt.shape)
xb = xt
xq = xt
gt = xt


# (xt, xb, xq, gt) = load_dataset(args.database, args.device)  # 加载向量数据.

# triplet.train_triplet(xt, None, None, None, None)
def forward_pass(net, xall, bs=128, device=None):
    if device is None:
        device = next(net.parameters()).device
    xl_net = []
    for i0 in range(0, xall.shape[0], bs):
        x = torch.from_numpy(xall[i0:i0 + bs])
        x = x.to(device)
        x = x.type(torch.float32)
        xl_net.append(net(x).data.cpu().numpy())

    return np.vstack(xl_net)


def triplet_optimize(xt, xv, positives, xq, gt, net, args):
    val_k = 2 * args.dout
    # margin = 0
    margin = 0
    lr_schedule = [float(x.rstrip().lstrip()) for x in args.lr_schedule.split(",")]
    assert args.epochs % len(lr_schedule) == 0
    lr_schedule = repeat(lr_schedule, args.epochs // len(lr_schedule))
    print("Lr schedule", lr_schedule)

    # N = positives.shape[0]
    triplets = np.array(io.load_as_json("/ddisk/lj/DBLP/data/V4/", "author_triplets_muti_cluster"), dtype=np.int32)
    N = len(triplets)
    acc = []
    # xt_var = torch.from_numpy(xt).to(args.device)
    # xt_var = xt_var.type(torch.float32)
    xt_var = xt
    qt = lambda x: x

    # prepare optimizer
    optimizer = optim.SGD(net.parameters(), lr_schedule[0], momentum=args.momentum)
    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    pdist = nn.PairwiseDistance(2)

    for epoch in range(args.epochs):
        # Update learning rate
        args.lr = lr_schedule[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

        t0 = time.time()

        net.eval()
        # training pass
        print("  Train")
        net.train()
        avg_triplet, avg_loss = 0, 0
        offending = idx_batch = 0

        # process dataset in a random order
        perm = np.random.permutation(N)

        t1 = time.time()

        for i0 in range(0, N, args.batch_size):
            i1 = min(i0 + args.batch_size, N)
            n = i1 - i0
            data_idx = perm[i0:i1]
            data = triplets[data_idx]
            # print(data[:, 0])
            # print(data[:, 0][:, 0])
            # print(data[:, 0][:, 1])
            ins, pos, neg = net(xt_var[data[:, 0][0]][data[:, 0][1]]), net(xt_var[data[:, 1]]), net(xt_var[data[:, 2]])
            pos, neg = qt(pos), qt(neg)

            # triplet loss
            per_point_loss = pdist(ins, pos) - pdist(ins, neg) + margin
            per_point_loss = F.relu(per_point_loss)
            loss_triplet = per_point_loss.mean()
            offending += torch.sum(per_point_loss.data > 0).item()

            # combined loss
            loss = loss_triplet

            # collect some stats
            avg_triplet += loss_triplet.data.item()
            avg_loss += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            idx_batch += 1

        avg_triplet /= idx_batch
        avg_loss /= idx_batch

        t2 = time.time()

        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            logs_val = validation_vanilla(net, xt, xv, xq, gt, A_A, args, val_k)
            net.train()

        t3 = time.time()

        print('epoch %d, times: [hn %.2f s epoch %.2f s val %.2f s]'
              ' lr = %f'
              ' loss = %g = %g, offending %d' % (
                  epoch, t1 - t0, t2 - t1, t3 - t2,
                  args.lr,
                  avg_loss, avg_triplet, offending
              ))


def triplet_optimize_muti(xt, xv, positives, xq, gt, net, args):
    val_k = 2 * args.dout
    # margin = 0
    margin = 0
    lr_schedule = [float(x.rstrip().lstrip()) for x in args.lr_schedule.split(",")]
    assert args.epochs % len(lr_schedule) == 0
    lr_schedule = repeat(lr_schedule, args.epochs // len(lr_schedule))
    print("Lr schedule", lr_schedule)

    # N = positives.shape[0]
    triplets = np.array(io.load_as_json("/ddisk/lj/DBLP/data/V4/", "author_triplets_muti_cluster"), dtype=np.int32)
    N = len(triplets)
    acc = []
    xt_var = torch.from_numpy(xt).to(args.device)  # .to(args.device)
    xt_var = xt_var.type(torch.float32)
    # xt_var = xt
    qt = lambda x: x

    # prepare optimizer
    optimizer = optim.SGD(net.parameters(), lr_schedule[0], momentum=args.momentum)
    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    pdist = nn.PairwiseDistance(2)

    for epoch in range(args.epochs):
        # Update learning rate
        args.lr = lr_schedule[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

        t0 = time.time()

        net.eval()
        # training pass
        print("  Train")
        net.train()
        avg_triplet, avg_loss = 0, 0
        offending = idx_batch = 0

        # process dataset in a random order
        perm = np.random.permutation(N)

        t1 = time.time()

        for i0 in range(0, N, args.batch_size):
            i1 = min(i0 + args.batch_size, N)
            n = i1 - i0
            data_idx = perm[i0:i1]
            data = triplets[data_idx]
            # print(data)
            # print(data[:, 0])
            # print(data[:, 0][:, 0])
            # print(data[:, 0][:, 1])
            # print(xt_var[data[:, 0][:, 0]].shape)
            # abc = xt_var[data[:, 0][:, 0]]
            # print(abc[:, 0][data[:, 0][:, 1]].shape)
            anchor = xt_var[data[:, 0][:, 0]][:, 0][data[:, 0][:, 1]]
            pos = xt_var[data[:, 1][:, 0]][:, 0][data[:, 1][:, 1]]
            neg = xt_var[data[:, 2][:, 0]][:, 0][data[:, 2][:, 1]]
            # anchor = xt_var[data[:, 0][:, 0]][:, data[:, 0][:, 1]]
            # pos = xt_var[data[:, 1][:, 0]][data[:, 1][:, 1]]
            # neg = xt_var[data[:, 2][:, 0]][data[:, 2][:, 1]]
            # ins, pos, neg = net(xt_var[data[:, 0][0]][data[:, 0][1]]), net(xt_var[data[:, 1]]), net(xt_var[data[:, 2]])
            ins, pos, neg = net(anchor), net(pos), net(neg)
            pos, neg = qt(pos), qt(neg)

            # triplet loss
            per_point_loss = pdist(ins, pos) - pdist(ins, neg) + margin
            per_point_loss = F.relu(per_point_loss)
            loss_triplet = per_point_loss.mean()
            offending += torch.sum(per_point_loss.data > 0).item()

            # combined loss
            loss = loss_triplet

            # collect some stats
            avg_triplet += loss_triplet.data.item()
            avg_loss += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            idx_batch += 1

        avg_triplet /= idx_batch
        avg_loss /= idx_batch

        t2 = time.time()

        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            # logs_val = validation_vanilla(net, xt, xv, xq, gt, A_A, args, val_k)
            net.train()

        t3 = time.time()

        print('epoch %d, times: [hn %.2f s epoch %.2f s val %.2f s]'
              ' lr = %f'
              ' loss = %g = %g, offending %d' % (
                  epoch, t1 - t0, t2 - t1, t3 - t2,
                  args.lr,
                  avg_loss, avg_triplet, offending
              ))


def train_triplet(xt, xb, xq, gt, args):
    print("computing training ground truth")
    # dim = xt.shape[1]
    dim = 768
    dint, dout = args.dint, args.dout
    print("len of embedding", xt.shape[0])
    print("build network")
    net = nn.Sequential(
        nn.Linear(in_features=dim, out_features=dint, bias=True),
        nn.BatchNorm1d(dint),
        nn.LeakyReLU(),
        nn.Linear(in_features=dint, out_features=dint, bias=True),
        nn.BatchNorm1d(dint),
        nn.LeakyReLU(),
        nn.Linear(in_features=dint, out_features=dout, bias=True),
        Normalize()
    )
    net.to(args.device)
    triplet_optimize_muti(xt, xb, [], xq, gt, net, args)

    # save dataset
    yb = forward_pass(net, embedding_AEC, 512)
    # a = forward_pass(net, embedding_AEC, 512)
    trans = forward_pass(net, embedding_docs, 512)
    #

    gt_low_path = "/ddisk/lj/DBLP/data/V4/dataset_associations/embedding/"
    io.save_as_json(gt_low_path, "AEN", yb)
    io.save_as_json(gt_low_path, "trans", trans)
    # get_nearestneighbors_partly(yq, yb, 100, args.device, bs=3 * 10 ** 5, needs_exact=True, path=gt_low_path)


def get_nearestneighbors_graph(xq, xb, graph, k, device, needs_exact=True, verbose=False):
    # assert device in ["cpu", "cuda"]
    start = time.time()
    I = [[] for i in range(len(graph))]
    for i, a in enumerate(A_A):
        author_index = np.flatnonzero(A_A[i])[0:k]
        a_list = [i for _ in range(k)]
        for j, a in enumerate(author_index):
            a_list[j] = a
        I[i] = a_list
    return np.array(I, dtype="float32")


def get_nearestneighbors_APA_graph(xq, xb, graph, k, device, needs_exact=True, verbose=False):
    # assert device in ["cpu", "cuda"]
    start = time.time()
    # I = [[] for i in range(len(graph))]
    I = list()
    for i, a in enumerate(A_A):
        author_index = np.flatnonzero(A_A[i])
        a_list = [i for _ in range(len(author_index))]
        for j, a in enumerate(author_index):
            a_list[j] = a
        I.append(a_list)
    # return np.array(I, dtype="float32")
    return I


def get_nearestneighbors_APA(xq, xb, graph, k, device, needs_exact=True, verbose=False):
    # assert device in ["cpu", "cuda"]
    embedding_docs = np.array(
        io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "V4_0.8_PAP_15core"))
    A_A = np.array(io.load_as_json("/ddisk/lj/DBLP/data/", "A_A"))

    start = time.time()
    I = [[] for i in range(len(draph))]
    for i, a in enumerate(A_A):
        author_index = np.flatnonzero(A_A[i])[0:k]
        a_list = [i for _ in range(k)]
        for j, a in enumerate(author_index):
            a_list[j] = a
        I[i] = a_list
    return np.array(I, dtype="float32")


def get_k_core_commity_PAP(k):
    dataset = sets.DataSet("aminer", type=None)
    data_path = os.path.join("/ddisk/lj/DBLP/data", "V4", "dataset_associations")
    dataset.load(data_path)
    # A_dd = dataset.ds.citations
    A_da = dataset.ds.associations
    A_ad = A_da.transpose()
    I = [[] for i in range(A_ad.shape[0])]

    for seed_paper in range(0, (A_ad.shape[0])):
        Tque = set()
        Qque = set()
        N = set()
        S = set()
        vis = A_da.shape[0] * [0]
        # vis = [0 for i in range(A_da.shape[0])]
        # print(len(vis))
        Phi = [set() for i in range(A_ad.shape[0])]

        Tque.add(seed_paper)
        S.add(seed_paper)
        vis[seed_paper] = 1

        while len(Tque) > 0:
            doc = Tque.pop()
            for author in np.nonzero(A_ad[doc].toarray()[0])[0]:
                for a_doc in np.nonzero(A_da[author].toarray()[0])[0]:
                    if vis[a_doc] == 0:
                        vis[a_doc] = 1
                        Phi[doc].add(a_doc)
                        S.add(a_doc)

            if len(Phi[doc]) >= k:
                Tque = Tque.union(Phi[doc])
            else:
                Qque.add(doc)
        near_negative = []
        while len(Qque) != 0:
            v = Qque.pop()
            near_negative.append(v)
            S.remove(v)
            for u in Phi[v].copy():
                if (u not in S): continue
                if (v in Phi[u]): Phi[u].remove(v)
                if len(Phi[u]) < k:
                    Qque.add(u)

        for author in np.nonzero(A_ad[seed_paper].toarray()[0])[0]:
            for a_doc in np.nonzero(A_da[author].toarray()[0])[0]:
                N.add(a_doc)
        near_negative = set(near_negative) - N
        commity = S.union(N)
        I[seed_paper] = list(commity)
    io.save_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/", "author_community", I)
    return I
    # return commity


def get_nearestneighbors_negative(xq, xb, gt_nn, graph, k, device, needs_exact=True, verbose=False):
    I = [[] for i in range(len(graph))]
    all = range(len(graph))
    for i, a in enumerate(A_A):
        # author_index = np.flatnonzero(A_A[i])[0:k]
        pos = gt_nn[i]
        negative = list(set(all) - set(pos))
        a_list = [i for _ in range(k)]
        for j in range(0, k):
            a_list[j] = np.random.choice(negative)
        I[i] = a_list
    return np.array(I, dtype="float32")


def build_author_triplets():
    embedding_docs = np.array(
        io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "V4_0.8_PAP_15core"))
    A_A = np.array(io.load_as_json("/ddisk/lj/DBLP/data/", "A_A"))
    # I = [[] for i in range(len(A_A))]
    triplets = list()
    I = np.array(io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/", "author_community"))
    # I = get_k_core_commity_PAP(16)
    all = range(0, len(A_A))
    for i in all:
        negative = list(set(all) - set(I[i]))
        for pos in I[i]:
            neg = np.random.choice(negative)
            triplets.append([i, pos, neg])
            # print([i, pos, neg])
    print("triples sample num: ", len(triplets))

    io.save_as_json("/ddisk/lj/DBLP/data/V4/", "author_triplets_cluster", triplets)


def build_author_document_triplets():
    embedding_docs = np.array(
        io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "V4_0.8_PAP_15core"))
    A_A = np.array(io.load_as_json("/ddisk/lj/DBLP/data/", "A_A"))
    N = len(embedding_docs)

    triplets = list()
    I = np.array(io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/", "author_community"))
    all = range(0, len(A_A))
    for i in all:
        negative = list(set(all) - set(I[i]))
        for pos in I[i]:
            neg = np.random.choice(negative)
            triplets.append([i + N, pos + N, neg + N])

    dataset = sets.DataSet("aminer")
    data_path = os.path.join("/ddisk/lj/DBLP/data/", "V4", "dataset_associations")
    dataset.load(data_path)
    A_da = dataset.ds.associations.toarray()
    A_ad = A_da.T

    d_all = range(0, N)
    for i, author in enumerate(A_ad):
        documents_index = np.flatnonzero(A_ad[i])
        negative = list(set(d_all) - set(documents_index))
        for pos in documents_index:
            neg = np.random.choice(negative)
            triplets.append([i + N, pos, neg])

    for i in range(len(A_da)):
        author_index = np.flatnonzero(A_da[i])
        for author in author_index:
            doc_index = np.flatnonzero(A_ad[author])
            negative = list(set(d_all) - set(doc_index))
            for doc in doc_index[0:5]:
                neg = np.random.choice(negative)
                triplets.append([i, doc, neg])

    print("triples sample num: ", len(triplets))
    io.save_as_json("/ddisk/lj/DBLP/data/V4/", "author_document_triplets", triplets)


def build_author_document_triplets_cluster():
    embedding_docs = np.array(
        io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "V4_0.8_PAP_15core"))
    A_A = np.array(io.load_as_json("/ddisk/lj/DBLP/data/", "A_A"))
    N = len(embedding_docs)

    triplets = list()
    I = np.array(io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/", "author_community"))
    all = range(0, len(A_A))
    for i in all:
        negative = list(set(all) - set(I[i]))
        for pos in I[i]:
            neg = np.random.choice(negative)
            triplets.append([i + N, pos + N, neg + N])

    dataset = sets.DataSet("aminer")
    data_path = os.path.join("/ddisk/lj/DBLP/data/", "V4", "dataset_associations")
    dataset.load(data_path)
    A_da = dataset.ds.associations.toarray()
    A_ad = A_da.T

    d_all = range(0, N)
    for i, author in enumerate(A_ad):
        documents_index = np.flatnonzero(A_ad[i])
        negative = list(set(d_all) - set(documents_index))
        for pos in documents_index:
            neg = np.random.choice(negative)
            triplets.append([i + N, pos, neg])

    for i in range(len(A_da)):
        author_index = np.flatnonzero(A_da[i])
        for author in author_index:
            doc_index = np.flatnonzero(A_ad[author])
            negative = list(set(d_all) - set(doc_index))
            for doc in doc_index[0:5]:
                neg = np.random.choice(negative)
                triplets.append([i, doc, neg])

    print("triples sample num: ", len(triplets))
    # io.save_as_json("/ddisk/lj/DBLP/data/V4/", "author_document_triplets_cluster", triplets)


def build_author_triplets_cluster():
    embedding_docs = np.array(
        io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "V4_0.8_PAP_15core"))

    A_A = np.array(io.load_as_json("/ddisk/lj/DBLP/data/", "A_A"))
    # I = [[] for i in range(len(A_A))]
    triplets = list()
    I = np.array(io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/", "author_community"))
    documents_cluster = np.array(
        io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "AECD"))
    N = len(embedding_docs)
    all = range(0, len(A_A))
    for i in range(0, len(I)):
        documents_cluster_1 = documents_cluster[i * 2]
        documents_cluster_2 = documents_cluster[i * 2 + 1]
        negative = list(set(all) - set(I[i]) - set(documents_cluster_1) - set(documents_cluster_2))
        for doc in documents_cluster_1:
            neg = np.random.choice(negative)
            triplets.append([2 * i + N, doc, neg])
        for doc in documents_cluster_2:
            neg = np.random.choice(negative)
            triplets.append([2 * i + 1 + N, doc, neg])

        for pos in I[i]:
            # if pos == i: continue
            a_11 = author_emebedings[2 * i + N].dot(author_emebedings[pos * 2 + N].T)
            a_12 = author_emebedings[2 * i + N].dot(author_emebedings[pos * 2 + 1 + N].T)
            a_21 = author_emebedings[2 * i + 1 + N].dot(author_emebedings[pos * 2 + N].T)
            a_22 = author_emebedings[2 * i + 1 + N].dot(author_emebedings[pos * 2 + 1 + N].T)
            neg = np.random.choice(negative)
            if a_11 >= a_12:  # a_1 距离  pos_1  更近
                triplets.append([2 * i + N, pos * 2 + N, neg * 2 + N])
            else:
                triplets.append([2 * i + N, pos * 2 + 1 + N, neg * 2 + N])
            # triplets.append([2 * i + 1, pos * 2 + 1, neg * 2 + 1])
            neg = np.random.choice(negative)
            if a_21 >= a_22:  # a_1 距离  pos_1  更近
                triplets.append([2 * i + 1 + N, pos * 2 + N, neg * 2 + 1 + N])
            else:
                triplets.append([2 * i + 1 + N, pos * 2 + 1 + N, neg * 2 + 1 + N])

    print("triples sample num: ", len(triplets))
    io.save_as_json("/ddisk/lj/DBLP/data/V4/", "author_triplets_cluster", triplets)


def build_author_triplets_muti_cluster():
    """
     层次聚类. only the most similarity vector with author embedding.
    :return:
    """
    A_A = np.array(io.load_as_json("/ddisk/lj/DBLP/data/", "A_A"))
    auth_embeddings = np.array(
        io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "AECM"))

    embedding_docs = np.array(
        io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "V4_0.8_PAP_15core"))
    I = np.array(io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/", "author_community"))
    documents_cluster = np.array(
        io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "AECDM"))
    N = len(embedding_docs)
    all = range(0, len(A_A))

    triplets = list()
    for i in range(0, len(I)):  # current author
        negative = list(set(all) - set(I[i]))
        current_author_embeddings = np.array(auth_embeddings[i])
        for idx, current_author_cluster in enumerate(current_author_embeddings):
            for pos in I[i]:
                pos_author_embeddings = np.array(auth_embeddings[pos])
                similarity = 0
                index = 0
                for jdx, pos_author_cluster in enumerate(pos_author_embeddings):
                    sim = current_author_cluster.dot(pos_author_cluster.T)
                    if sim >= similarity:
                        similarity = sim
                        index = jdx
                neg = np.random.choice(negative)
                # neg_idx = np.random.choice(range(0, len(author_emebedings[neg])))
                triplets.append([[i, idx], [pos, index], [neg, 0]])

    print("triples sample num: ", len(triplets))
    io.save_as_json("/ddisk/lj/DBLP/data/V4/", "author_triplets_muti_cluster", triplets)


def build_author_triplets_muti_cluster_mutivec():
    """
    层次聚类. 10%100 vector with author embedding.
    :return:
    """
    A_A = np.array(io.load_as_json("/ddisk/lj/DBLP/data/", "A_A"))
    auth_embeddings = np.array(
        io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "AECM"))

    embedding_docs = np.array(
        io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "V4_0.8_PAP_15core"))
    I = np.array(io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/", "author_community"))
    documents_cluster = np.array(
        io.load_as_json("/ddisk/lj/DBLP/data/V4/dataset_associations/embedding", "AECDM"))
    # N = len(embedding_docs)
    all = range(0, len(A_A))

    triplets = list()
    for i in range(0, len(I)):  # current author
        negative = list(set(all) - set(I[i]))
        current_author_embeddings = np.array(auth_embeddings[i])
        for idx, current_author_cluster in enumerate(current_author_embeddings):
            for pos in I[i]:
                pos_author_embeddings = np.array(auth_embeddings[pos])
                first = 0
                second = 0
                f_index = 0
                s_index = 0
                # if len(pos_author_embeddings) < 1:
                #     print("pos", pos)

                for jdx, pos_author_cluster in enumerate(pos_author_embeddings):
                    sim = current_author_cluster.dot(pos_author_cluster.T)
                    if sim >= first:
                        second = first
                        s_index = f_index
                        first = sim
                        f_index = jdx
                    elif sim >= second:
                        second = sim
                        s_index = f_index

                neg1 = np.random.choice(negative)
                neg2 = np.random.choice(negative)

                triplets.append([[i, idx], [pos, f_index], [neg1, 0]])
                triplets.append([[i, idx], [pos, s_index], [neg2, 0]])

    print("triples sample num: ", len(triplets))
    io.save_as_json("/ddisk/lj/DBLP/data/V4/", "author_triplets_muti_cluster", triplets)


build_author_triplets_muti_cluster_mutivec()
# build_author_triplets_muti_cluster()
# build_author_triplets()
# build_author_document_triplets()
train_triplet(xt, xb, xq, gt, args)
# get_k_core_commity_PAP(16)


# dataset = sets.DataSet("aminer")
# data_path = os.path.join("/ddisk/lj/DBLP/data/", "V4", "dataset_associations")
# dataset.load(data_path)
# work_dir = "/ddisk/lj/DBLP/data/"
# version = "V4"
# data_type = "dataset_associations"
#
# save_dir = os.path.join(work_dir, version, data_type, "embedding")
# model_name = version + "_" + str(0.5) + "_" + "PAP" + "_" + str(15) + "core"
# embeddings = io.load_as_json(save_dir, model_name)
