from __future__ import division
import time
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from support_func import loss_permutation, loss_top_1_in_lat_top_k, forward_pass, Normalize,\
    repeat, get_nearestneighbors_partly, save_transformed_data, get_nearestneighbors


# def validation_vanilla(net, xt, xv, xq, gt, args, val_k):
#     logs = {}
#     net.eval()
#     yt = forward_pass(net, xt, 1024)
#     yv = forward_pass(net, xv, 1024)
#     logs['perm'] = loss_permutation(xt, yt, args, k=val_k, size=10**4)
#
#     logs['train_top1_k'] = loss_top_1_in_lat_top_k(xt, xt, yt, yt, args, 2, val_k, size=10**5, name="TRAIN")
#     logs['valid_top1_k'] = loss_top_1_in_lat_top_k(xv, xt, yv, yt, args, 1, val_k, size=10**5, name="VALID")
#
#     yq = forward_pass(net, xq, 1024)
#     logs['query_top1_k'] = loss_top_1_in_lat_top_k(xq, xt, yq, yt, args, 1, val_k, size=10**4, name="QUERY_tr")
#     logs['query_top1_2k'] = loss_top_1_in_lat_top_k(xq, xt, yq, yt, args, 1, 2*val_k, size=10**4, name="QUERY_tr")
#
#     net.train()
#
#     return logs


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def get_graph_dot_prod(graph_cur, xt_var):
    graph_directions = graph_cur - xt_var.reshape(graph_cur.shape[0], 1, graph_cur.shape[-1])
    graph_directions = graph_directions / (graph_directions.norm(dim=-1, keepdim=True) + 0.00001)
    dot_prod = torch.bmm(graph_directions, graph_directions.permute(0, 2, 1))

    return dot_prod


def angular_loss(graph, xt_var, net, bn, args):
    x_dot_prod = get_graph_dot_prod(graph, xt_var)

    graph_cur_y = net(graph.reshape(-1, graph.shape[-1])).reshape(bn, -1, args.dout)
    yt_var_cur = net(xt_var)
    y_dot_prod = get_graph_dot_prod(graph_cur_y, yt_var_cur)

    return ((x_dot_prod - y_dot_prod) ** 2).mean()


def angular_optimize(xt, xv, gt_nn, xq, gt, net, args):
    val_k = 2 * args.dout
    margin = 0.1
    rgs = 25
    graph = np.random.randint(0, xt.shape[0], (xt.shape[0], rgs))

    lr_schedule = [float(x.rstrip().lstrip()) for x in args.lr_schedule.split(",")]
    assert args.epochs % len(lr_schedule) == 0
    lr_schedule = repeat(lr_schedule, args.epochs // len(lr_schedule))
    print("Lr schedule", lr_schedule)

    N = gt_nn.shape[0]
    acc = []
    xt_var = torch.from_numpy(xt).to(args.device)

    qt = lambda x: x

    # prepare optimizer
    optimizer = optim.SGD(net.parameters(), lr_schedule[0], momentum=args.momentum)
    pdist = nn.PairwiseDistance(2)

    for epoch in range(args.epochs):
        # Update learning rate
        args.lr = lr_schedule[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

        t0 = time.time()

        # Sample positives for triplet
        rank_pos = np.random.choice(args.rank_positive, size=N)
        positive_idx = gt_nn[np.arange(N), rank_pos]

        # Sample negatives for triplet
        net.eval()
        print("  Forward pass")
        xl_net = forward_pass(net, xt, 1024)
        print("  Distances")

        I = get_nearestneighbors(xl_net, xl_net, args.rank_negative, args.device, needs_exact=False)
        
        negative_idx = I[:, -1]

        # training pass
        print("  Train")
        net.train()
        avg_ang, avg_triplet, avg_uniform, avg_loss = 0, 0, 0, 0
        offending = idx_batch = 0

        # process dataset in a random order
        perm = np.random.permutation(N)

        t1 = time.time()

        for i0 in range(0, N, args.batch_size):
            i1 = min(i0 + args.batch_size, N)
            n = i1 - i0
            data_idx = perm[i0:i1]

            # anchor, positives, negatives
            ins = xt_var[data_idx]
            pos = xt_var[positive_idx[data_idx]]
            neg = xt_var[negative_idx[data_idx]]

            # do the forward pass (+ record gradients)
            ins, pos, neg = net(ins), net(pos), net(neg)
            pos, neg = qt(pos), qt(neg)

            # triplet loss
            per_point_loss = pdist(ins, pos) - pdist(ins, neg) + margin
            per_point_loss = F.relu(per_point_loss)
            loss_triplet = per_point_loss.mean()
            offending += torch.sum(per_point_loss.data > 0).item()

            # angular loss
            graph_cur = xt_var[graph[data_idx]]
            loss_ang = angular_loss(graph_cur,  xt_var[data_idx], net, n, args)

            # uniform
            loss_uniform = uniform_loss(ins)

            # combined loss
            loss = loss_triplet + args.lambda_uniform * loss_uniform + args.lambda_angular * loss_ang

            # collect some stats
            avg_triplet += loss_triplet.data.item()
            avg_uniform += loss_uniform.data.item()
            avg_loss += loss.data.item()
            avg_ang += loss_ang.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            idx_batch += 1

        avg_triplet /= idx_batch
        avg_uniform /= idx_batch
        avg_loss /= idx_batch
        avg_ang /= idx_batch

        t2 = time.time()

        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            logs_val = validation_vanilla(net, xt, xv, xq, gt, args, val_k)
            net.train()

        t3 = time.time()

        print ('epoch %d, times: [hn %.2f s epoch %.2f s val %.2f s]'
               ' lr = %f'
               ' loss = %g = %g + lam * %g + ang * %g, offending %d' % (
            epoch, t1 - t0, t2 - t1, t3 - t2,
            args.lr,
            avg_loss, avg_triplet, avg_uniform, avg_ang, offending
        ))


def train_angular(xt, xb, xq, gt, args):    
    print ("computing training ground truth")
    xt_gt = get_nearestneighbors(xt, xt, args.rank_positive, device=args.device)


    dim = xt.shape[1]
    dint, dout = args.dint, args.dout

    print ("build network")
    net = nn.Sequential(
        nn.Linear(in_features=dim, out_features=dint, bias=True),
        nn.BatchNorm1d(dint),
        nn.ReLU(),
        nn.Linear(in_features=dint, out_features=dint, bias=True),
        nn.BatchNorm1d(dint),
        nn.ReLU(),
        nn.Linear(in_features=dint, out_features=dout, bias=True),
        Normalize()
    )
    net.to(args.device)


    angular_optimize(xt, xb, xt_gt, xq, gt, net, args)


    # save dataset
    yb = forward_pass(net, xb, 1024)
    yq = forward_pass(net, xq, 1024)

    net_style = "angular"

    gt_low_path = "/home/zjlab/ANNS/yq/paper/BREWESS/results/data/" + args.database + "/"\
                    + args.database + "_groundtruth_" + net_style + ".ivecs"
    get_nearestneighbors_partly(yq, yb, 100, args.device, bs=3*10**5, needs_exact=True, path=gt_low_path)

    save_transformed_data(xb, net, args.database + "/" + args.database + "_base_" + net_style + ".fvecs",
                            args.device)
    save_transformed_data(xq, net, args.database + "/" + args.database + "_query_" + net_style + ".fvecs",
                            args.device)
