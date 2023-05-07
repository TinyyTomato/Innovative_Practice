from __future__ import division
import torch
import argparse
import numpy as np
import triplet, angular, brewess, catalyst
from support_func import sanitize, eval
from data import load_dataset, load_compress_dataset

######################################################
# Load parameters
######################################################

# nohup python -u main.py --database sift >> /home/zjlab/ANNS/yq/paper/_log/nang_sift_sample.out &
# nohup python -u main.py --database sift --method triplet --batch_size 512 --dout 16 --lambda_uniform 0.005 >> /home/zjlab/ANNS/yq/paper/_log/triplet_sift_sample.out &
# nohup python -u main.py --database sift --method angular --batch_size 512 --dout 16 --lambda_uniform 0.005 >> /home/zjlab/ANNS/yq/paper/_log/angular_sift_sample.out &
# nohup python -u main.py --database sift --method brewess --batch_size 512 --dout 16 --lambda_uniform 0.005 >> /home/zjlab/ANNS/yq/paper/_log/brewess_sift_sample.out &
# nohup python -u main.py --database sift --method catalyst --batch_size 512 --dout 16 --lambda_uniform 0.005 >> /home/zjlab/ANNS/yq/paper/_log/catalyst_sift_sample.out &


parser = argparse.ArgumentParser()

def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)

group = parser.add_argument_group('dataset options')
aa("--database", default="sift")  # can be "sift", "gist"

group = parser.add_argument_group('Model hyperparameters')
aa("--dout", type=int, default=32,
    help="output dimension")
aa("--dint", type=int, default=1024,
    help="size of hidden states")
aa("--method", type=str, default="triplet")  # can be "triplet" or "angular"
aa("--lambda_uniform", type=float, default=0.05,
    help="weight of the uniformity loss")
aa("--lambda_angular", type=float, default=0.05)

group = parser.add_argument_group('Training hyperparameters')
aa("--batch_size", type=int, default=64)
aa("--epochs", type=int, default=40)
aa("--momentum", type=float, default=0.9)
aa("--rank_positive", type=int, default=5,
    help="this number of vectors are considered positives")
aa("--rank_negative", type=int, default=10,
    help="these are considered negatives")

group = parser.add_argument_group('Computation params')
aa("--seed", type=int, default=1234)
aa("--device", choices=["cuda", "cpu", "auto"], default="auto")
aa("--lr_schedule", type=str, default="0.1,0.1,0.05,0.01")
aa("--val_freq", type=int, default=10,
    help="frequency of validation calls")

args = parser.parse_args()
if args.device == "auto":
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)


######################################################
# Train
######################################################


K = 10

# load dataset
print("load dataset %s" % args.database)
(xt, xb, xq, gt) = load_dataset(args.database, args.device)
xt = sanitize(xt)  # 训练集
xb = sanitize(xb)
xq = sanitize(xq)
print(xt.shape, xb.shape, xq.shape, gt.shape)

# # origin data test
# eval(xb, xq, gt, K)

#train
if args.method == "triplet":
    triplet.train_triplet(xt, xb, xq, gt, args)
elif args.method == "catalyst":
    catalyst.train_triplet(xt, xb, xq, gt, args)
elif args.method == "brewess":
    brewess.train_triplet(xt, xb, xq, gt, args)
else:
    angular.train_angular(xt, xb, xq, gt, args)
 
# compress data test
(xb, xq, gt) = load_compress_dataset(args.database, args.device)
xb = sanitize(xb)
xq = sanitize(xq)
print(xb.shape, xq.shape, gt.shape)
eval(xb, xq, gt, K)
