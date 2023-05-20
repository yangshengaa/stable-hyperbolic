"""
lift trees to hyperbolic space and directly optimize the embedding to minimize distortion
"""

# load packages 
import os
import time 
import argparse
from functools import partial

import numpy as np

import torch
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter

from geoopt.optim import RiemannianAdam, RiemannianSGD

# load files 
from utils import visualize_embeddings, convert_fig_to_array
from obj import ae_pairwise_dist_objective, _euclidean_pairwise_dist, _poincare_pairwise_dist, _lorentz_pairwise_dist, metric_report
from models import architectures
from utils import load_config

torch.backends.cudnn.benchmark = True

# force 64 
dtype = torch.float64
torch.set_default_dtype(dtype)

# =====================
# ------ arguments ----
# =====================

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

# parameters
parser.add_argument('--c',          default=1,                  type=int,   help='curvature of the poincareball')
parser.add_argument('--thr',        default=1.-1e-16,           type=float, help='the hard boundary for hyperbolic numerical stability')

# models and loss func 
parser.add_argument('--data',       default='sim_tree_10121',   type=str,   help='the idx of dataset to read')
parser.add_argument('--model',      default='TreePoincare',     type=str,   help='model type')
parser.add_argument('--obj',        default='scaled',           type=str,   help='objective function')
parser.add_argument("--lift-type",  default='expmap',           type=str,   help='the initial lifting type of the hyperbolic models')

# hyperparameters 
parser.add_argument('--epochs',     default=3000,               type=int,   help='number of epochs')
parser.add_argument('--lr',         default=1e-2,               type=float, help='learning rate')
parser.add_argument('--opt',        default='riemannian_adam',  type=str,   help='type of optimizer')
parser.add_argument('--nesterov',   default=False,              action='store_true', help='use nestorov acceleration')
parser.add_argument('--momentum',   default=0.,                 type=float, help='the rate of momentum')

parser.add_argument('--log-train',  default=False,              action='store_true', help='store training loss')
parser.add_argument('--log-train-epochs', default=20,           type=int,   help='number of epochs to save loss')
parser.add_argument('--log-gradient', default=False,            action='store_true', help='True to log riemannian gradient norm at certain epoch')
parser.add_argument('--no-report-print', default=False, action='store_true',help='whether to print report')
parser.add_argument('--report-freq',default=100,                type=int,   help='number of epochs to report metrics')

# technical 
parser.add_argument('--no-gpu',     default=False,  action='store_true',    help='whether to disable gpu')
parser.add_argument('--tag',        default='tree',             type=str,   help='the tag for tree experiments')

args = parser.parse_args()

# load config 
paths = load_config(tag=args.tag)

test_name = f'{args.data}_{args.model}_{args.obj}_{args.epochs}_{args.lr}_{args.opt}_{args.nesterov}_{args.momentum}'
if args.log_train:
    tb_writer = SummaryWriter(log_dir=f'{paths["result_dir"]}/{test_name}')

# log gradient 
if args.log_gradient:
    gradient_path = os.path.join(paths['result_dir'], test_name, 'gradient')
    if not os.path.exists(gradient_path):
        os.mkdir(gradient_path)

# =====================
# ---- load data ------
# =====================

device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')

dataset_path = os.path.join(paths['data_dir'], args.data)
with open(os.path.join(dataset_path, 'sim_tree_points.npy'), 'rb') as f:
    node_positions = np.load(f)
with open(os.path.join(dataset_path, 'sim_tree_dist_mat.npy'), 'rb') as f:
    dist_mat = np.load(f)
with open(os.path.join(dataset_path, 'sim_tree_edges.npy'), 'rb') as f:
    edges = np.load(f)

node_positions = torch.as_tensor(node_positions, dtype=torch.float64, device=device)
dist_mat = torch.as_tensor(dist_mat, dtype=torch.float64, device=device)

# =====================
# --select parameters--
# =====================

# select manifold 
c = torch.tensor(args.c, requires_grad=False, dtype=dtype, device=device)
thr = torch.tensor(args.thr, requires_grad=False, dtype=dtype, device=device)

# select model 
model = getattr(architectures, args.model)(args.lift_type, node_positions, c=c, thr=thr).to(device)

# select dist function 
if args.model == 'TreeEuclidean':
    dist_func = _euclidean_pairwise_dist
elif args.model == 'TreePoincare' or args.model == 'TreeEuclideanLiftPoincare':
    dist_func = partial(_poincare_pairwise_dist, c=c, thr=thr)
elif args.model == 'TreeLorentz' or args.model == 'TreeEuclideanLiftLorentz':
    dist_func = partial(_lorentz_pairwise_dist, c=c, thr=thr)
else:
    raise NotImplementedError(f'model {args.model} not supported')

# hyperparameters 
epochs = args.epochs
lr = args.lr

if args.opt == 'adam':
    opt_type = Adam 
    opt = opt_type(model.parameters(), lr=lr, amsgrad=True , betas=(0.9, 0.999), weight_decay=0)
elif args.opt == 'riemannian_adam':
    opt_type = RiemannianAdam
    opt = opt_type(model.parameters(), lr=lr, amsgrad=True, betas=(0.9, 0.999), weight_decay=0)
elif args.opt == 'sgd':
    opt_type = SGD
    opt = opt_type(model.parameters(), lr=lr, nesterov=args.nesterov, momentum=args.momentum)
elif args.opt == 'riemannian_sgd':
    opt_type = RiemannianSGD
    opt = opt_type(model.parameters(), lr=lr, nesterov=args.nesterov, momentum=args.momentum)
else:
    raise NotImplementedError(f'opt {args.opt} not supported')

# append extra parameter
if args.obj == 'learning_relative':
    alpha = Parameter(torch.tensor(1., dtype=dtype, device=device))  # initialize to 1 
    model.learning_alpha = alpha
    opt.add_param_group({'params': alpha})

# =====================
# ----- training ------ 
# =====================
def train():
    loss = None 

    for epoch in range(epochs):
        opt.zero_grad()
        reconstructed_data = model()
        (
            loss, 
            (emb_dists_selected, real_dists_selected, emb_dists) 
        ) = ae_pairwise_dist_objective(model, reconstructed_data, dist_mat, dist_func, args.obj, c=c, thr=thr)
        loss.backward()

        # print message 
        if (epoch % args.report_freq == 0) or (epoch % args.log_train_epochs == 0 and args.log_train): 
            with torch.no_grad():
                ( 
                    distortion, individual_distortion, max_distortion,
                    relative_rate, scaled_rate,
                    contractions_std, expansions_std,
                    diameter 
                ) = metric_report(emb_dists_selected, real_dists_selected, emb_dists, dist_mat)

            if epoch % args.report_freq == 0 and not args.no_report_print:
                print(f'====> Epoch: {epoch:04d}, ' +
                    f'Loss: {loss:.4f}, Distortion: {distortion:.4f}, Idv Distortion: {individual_distortion:.4f}, Max Distortion {max_distortion:.2f}, ' +
                    f'relative: {relative_rate:.2f}, scaled: {scaled_rate:.2f}, ' +
                    f'Contraction Std {contractions_std:.4f}, Expansion Std {expansions_std:.6f}, Diameter {diameter:.2f}')
            
            # logging 
            if epoch % args.log_train_epochs == 0 and args.log_train:
                # vis 
                fig = visualize_embeddings(
                    reconstructed_data.detach().cpu().numpy(), edges, dist_mat, args.model, c=args.c, root_idx=0, thr=args.thr,
                    loss=loss, diameter=diameter, distortion=distortion, max_distortion=max_distortion
                )
                img_arr = convert_fig_to_array(fig)
                img_arr = torch.tensor(img_arr)

                # write to tensorboard 
                tb_writer.add_image('embedding/embedding_image', img_arr, epoch, dataformats='HWC')
                
                # add metrics 
                tb_writer.add_scalar('train/loss', loss, epoch)
                tb_writer.add_scalar('train/distortion', distortion, epoch)
                tb_writer.add_scalar('train/individual_distortion', individual_distortion, epoch)
                tb_writer.add_scalar('train/max_distortion', max_distortion, epoch)
                tb_writer.add_scalar('train/relative', relative_rate, epoch)
                tb_writer.add_scalar('train/scaled', scaled_rate, epoch)
                tb_writer.add_scalar('train/contraction_std', contractions_std, epoch)
                tb_writer.add_scalar('train/expansion_std', expansions_std, epoch)
                tb_writer.add_scalar('train/diameter', diameter, epoch)
                
                # flush to disk
                tb_writer.flush()

                # log gradient
                if args.log_gradient:
                    points = model.X.detach().clone()
                    gradients = model.X.grad.detach().clone()
                    rgrad_norm = model.get_rgrad_norm(points, gradients)
                    torch.save(
                        rgrad_norm, 
                        os.path.join(gradient_path, f'rgrad_norm_e{epoch}.pt')
                    )

        # update parameters
        opt.step()

    return loss, distortion, max_distortion, diameter
        
# =====================
# ----- logggin -------
# =====================

def log_info(time, loss, distortion, max_distortion, diameter, **kwargs):
    log_information = ','.join([str(x) for x in kwargs.values()]) + f',{time:.4f},{loss},{distortion},{max_distortion},{diameter}'
    with open(os.path.join(paths['result_dir'], 'time_record.csv'), 'a+') as f:
        f.write(log_information)
        f.write('\n') 

def log_train_flow():
    """ only log train info """
    train() 

def speed_test_flow():
    """ only speed test """
    # train 
    start = time.time()
    final_loss, distortion, max_distortion, diameter = train()
    end = time.time()

    duration = end - start 

    # log 
    log_info(
        duration, final_loss, distortion, max_distortion, diameter,
        
        data=args.data,
        model=args.model,
        obj=args.obj,
        opt=args.opt,

        c=args.c,

        epochs=args.epochs,
        lr=args.lr,

        nesterov=args.nesterov,
        momentum=args.momentum
    )

def main():
    if args.log_train:
        log_train_flow()
    else:
        speed_test_flow()

if __name__ == '__main__':
    main()
