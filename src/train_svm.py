"""
train svm 
"""

# load packages 
import os 
import time 
import argparse
import warnings 
warnings.filterwarnings('ignore')

import numpy as np

# load files 
import models
from utils import load_config, metric_report
from methods import convex_hull, min_d_pair, weighted_midpoint, platt_train, platt_test, poincare_to_loretnz


# ========== arguments ===========
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

# model 
parser.add_argument("--model",          default="LSVMPP",       type=str,       help='SVM model', choices=["ESVM", "LSVM", "PSVM", "LSVMPP"])

# hyperparameters
parser.add_argument("--init-epochs",    default=3000,           type=int,       help="the initial training epochs, used for LSVM and LSVMPP")
parser.add_argument("--epochs",         default=5000,           type=int,       help='number of iterations to train')
parser.add_argument("--lr",             default=1e-3,           type=float,     help='the learning rate to learn')
parser.add_argument("--C",              default=1,              type=float,     help="tolerance of misclassification")
parser.add_argument("--penalty",        default="l2",           type=str,       help="the type of penalty",     choices=["l1", 'l2'])
parser.add_argument("--loss",           default="squared_hinge",type=str,       help="the loss type", choices=["hinge", "squared_hinge"])

# data 
parser.add_argument("--data",           default='cifar',        type=str,       help="the data name")
parser.add_argument("--refpt",          default="precompute",   type=str,       help="compute or train reference point", choices=['precompute', 'raw'])
parser.add_argument("--trails",         default=5,              type=int,       help="How many trails to run")

# calibration
parser.add_argument("--no-calibration", default=False,  action="store_true",    help="True to use platt scaling, False otherwise")
parser.add_argument("--cal-iter",       default=100,            type=int,       help="max iteration in calibration")
parser.add_argument("--min-step",       default=1e-10,          type=float,     help="minimum steps allowed in line search")
parser.add_argument("--sigma",          default=1e-12,          type=float,     help="for numerical PD")
parser.add_argument("--eps",            default=1e-5,           type=float,     help="the convergence guarantee")

# path
parser.add_argument("--tag",            default='svm',          type=str,       help="the config tag to specify paths")

# technical
parser.add_argument("--device",         default='cpu',          type=str,       help='whether to use cuda in LSVMPP', choices=['cuda', 'cpu'])

# model specific 
parser.add_argument("--decision-mode",  default='linear',       type=str,       help='the mode to give decisions', choices=['linear', 'arcsinh', 'arcsinh_norm'])

args = parser.parse_args()

logging_name = f"{args.model},{args.init_epochs},{args.epochs},{args.lr},{args.C},{args.penalty},{args.data},{args.no_calibration}"


# =========== model ===============
modelC = getattr(models, args.model)

# model specific parameters
if args.model == 'LSVMPP':
    other_args = {
        "device": args.device,
        "decision_mode": args.decision_mode
    }
else:
    other_args = {}

model = modelC(
    args.epochs, args.lr, args.C, args.penalty, args.loss,
    args.init_epochs, **other_args
)


# ============ data ===============
paths = load_config(args.tag)

data_dir = os.path.join(paths['data_dir'], args.data)
data = np.load(os.path.join(data_dir, f"{args.data}_poincare_embedding.npz"))

# =========== training =============
def binary_loop(
        train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray,
        p: np.ndarray=None, is_binary=False
    ) -> np.ndarray:
    """ 
    perform a binary classification once, and return the decision values

    :param p: the precomputed reference point, if any; None means use raw
    :param is_binary: True if is binary, False otherwise 
    :return array of 1s and -1s if no calibration, otherwise platt-calibrated probabilities
    """
    # projection 
    if 'LSVM' in args.model:
        train_X = poincare_to_loretnz(train_X)
        test_X = poincare_to_loretnz(test_X)
    
    # train model and make decisions
    model.train(train_X, train_y, p)
    train_decision_vals, train_decisions = model.decision(train_X, p)
    test_decision_vals, test_decisions = model.decision(test_X, p)
    
    # only binary if no calibration or binary classification
    if args.no_calibration or is_binary: 
        return test_decisions

    # platt scaling 
    else:
        ab = platt_train(
            train_decision_vals, train_y, prior0=None, prior1=None,
            max_iteration=args.cal_iter, min_step=args.min_step,
            sigma=args.sigma, eps=args.eps
        )
        test_decision_prob = platt_test(test_decision_vals, ab=ab)
        return test_decision_prob


def get_reference_points(train_X: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    """ 
    load precompute reference points or compute from raw 
    
    adapted from [Highly Scalable and Provably Accurate Classification in Poincare Balls](https://arxiv.org/pdf/2109.03781.pdf)
    """
    if args.model == 'PSVM':  # only PSVM would require a reference point 
        if args.refpt == 'precompute':
            print('Load precompute reference points')
            p = np.load(os.path.join(data_dir, f"{args.data}_reference_points_gt.npy"))
        elif args.refpt == 'raw':
            print('Compute raw')
            p_list = []
            for label in np.unique(train_y):
                # Find all training points with label
                X = train_X[train_y == label]
                # Find all training points for the rest labels
                X_rest = train_X[train_y != label]
                # Find convex hull for these two group of points
                ch_label = convex_hull(X)
                ch_rest = convex_hull(X_rest)
                # Find min dist pair on these two convex hull
                min_dist_pairs = min_d_pair(ch_label, ch_rest)
                # choose p as their mid point
                p_list.append(weighted_midpoint(min_dist_pairs[:, [0]], min_dist_pairs[:, [1]], 0.5))

            p = np.hstack(p_list).T
    else:
        p = np.array([None] * len(np.unique(train_y)))
    
    return p

def train_test():
    """ the train test loops """
    # timing 
    print("=========== start training ===========")
    start = time.time()

    # prepare 
    labels = np.unique(data['y_train'])
    multiclass = len(labels) > 2

    # find reference point if needed 
    p_arr = get_reference_points(data['x_train'], data["y_train"])

    # repeat for trails many times
    acc_arr, f1_arr = [], [] 
    for t in range(args.trails):
        
        # multiclass processing flow 
        if multiclass:
            test_decision_mat = []
            for i, label in enumerate(labels):
                train_binarized_y = ((data['y_train'] == label) * 2 - 1).astype(int)  # binarize labels
                test_decision_raw = binary_loop(data['x_train'], train_binarized_y, data["x_test"], p_arr[i], is_binary=False)
                test_decision_mat.append(test_decision_raw)
            
            test_decision_mat = np.vstack(test_decision_mat).T
        
        # binary classification flow
        else:
            train_binarized_y = ((data['y_train'] == 1) * 2 - 1).astype(int)  # binarized label
            test_decision_raw = binary_loop(data['x_train'], train_binarized_y, data["x_test"], p_arr[0], is_binary=True)
            test_decision_mat = (np.array([test_decision_raw]).T + 1) // 2  # convert to 0  
        
        # evaluation
        acc, f1 = metric_report(test_decision_mat, data["y_test"])
        acc_arr.append(acc)
        f1_arr.append(f1)
        print(f"====> run {t + 1}: acc {acc:.4f}, f1 {f1:.4f}")
    
    end = time.time()
    average_time = (end - start) / args.trails
    print("========== finish training ==========")
    print(f"using {average_time:.2f} seconds on average")

    # logging 
    average_acc = np.mean(acc_arr)
    average_f1 = np.mean(f1_arr)
    print(f"average acc: {average_acc:.4f}, average f1: {average_f1:.4f}")

    with open(os.path.join(paths['result_dir'], 'result.csv'), 'a+') as f:
        f.write(logging_name + ',')
        f.write(f"{np.mean(acc_arr)},{np.median(acc_arr)},{np.std(acc_arr)},")
        f.write(f"{np.median(f1_arr)},{np.median(f1_arr)},{np.std(f1_arr)},")
        f.write(f"{average_time}")
        f.write('\n')    


# ============= driver ==============
if __name__ == '__main__':
    train_test()
