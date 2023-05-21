"""
Algorithms to find the reference point

Adapted from [Highly Scalable and Provably Accurate Classification in Poincare Balls](https://arxiv.org/pdf/2109.03781.pdf)
with speed hack 
"""

# load packages 
import numpy as np

# constants 
EPS = 1e-15

# ======= utils ==========
def log_map(p: np.ndarray, X: np.ndarray) -> np.ndarray:
    """ vectorized log map """
    # -p \oplus x
    neg_p = - p 
    dot_prod = X @ neg_p
    X_norm_squared = np.linalg.norm(X, axis=-1, keepdims=True) ** 2 
    p_norm_squared = np.linalg.norm(p) ** 2

    num = (1 + 2 * dot_prod + X_norm_squared) * neg_p.T + (1 - p_norm_squared) * X 
    denom = (1 + 2 * dot_prod + X_norm_squared * p_norm_squared)

    mobius_result = num / denom

    # log map
    mobius_norm = np.linalg.norm(mobius_result, axis=-1, keepdims=True) 
    result = (1 - p_norm_squared) * np.arctanh(mobius_norm) * mobius_result / (mobius_norm + EPS)
    return result 

def mobius_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dot_prod = x.T @ y
    x_norm_squared = np.linalg.norm(x) ** 2
    y_norm_squared = np.linalg.norm(y) ** 2

    num = (1 + 2 * dot_prod + y_norm_squared) * x + (1 - x_norm_squared) * y 
    denom = (1 + 2 * dot_prod + x_norm_squared * y_norm_squared)

    mobius_result = num / denom 
    return mobius_result

# ======= algos =========
def convex_hull(X: np.ndarray):
    """ find convex hull on the Poincare Ball"""
    assert X.shape[-1] == 2, "convex hull search only supports 2D inputs for now"
    # find p0 
    p0_idx, p0 = min(zip(range(len(X)), X), key=lambda x: (x[1][1], x[1][0])) # by y then by x
    p0 = np.array([p0]).T  # cast to 2D

    # sort by inner product angle with p0 
    log_X = log_map(p0, X) 
    iplist = log_X @ np.array([[1], [0]]) / (np.linalg.norm(log_X, axis=-1, keepdims=True) + EPS)
    iplist[p0_idx, 0] = -np.inf
    ipidx = np.flip(np.argsort(iplist, axis=0))

    points = X[ipidx]

    # find hull
    n, d = X.shape
    stack = np.zeros((n+1, d))
    end_idx = 0
    stack[0] = p0.T

    for point in points:
        # print(point.shape)
        while (end_idx > 0) and (ccw(stack[[end_idx - 1]].T, stack[[end_idx]].T, point.T) < 0):
            end_idx -= 1
            
        end_idx += 1
        stack[end_idx] = point
        
    return stack[:(end_idx + 1)]


def ccw(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """ outer product in hyperbolic graham search """
    v01 = log_map(p0, p1.T).flatten()
    v12 = log_map(p0, p2.T).flatten()
    return (v01[0] * v12[1] - v01[1] * v12[0]).item()


def min_d_pair(ch1: np.ndarray, ch2: np.ndarray):
    """
    Finding minimum distance pair for convex hull CH1 and CH2 in Poincare disk.
    """
    N1 = np.shape(ch1)[0]
    N2 = np.shape(ch2)[0]
    cur_min_d = np.inf
    output = np.zeros((2, 2))
    for n1 in range(N1):
        for n2 in range(N2):
            # poincare distance 
            dist = np.arccosh(1 + 2 * (np.linalg.norm(ch1[n1] - ch2[n2]) ** 2 / ((1 - np.linalg.norm(ch1[n1]) ** 2) * (1 - np.linalg.norm(ch2[n2]) ** 2))))
            if dist < cur_min_d:
                output[:, [0]] = ch1[[n1]].T
                output[:, [1]] = ch2[[n2]].T
                cur_min_d = dist
    return output

def weighted_midpoint(c1: np.ndarray, c2: np.ndarray, t: float) -> np.ndarray:
    """
    Compute the weighted midpoint from C1 to C2 in Poincare ball. t is the time where t=0 we get C1 and t=1 we get C2.
    """
    v = mobius_add(-c1, c2)
    v = np.tanh(t * np.arctanh(np.linalg.norm(v))) * v / np.linalg.norm(v)
    return mobius_add(c1, v)
