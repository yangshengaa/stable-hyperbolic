"""
objective functions for simple experiments 
"""

# load packages 
import torch 

# eps 
EPS = 1e-15  # an appropriate lower bound for float64

# ==================== for trees ============================

# metics 
def _euclidean_pairwise_dist(data_mat):
    """ 
    compute the pairwise euclidean distance matrix

    :param data_mat: of N by D 
    :return dist_mat: of N by N 
    """
    dist_mat = torch.cdist(data_mat, data_mat, p=2, compute_mode="donot_use_mm_for_euclid_dist") # for accuracy
    return dist_mat

def _poincare_pairwise_dist(data_mat, c=1, thr=0.9999):
    """ compute pairwise hyperbolic distance """
    # hard threshold: https://doi.org/10.48550/arXiv.2107.11472
    data_mat_rescaled = data_mat * torch.clamp(thr / torch.linalg.norm(data_mat, dim=1, keepdim=True), max=1)
    data_mat_rescaled_norm = torch.linalg.norm(data_mat_rescaled, dim=1, keepdim=True)
    euclidean_dist_mat = _euclidean_pairwise_dist(data_mat_rescaled)
    # denom = (1 - c * data_mat_rescaled_norm ** 2) @ (1 - c * data_mat_rescaled_norm ** 2).T

    dist_mat = 1 / torch.sqrt(c) * torch.arccosh(
        1 + 
        (
            2 * c * euclidean_dist_mat ** 2 / (1 - c * data_mat_rescaled_norm ** 2) / (1 - c * data_mat_rescaled_norm ** 2).T
        ).clamp(min=EPS)  # * note that the gradient of arccosh could only be computed when input is larger than 1 + EPS for 64
    )
    dist_mat.fill_diagonal_(0.)  # force for diag 
    return dist_mat

def _lorentz_pairwise_dist(data_mat, c=torch.tensor(1.), thr=torch.tensor(0.9999)):
    """ compute pairwise lorentz distance """
    # TODO: thresholding 
    dim = data_mat.size(1) - 1
    x0 = data_mat.narrow(-1, 0, 1)
    xr = data_mat.narrow(-1, 1, dim)
    inner_neg = x0 @ x0.T - xr @ xr.T 
    dist_mat = c.sqrt() * torch.arccosh((inner_neg / c).clamp(min=1. + EPS))
    dist_mat.fill_diagonal_(0.)
    return dist_mat

def _approx_pairwise_dist(data_mat, c=torch.tensor(1.), thr=torch.tensor(0.9999)):
    """ 
    approximate hyperbolic distance 
    
    :param data_mat: euclidean features!
    """
    data_mat_norm = data_mat.norm(dim=-1, keepdim=True)
    theta_diff = data_mat @ data_mat.T / data_mat_norm.T / data_mat_norm
    theta = torch.pi - torch.abs(torch.pi - torch.abs(theta_diff))
    dist_mat = data_mat_norm + data_mat_norm.T + 2 / c.sqrt() * torch.log(torch.sin(theta))
    dist_mat.fill_diagonal_(0.)
    return dist_mat

def _diameter(emb_dists):
    """ compute the diameter of the embeddings """
    return torch.max(emb_dists)

# ===============================================
# -------------- loss functions -----------------
# ===============================================

def _select_upper_triangular(emb_dists, real_dists):
    """ select the upper triangular portion of the distance matrix """
    mask = torch.triu(torch.ones_like(real_dists), diagonal=1) > 0
    emb_dists_selected = torch.masked_select(emb_dists, mask)
    real_dists_selected = torch.masked_select(real_dists, mask)
    return emb_dists_selected, real_dists_selected

def _pairwise_dist_loss(emb_dists_selected, real_dists_selected):
    """ equally weighted pairwise loss """
    loss = torch.mean((emb_dists_selected - real_dists_selected) ** 2)
    return loss 

def _relative_pairwise_dist_loss(emb_dists_selected, real_dists_selected):
    """ 
    relative distance pairwise loss, given by 
    ((d_e - d_r) / d_r) ** 2
    """ 
    loss = torch.mean((emb_dists_selected / real_dists_selected - 1) ** 2)
    return loss 

def _relative_learning_pairwise_dist_loss(emb_dists_selected, real_dists_selected, alpha: torch.nn.parameter.Parameter):
    """ relative loss with an additional learning parameter """
    loss = torch.mean((alpha * emb_dists_selected / real_dists_selected - 1) ** 2)
    return loss 

def _scaled_pairwise_dist_loss(emb_dists_selected, real_dists_selected):
    """ 
    scaled version that is more presumably more compatible with optimizing distortion 
    (d_e / mean(d_e) - d_r / mean(d_r)) ** 2
    """
    loss = torch.mean(
        ((emb_dists_selected / emb_dists_selected.mean()) - (real_dists_selected / real_dists_selected.mean())) ** 2
    )
    return loss

def _robust_scaled_pairwise_dist_loss(emb_dists_selected, real_dists_selected):
    """ convert to cosh before computing the scaled loss """
    return _scaled_pairwise_dist_loss(torch.cosh(emb_dists_selected), torch.cosh(real_dists_selected))

def _distortion_loss(emb_dists_selected, real_dists_selected):
    """ directly use average distortion as the loss """
    loss = torch.mean(emb_dists_selected / (real_dists_selected)) * torch.mean(real_dists_selected / (emb_dists_selected))
    return loss

def _individual_distortion_loss(emb_dists, real_dists):
    """ 
    compute the following loss 
    
    average of the average distortion 
    """
    n = real_dists.shape[0]
    pairwise_contraction = real_dists / (emb_dists + 1e-5)
    pairwise_expansion = emb_dists / (real_dists + 1e-5)
    pairwise_contraction.fill_diagonal_(0)
    pairwise_expansion.fill_diagonal_(0)

    # print(torch.max(pairwise_contraction))
    # print(torch.max(pairwise_expansion))
    # TODO: truly remove diagonal
    # compute individual
    individual_pairwise_contraction = pairwise_contraction.sum(axis=1) / (n - 1)
    individual_pairwise_expansion = pairwise_expansion.sum(axis=1) / (n - 1)
    individual_distortion = individual_pairwise_contraction * individual_pairwise_expansion
    
    loss =  individual_distortion.mean()
    return loss

# distortion evaluations 
def _max_distortion_rate(contractions, expansions):
    """ compute max distortion rate """ 
    with torch.no_grad():
        contraction = torch.max(contractions)       # max 
        expansion = torch.max(expansions)           # max 
        distortion = contraction * expansion
        return distortion

def _distortion_rate(contractions, expansions):
    """ compute 'average' distortion rate """
    with torch.no_grad():
        contraction = torch.mean(contractions)      # mean 
        expansion = torch.mean(expansions)          # mean 
        distortion = contraction * expansion
        return distortion
    
def _individual_distortion_rate(emb_dists, real_dists):
    """ compute the average avarage distortion rate """
    with torch.no_grad():
        return _individual_distortion_loss(emb_dists, real_dists)


def ae_pairwise_dist_objective(model, reconstructed_data, shortest_path_mat, dist_func, loss_function_type='scaled', c=1, thr=0.9999, use_approx=False):
    """
    minimize regression MSE (equally weighted) on the estimated pairwise distance. The output distance is 
    either measured in Euclidean or in hyperbolic sense

    assume that the data comes in the original sequence (shuffle = False)

    :param c: the curvature, if use_hyperbolic is true 
    :param loss_function_type: raw for the normal one, relative for relative dist, scaled for scaled
    :param use_approx: use approximate dist
    """
    # compute 
    emb_dists = dist_func(reconstructed_data)
    
    # select upper triangular portion 
    emb_dists_selected, real_dists_selected = _select_upper_triangular(emb_dists, shortest_path_mat)

    if use_approx:
        emb_dists_approx = _approx_pairwise_dist(model.X)  # feed euclidean inputs
        emb_dsits_approx_selected, real_dists_approx_selected = _select_upper_triangular(emb_dists_approx[1:, 1:], shortest_path_mat[1:, 1:])
        # remove one row 

    # select loss function 
    if loss_function_type == 'raw':
        loss = _pairwise_dist_loss(emb_dists_selected, real_dists_selected)
    elif loss_function_type == 'relative':
        loss = _relative_pairwise_dist_loss(emb_dists_selected, real_dists_selected)
    elif loss_function_type == 'scaled':
        loss = _scaled_pairwise_dist_loss(emb_dists_selected, real_dists_selected)
    elif loss_function_type == 'robust_scaled':
        loss = _robust_scaled_pairwise_dist_loss(emb_dists_selected, real_dists_selected)
    elif loss_function_type == 'distortion':
        loss = _distortion_loss(emb_dists_selected, real_dists_selected)
    elif loss_function_type == 'individual_distortion':
        loss = _individual_distortion_loss(emb_dists, shortest_path_mat)
    elif loss_function_type == 'learning_relative':
        loss = _relative_learning_pairwise_dist_loss(emb_dists_selected, real_dists_selected, model.learning_alpha)
    else:
        raise NotImplementedError(f'loss function type {loss_function_type} not available')

    if use_approx:
        return loss, (emb_dists_selected, real_dists_selected, emb_dists, emb_dsits_approx_selected, real_dists_approx_selected, emb_dists_approx)
    else:
        return loss, (emb_dists_selected, real_dists_selected, emb_dists)


def metric_report(
        emb_dists_selected, real_dists_selected,
        emb_dists, real_dists,
        emb_dists_approx_selected=None, real_dists_approx_selected=None,
    ):
    """ report metric along training """
    with torch.no_grad():
        contractions = real_dists_selected / (emb_dists_selected)
        expansions = emb_dists_selected / (real_dists_selected)
        contractions_std = torch.std(contractions)
        expansions_std = torch.std(expansions)
        
        # all candidate loss 
        distortion_rate = _distortion_rate(contractions, expansions)
        max_distortion_rate = _max_distortion_rate(contractions, expansions)
        individual_distortion_rate = _individual_distortion_rate(emb_dists, real_dists)
        relative_rate = _relative_pairwise_dist_loss(emb_dists_selected, real_dists_selected)
        scaled_rate = _scaled_pairwise_dist_loss(emb_dists_selected, real_dists_selected)

        diameter = _diameter(emb_dists_selected)

        return (
            distortion_rate, individual_distortion_rate, max_distortion_rate, 
            relative_rate, scaled_rate, 
            contractions_std, expansions_std, 
            diameter
        )
