"""
simple architectures 
"""

# load packages 
from functools import partial

import torch 
import torch.nn as nn
from torch.nn.parameter import Parameter

import geoopt 
from geoopt import ManifoldParameter
from geoopt.manifolds import Euclidean, PoincareBallExact, Lorentz

# load file 
from .manifold_layers import HyperbolicLorentz, HyperbolicLorentzPP, HyperbolicPoincare

# init params 
INIT_MEAN = 0
INIT_STD = 0.5
EPS = 1e-16

# lifting function
def expmap0(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x_norm = x.norm(dim=1, keepdim=True)
    c_sqrt = c.sqrt()
    mapped_x = torch.tanh(c_sqrt * x_norm) * x / (c_sqrt * x_norm + EPS)
    return mapped_x

def expmap0_lorentz(c: torch.Tensor, x: torch.tensor) -> torch.Tensor:
    x_norm = x.norm(dim=1, keepdim=True)
    x0 = torch.cosh(c.sqrt() * x_norm) * c.sqrt()
    xr = torch.sinh(c.sqrt() * x_norm) * x / (c.sqrt() * x_norm + EPS)
    mapped_x = torch.cat([x0, xr], dim=-1)
    return mapped_x

def sinh_direct_map(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """ insert sinh before direct map, suggested by Zhengchao """
    c_sqrt = c.sqrt()
    w = torch.sinh(c_sqrt * x) / c_sqrt
    mapped_x = direct_map(c, w)
    return mapped_x

def direct_map(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """ 
    a direct map to within the circle, suggested by Zhengchao 
    from arXiv:2006.08210, equation 7 
    """
    mapped_x = x / (1 + torch.sqrt(1 + c * torch.linalg.norm(x, dim=1, keepdim=True) ** 2))
    return mapped_x

def direct_map_lorentz(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    lift euclidean features to lorentz surface through a direct projection 
    """
    x0 = (x.norm(dim=1, keepdim=True) ** 2 + 1 / c).sqrt()
    lorentz_features = torch.cat([x0, x], dim=1)
    return lorentz_features


# architectures 
class ModelPoincare(nn.Module):
    def __init__(self, manifold, init_points: torch.Tensor) -> None:
        super().__init__()
        self.X = ManifoldParameter(init_points, manifold=manifold)
    
    def forward(self): return self.X


class ModelLorentz(nn.Module):
    def __init__(self, manifold, init_points: torch.Tensor) -> None:
        super().__init__()
        self.X = ManifoldParameter(init_points, manifold=manifold)
    
    def forward(self): return self.X


class ModelExpmap(nn.Module):
    def __init__(self, manifold, init_points: torch.Tensor) -> None:
        super().__init__()
        self.X = Parameter(init_points)
        self.c = manifold.c
    
    def forward(self):
        hyperbolic_feature = expmap0(self.c, self.X / 2)
        return hyperbolic_feature


class ModelDirect(nn.Module):
    def __init__(self, manifold, init_points: torch.Tensor) -> None:
        super().__init__()
        self.X = Parameter(init_points)
        self.c = manifold.c

    def forward(self):
        hyperbolic_feature = direct_map(self.c, self.X)
        return hyperbolic_feature


class ModelSinhDirect(nn.Module):
    def __init__(self, manifold, init_points: torch.Tensor) -> None:
        super().__init__()
        self.X = Parameter(init_points)
        self.c = manifold.c

    def forward(self):
        hyperbolic_feature = sinh_direct_map(self.c, self.X)
        return hyperbolic_feature

# ================= sim trees ===================
class TreeEuclidean(nn.Module):
    def __init__(self, lift_type, init_points: torch.Tensor, c=1., thr=0.9999) -> None:
        super().__init__()
        self.c = c
        manifold = Euclidean(init_points.shape[1])
        self.X = ManifoldParameter(init_points, manifold=manifold)
    
    def forward(self):
        return self.X

    def get_rgrad_norm(self, point: torch.Tensor, grad: torch.Tensor) -> torch.Tensor: 
        """assumming weight_decay=0"""
        rgrad = grad
        rgrad_norm = rgrad.norm(dim=-1, keepdim=True)
        return rgrad_norm
    
class TreePoincare(nn.Module):
    def __init__(self, lift_type, init_points: torch.Tensor, c=1., thr=0.9999) -> None:
        super().__init__()
        self.c = c
        manifold = PoincareBallExact(c=c)
        if lift_type == 'direct':
            self.X = ManifoldParameter(direct_map(c, init_points), manifold=manifold)
        elif lift_type == 'expmap':
            self.X = ManifoldParameter(expmap0(c, init_points / 2), manifold=manifold)
        elif lift_type == 'sinh_direct':
            self.X = ManifoldParameter(sinh_direct_map(c, init_points), manifold=manifold)
        else:
            raise NotImplementedError(f"lift type {lift_type} not available")
    
    def forward(self):
        return self.X 
    
    def get_rgrad_norm(self, point: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        assuming weight_decay=0
        rgrad_norm = lambda_x * ||grad||_2
        """
        lambda_point = 2 / (1 - self.c * point.norm(dim=-1, keepdim=True))  # TODO: verify c
        # egrad2rgrad
        rgrad = grad / lambda_point ** 2

        # rgrad norm 
        rgrad_norm = lambda_point * rgrad.norm(dim=-1, keepdim=True)
        return rgrad_norm
        

class TreeEuclideanLiftPoincare(nn.Module):
    """ parametrize on euclidean spaces """
    def __init__(self, lift_type, init_points: torch.Tensor, c=1., thr=0.9999) -> None:
        super().__init__()
        self.c = c
        self.X = Parameter(init_points)
        # select lift function 
        if lift_type == 'direct':
            self.lift_func = direct_map
        elif lift_type == 'expmap':
            self.lift_func = lambda c, x: expmap0(c, x / 2)
        elif lift_type == 'sinh_direct':
            self.lift_func = sinh_direct_map
        else:
            raise NotImplementedError(f"lift type {lift_type} not available")
    
    def forward(self):
        return self.lift_func(self.c, self.X) 
    
    def get_rgrad_norm(self, point: torch.Tensor, grad: torch.Tensor) -> torch.Tensor: 
        """assuming weight_decay=0"""
        rgrad = grad
        rgrad_norm = rgrad.norm(dim=-1, keepdim=True)
        return rgrad_norm


class TreeLorentz(nn.Module):
    def __init__(self, lift_type, init_points: torch.Tensor, c=1., thr=0.9999) -> None:
        super().__init__()
        self.c = c
        manifold = Lorentz(k=c)
        if lift_type == 'direct':
            self.X = ManifoldParameter(direct_map_lorentz(c, init_points), manifold=manifold)
        elif lift_type == 'expmap':
            self.X = ManifoldParameter(expmap0_lorentz(c, init_points), manifold=manifold)
        else:
            raise NotImplementedError(f"lift type {lift_type} not available")

    def forward(self):
        return self.X 
    
    def get_rgrad_norm(self, point: torch.Tensor, grad: torch.Tensor) -> torch.Tensor: 
        """
        assuming weight_decay=0
        rgrad_norm = rgrad
        """
        # egrad2rgrad
        d = point.shape[-1] - 1
        grad.narrow(-1, 0, 1).mul_(-1)
        inner_prod_raw = point * grad 
        inner_prod = - inner_prod_raw.narrow(-1, 0, 1) + inner_prod_raw.narrow(-1, 1, d).sum(dim=-1, keepdim=True)
        rgrad = grad + inner_prod * point / self.c  # TODO: verify c

        # get norm 
        rgrad_norm = (- rgrad.narrow(-1, 0, 1).square() + rgrad.narrow(-1, 1, d).square().sum(dim=-1, keepdim=True)).sqrt()
        return rgrad_norm


class TreeEuclideanLiftLorentz(nn.Module):
    """ parametrize on euclidean spaces """
    def __init__(self, lift_type, init_points: torch.Tensor, c=1., thr=0.9999) -> None:
        super().__init__()
        self.c = c 
        self.X = Parameter(init_points)
        if lift_type == 'direct':
            self.lift_func = direct_map_lorentz
        elif lift_type == 'expmap':
            self.lift_func = expmap0_lorentz
        else:
            raise NotImplementedError(f"lift type {lift_type} not available")

    def forward(self):
        return self.lift_func(self.c, self.X) 
    
    def get_rgrad_norm(self, point: torch.Tensor, grad: torch.Tensor) -> torch.Tensor: 
        """assuming weight_decay=0"""
        rgrad = grad 
        rgrad_norm = rgrad.norm(dim=-1, keepdim=True)
        return rgrad_norm


# ================== MLP ===================
class TreeMLP(nn.Module):
    """ Mixture model, lorentz and poincare """
    def __init__(self, in_dim, hidden_dims, out_dim, num_hyperbolic_layers, c, no_final_lift, no_bn, nl, hyp_nl) -> None: 
        super().__init__()
        self.in_dim = in_dim 
        self.non_lin = nl
        self.hyp_nl = hyp_nl
        self.c = c
        self.no_final_lift = no_final_lift
        self.hidden_dims = hidden_dims
        self.dims_list = [in_dim, *self.hidden_dims, out_dim]
        self.num_hyperbolic_layers = num_hyperbolic_layers
        self.num_euclidean_layers = len(hidden_dims) + 1 - self.num_hyperbolic_layers

        # batchnormalization 
        if not no_bn:
            self.bn = nn.BatchNorm1d(self.dims_list[self.num_euclidean_layers])
        else:
            self.bn = nn.Identity()
        
        # to be filled by children classes
        self.manifold = None 
        self.hyperbolic_layer_type = None 
        self.euclidean_layers, self.bridge_map, self.hyperbolic_layers = None, None, None 

    def get_layers(self, hyperbolic_layer, manifold):
        """ create euclidean and hyperbolic layers """
        k, l = self.num_euclidean_layers, self.num_hyperbolic_layers

        euclidean_layers_list, hyperbolic_layers_list = [], []

        # construct euclidean layers 
        for i in range(k):
            euclidean_layers_list.append(nn.Linear(self.dims_list[i], self.dims_list[i + 1]))
            if i < k - 1:
                # no non_lin at the final layer, or before bridge
                euclidean_layers_list.append(self.non_lin)  

        # construct hyperbolic layers
        for i in range(k, k + l):
            cur_dim = self.dims_list[i]
            next_dim = self.dims_list[i + 1]
            hyp_nl = self.hyp_nl if i < k + l - 1 else nn.Identity() #  no activation in the final hyperbolic layer
            hyperbolic_layers_list.append(
                hyperbolic_layer(cur_dim, next_dim, manifold, hyp_nl)
            )

        # construct bridging map 
        if self.no_final_lift and l == 0:    # no hyperbolic layer and no lifting at the end
            bridge_map = nn.Identity()
        else:
            bridge_map = partial(hyperbolic_layer.lift_map, c=self.c)

        # final packing 
        euclidean_layers = nn.Sequential(*euclidean_layers_list)
        hyperbolic_layers = nn.Sequential(*hyperbolic_layers_list)
        
        return euclidean_layers, bridge_map, hyperbolic_layers

    def forward(self, x):
        euclidean_out = self.euclidean_layers(x)
        euclidean_out = self.bn(euclidean_out) # bn before lifting 
        lifted_out = self.bridge_map(euclidean_out)
        hyperbolic_out = self.hyperbolic_layers(lifted_out)
        return hyperbolic_out

class TreeMLPLorentz(TreeMLP):
    """ Mixture Model using Lorentz """
    def __init__(self, in_dim, hidden_dims, out_dim, num_hyperbolic_layers, c, no_final_lift, no_bn, nl, hyp_nl) -> None:
        super().__init__(in_dim, hidden_dims, out_dim, num_hyperbolic_layers, c, no_final_lift, no_bn, nl, hyp_nl)
        # select hyperbolic layer 
        self.manifold = Lorentz(k=self.c)
        self.hyperbolic_layer_type = HyperbolicLorentz
        self.euclidean_layers, self.bridge_map, self.hyperbolic_layers = self.get_layers(self.hyperbolic_layer_type, self.manifold)

class TreeMLPLorentzPP(TreeMLP):
    def __init__(self, in_dim, hidden_dims, out_dim, num_hyperbolic_layers, c, no_final_lift, no_bn, nl, hyp_nl) -> None:
        super().__init__(in_dim, hidden_dims, out_dim, num_hyperbolic_layers, c, no_final_lift, no_bn, nl, hyp_nl)
        # select hyperbolic layer 
        self.manifold = Lorentz(k=self.c)
        self.hyperbolic_layer_type = HyperbolicLorentzPP
        self.euclidean_layers, self.bridge_map, self.hyperbolic_layers = self.get_layers(self.hyperbolic_layer_type, self.manifold)

class TreeMLPPoincare(TreeMLP):
    """ Mixture model using Poincare """
    def __init__(self, in_dim, hidden_dims, out_dim, num_hyperbolic_layers, c, no_final_lift, no_bn, nl, hyp_nl) -> None:
        super().__init__(in_dim, hidden_dims, out_dim, num_hyperbolic_layers, c, no_final_lift, no_bn, nl, hyp_nl)
        # select hyperbolic layer 
        self.manifold = PoincareBallExact(c=self.c)
        self.hyperbolic_layer_type = HyperbolicPoincare
        self.euclidean_layers, self.bridge_map, self.hyperbolic_layers = self.get_layers(self.hyperbolic_layer_type, self.manifold)
