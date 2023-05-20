"""
manifold layers
"""

# load packages
import torch 
import torch.nn as nn 
from torch.nn.parameter import Parameter

# init params
INIT_MEAN = 0 
INIT_STD = 0.5

class HyperbolicLorentz(nn.Module):
    """ lorentz hyperbolic layer """
    def __init__(self, in_dim, out_dim, manifold, nl) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 
        self.manifold = manifold 
        self.c = self.manifold.k  # curvature of the lorentz model 
        self.nl = nl

        # bn 
        self.bn = nn.BatchNorm1d(out_dim)

        # init 
        self.z = Parameter(torch.normal(INIT_MEAN, INIT_STD, (in_dim, out_dim)))
        self.a = Parameter(torch.normal(INIT_MEAN, INIT_STD, (1, out_dim)))

    # aux 
    def get_w(self, z, a):
        """ get tangent space vectors from euclidean ones, by a direct map using Jacobian """
        z_norm_square = z.norm(dim=0, keepdim=True).square()
        w0 = z_norm_square / (self.c + z_norm_square)
        w = a * torch.cat([w0, z], dim=0)
        return w 

    # lifting 
    @staticmethod
    def lift_map(X: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """ direct lifting, from euclidean to Lorentz """
        x0 = (X.norm(dim=-1, keepdim=True) ** 2 + 1 / c).sqrt()
        lorentz_features = torch.cat([x0, X], dim=1)
        return lorentz_features
    
    # distance 
    def lorentz_dist2plane(self, W, X):
        """ 
        vectorized lorentz dist2plane 

        arcsinh(-<w, x>_L / (sqrt(<w, w>_L) * sqrt(c)) * ||w||_L
        adapted from https://proceedings.mlr.press/v89/cho19a.html, and a thorough discussiong with Zhengchao
        """
        numerator = - X.narrow(-1, 0, 1) @ W[[0]] + X.narrow(-1, 1, self.in_dim) @ W[1:]
        w_norm = torch.sqrt(- W.narrow(0, 0, 1).square() + W.narrow(0, 1, self.in_dim).square().sum(dim=0, keepdim=True))
        denom = w_norm * torch.sqrt(self.c)
        distance = torch.arcsinh(- numerator / denom) * w_norm
        return distance 

    def forward(self, X):
        W = self.get_w(self.z, self.a)
        euclidean_features = self.lorentz_dist2plane(W, X)
        nl_euclidean_features = self.nl(euclidean_features)
        bn_euclidean_features = self.bn(nl_euclidean_features)
        hyp_features = self.lift_map(bn_euclidean_features, self.c)
        return hyp_features

class HyperbolicLorentzPP(nn.Module):
    """ using the PP trick to ensure convexity """
    def __init__(self, in_dim, out_dim, manifold, nl) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 
        self.manifold = manifold 
        self.c = self.manifold.k  # curvature of the lorentz model 
        self.nl = nl

        # bn 
        self.bn = nn.BatchNorm1d(out_dim)

        # init 
        self.z = Parameter(torch.normal(INIT_MEAN, INIT_STD, (in_dim, out_dim)))
        self.a = Parameter(torch.normal(INIT_MEAN, INIT_STD, (1, out_dim)))

    # aux
    def get_w(self, z, a):
        """ get tangent space vectors from euclidean ones, by a expmap at 0 followed by a parallel transport (similar in PP) """
        z_norm = z.norm(dim=0, keepdim=True)
        w0 = z_norm * torch.sinh(a / self.c.sqrt())
        wr = torch.cosh(a / self.c.sqrt()) * z 
        w = torch.cat([w0, wr], dim=0)
        return w

    # lifting 
    @staticmethod
    def lift_map(X: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """ direct lifting, from euclidean to Lorentz """
        x0 = (X.norm(dim=-1, keepdim=True) ** 2 + 1 / c).sqrt()
        lorentz_features = torch.cat([x0, X], dim=1)
        return lorentz_features
    
    # distance 
    def lorentz_dist2plane(self, W, X):
        """ 
        vectorized lorentz dist2plane 

        arcsinh(-<w, x>_L / (||z||_2 * sqrt(c)) * ||z||_2
        adapted from https://proceedings.mlr.press/v89/cho19a.html, and a thorough discussiong with Zhengchao
        """
        numerator = - X.narrow(-1, 0, 1) @ W[[0]] + X.narrow(-1, 1, self.in_dim) @ W[1:]
        z_norm = self.z.norm(dim=0, keepdim=True)
        denom = z_norm * self.c.sqrt()
        distance = torch.arcsinh(numerator / denom) * z_norm
        return distance 

    def forward(self, X):
        W = self.get_w(self.z, self.a)
        euclidean_features = self.lorentz_dist2plane(W, X)
        nl_euclidean_features = self.nl(euclidean_features)
        bn_euclidean_features = self.bn(nl_euclidean_features)
        hyp_features = self.lift_map(bn_euclidean_features, self.c)
        return hyp_features


class HyperbolicPoincare(nn.Module):
    """ using the direct map """
    def __init__(self, in_dim, out_dim, manifold, nl) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.manifold = manifold 
        self.c = manifold.c  # curvature of the Poincare Model 
        self.nl = nl

        # bn 
        self.bn = nn.BatchNorm1d(out_dim)

        # init parameters
        self.z = Parameter(nn.init.normal_(torch.Tensor(in_dim, out_dim), mean=INIT_MEAN, std=INIT_STD))
        self.r = Parameter(nn.init.normal_(torch.Tensor(1, out_dim), mean=INIT_MEAN, std=INIT_STD))

    # distance 
    def normdist2planePP(self, x: torch.Tensor, z: torch.Tensor, r: torch.Tensor, c: torch.Tensor):
        """ hyperbolic neural network plus plus equation 6 """
        conformal_factor = 2 / (1 - c * x.norm(dim=-1, keepdim=True) ** 2)
        sqrt_c = c ** 0.5
        z_norm = z.norm(dim=0, keepdim=True)
        z_normalized = z / z_norm
        
        v_k = 2 / sqrt_c * z_norm * torch.arcsinh(
            conformal_factor * (sqrt_c * x @ z_normalized) * torch.cosh(2 * sqrt_c * r) 
            - (conformal_factor - 1) @ torch.sinh(2 * sqrt_c * r)
        )
        return v_k
    
    # lifting 
    @staticmethod
    def lift_map(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """ a direct map from euclidean to Poincare """
        mapped_x = x / (1 + torch.sqrt(1 + c * torch.linalg.norm(x, dim=-1, keepdim=True) ** 2))
        return mapped_x
    
    def forward(self, X):
        euclidean_features = self.normdist2planePP(X, self.z, self.r, self.c)
        euclidean_features_nl = self.nl(euclidean_features)
        euclidean_features_bn = self.bn(euclidean_features_nl)
        hyperbolic_features = self.lift_map(euclidean_features_bn, self.c)
        return hyperbolic_features
