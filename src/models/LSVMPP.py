"""
Plus-plus inspired lorentz SVM
"""

# load packages 
from typing import Tuple 

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.svm import LinearSVC

import torch
import torch.nn as nn 
from torch.nn.parameter import Parameter
from torch.optim import SGD 

# force 64
dtype = torch.float64


class LSVMPPBase(nn.Module):
    """ the torch module class for LSVMPP """
    def __init__(self, z: np.ndarray, a: np.ndarray, C: float, penalty: str, loss: str, device: str='cpu') -> None:
        super().__init__()
        self.C = C 
        self.penalty = penalty
        self.loss = loss

        self.arcsinh_1 = torch.arcsinh(torch.tensor(1., dtype=dtype, device=device))

        self.z = Parameter(torch.tensor(z, dtype=dtype, device=device))
        self.a = Parameter(torch.tensor(a, dtype=dtype, device=device))

    def reparmetrized_minkowski_product(self, X: torch.Tensor, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """ 
        < x, w >_L where w is reparametrized by z and a 
        :param X: lorentz features
        """
        # steregraphic projection
        dim = X.shape[1] - 1
        X0 = X.narrow(-1, 0, 1)
        Xr = X.narrow(-1, 1, dim)

        # compute product 
        z_norm = z.norm(dim=0)
        product = - X0 * z_norm * torch.sinh(a) + Xr @ z * torch.cosh(a)
        return product
    
    # ========= decisions ===========
    # one of them may be best suited to platt scaling 
    @torch.no_grad()
    def decision_linear(self, X: torch.Tensor) -> torch.Tensor:
        """ make prediction (still testing which is the best for platt) """
        raw = self.reparmetrized_minkowski_product(X, self.z, self.a)
        return - raw

    @torch.no_grad()
    def decision_arcsinh(self, X: torch.Tensor) -> torch.Tensor:
        """ make prediction (still testing which is the best for platt) """
        raw = self.reparmetrized_minkowski_product(X, self.z, self.a)
        return torch.arcsinh(- raw / self.z.norm())
    
    @torch.no_grad()
    def decision_arcsinh_norm(self, X: torch.Tensor) -> torch.Tensor:
        """ make prediction (still testing which is the best for platt) """
        raw = self.reparmetrized_minkowski_product(X, self.z, self.a)
        return torch.arcsinh(- raw / self.z.norm()) * self.z.norm()

    def forward(self, X: torch.Tensor, y: torch.LongTensor) -> torch.Tensor:
        """ loss function """
        # compute penalization
        if self.penalty == 'l1':
            penalty_term = 1/2 * self.z.norm()
        elif self.penalty == 'l2':
            penalty_term = 1/2 * self.z.norm().square()
        else:
            raise NotImplementedError(f'penalty {self.penalty} not supported')

        # compute loss 
        if self.loss == 'hinge':
            misclass_loss = self.C * (
                self.arcsinh_1 - torch.arcsinh(- y * self.reparmetrized_minkowski_product(X, self.z, self.a))
            ).clip(min=0).sum()
        elif self.loss == 'squared_hinge':
            misclass_loss = self.C * (
                self.arcsinh_1 - torch.arcsinh(- y * self.reparmetrized_minkowski_product(X, self.z, self.a)) 
            ).clip(min=0).square().sum()
        else:
            raise NotImplementedError(f'loss {self.loss} not supported')
        
        objective = penalty_term + misclass_loss
        return objective


class LSVMPP:
    def __init__(self,
            epochs: int, 
            lr: float, 
            C: float, 
            penalty: str, 
            loss: str,
            init_epochs: int=3000,
            device: str='cpu',
            decision_mode: str='linear'
        ) -> None:
        self.epochs = epochs
        self.lr = lr 
        self.C = C 
        self.penalty = penalty 
        self.loss = loss
        self.init_epochs = init_epochs
        self.device = device 
        self.decision_mode = decision_mode

        self.model = None 

    # =============== utils =================
    def proj_boundary(self, w: np.ndarray, alpha: float, eps: float=1e-6) -> np.ndarray:
        """
        project w to within the boundary
        """
        proj_w = w.copy()
        proj_w[1:] = (1 + alpha) * proj_w[1:]
        first_sgn = 1 if proj_w[0] >= 0 else -1
        proj_w[[0]] = first_sgn * np.sqrt(np.sum(proj_w[1:] ** 2) - eps)
        return proj_w
    
    def alpha_search(self, w: np.ndarray) -> float: 
        """ 
        use scipy to solve for alpha in projection 
        """
        res = minimize_scalar(lambda alpha: np.sum((self.proj_boundary(w, alpha) - w) ** 2))
        alpha = res.x
        return alpha

    def init_weights(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray]:
        """ init weights based on linear svc """
        # train svm 
        linear_svm = LinearSVC(
            fit_intercept=False, 
            loss='squared_hinge', C=self.C, 
            max_iter=self.init_epochs
        )
        w = linear_svm.fit(X, y).coef_[0]
        w[0] = - w[0]  # flip the first coordinate 
        
        # check feasibility 
        if - w[0] ** 2 + np.linalg.norm(w[1:]) ** 2 < 0:
            alpha = self.alpha_search(w)
            w = self.proj_boundary(w, alpha)

        # reverse engineer z and a 
        w0 = w[0]
        wr = w[1:]
        a_div_c = np.arctanh(w0 / np.linalg.norm(wr))
        z = wr / np.cosh(a_div_c)
        
        # reshape to column vectors 
        z = z.reshape(-1, 1)
        a = a_div_c.reshape(-1, 1)
        return z, a 
    
    def train(self, X: np.ndarray, y: np.ndarray, p: np.ndarray=None):
        """ train LSVMPP """
        assert p is None, "LSVMPP only supports starting from raw"

        # get init 
        z, a = self.init_weights(X, y)

        # init model 
        model = LSVMPPBase(z, a, self.C, self.penalty, self.loss, device=self.device)
        opt = SGD(model.parameters(), lr=self.lr)

        # convert to tensor 
        X_tensor = torch.tensor(X, dtype=dtype).to(self.device)
        y_tensor = torch.LongTensor(y).unsqueeze(1).to(self.device)

        model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = model(X_tensor, y_tensor)
            loss.backward()
            opt.step()
        
        self.model = model

    def decision(self, X: np.ndarray, p: np.ndarray=None) -> Tuple[np.ndarray]:
        """ make decision on LSMVPP """
        X_tensor = torch.tensor(X, dtype=dtype).to(self.device)
        
        if self.decision_mode == 'linear':
            decision_vals = self.model.decision_linear(X_tensor).detach().cpu().numpy().flatten()
        elif self.decision_mode == 'arcsinh':
            decision_vals = self.model.decision_arcsinh(X_tensor).detach().cpu().numpy().flatten()
        elif self.decision_mode == 'arcsinh_norm':
            decision_vals = self.model.decision_arcsinh_norm(X_tensor).detach().cpu().numpy().flatten()
        else:
            raise NotImplementedError(f"mode {self.decision_mode} not available")
        
        decisions = ((decision_vals > 0) * 2 - 1).astype(int)
        return decision_vals, decisions
