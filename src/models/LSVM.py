"""
Lorentz SVM, proposed in 
[Large-Margin Classification in Hyperbolic Space](https://proceedings.mlr.press/v89/cho19a.html)

Codes are adapted from the original matlab implementation with some pythonic improvements
"""

# load packages 
from typing import Tuple
import numpy as np 
from scipy.optimize import minimize_scalar
from sklearn.svm import LinearSVC

class LSVM:
    def __init__(self, 
            epochs: int, 
            lr: float, 
            C: float, 
            penalty: str, 
            loss: str,
            init_epochs: int=3000
        ) -> None:
        """
        :param epochs: the max epochs to train 
        :param lr: the learning rate for training
        :param C: the tolerance level
        :param penalty: l1 or l2
        :param loss: squared_hinge or hinge
        :param init_epochs: the number of epochs to init. 
        """
        self.epochs = epochs
        self.lr = lr 
        self.C = C 
        self.penalty = penalty
        self.loss = loss 
        self.init_epochs = init_epochs
        assert penalty == 'l1', "only l1 penalty is supported for LSVM"
        assert loss == 'squared_hinge', "only squared_hinge loss is supported for LSVM"

        self.model = None 

    # =========== utils =============
    def loss_fn(self, w: np.ndarray, X: np.ndarray, y: np.ndarray, C: float) -> float:
        """ compute the loss function, with l1 penalty and squared hinge loss """
        loss_term = 1 / 2 * (- np.square(w[0, 0]) + np.dot(w[1:].T, w[1:]).item())
        misclass_term = y.reshape(-1, 1) * (- w[[0]] * X[:, [0]] + X[:, 1:] @ w[1:])
        misclass_loss = np.arcsinh(1.) - np.arcsinh(- misclass_term)
        loss = loss_term + C * np.sum(np.where(misclass_loss > 0, misclass_loss, 0))
        return loss 

    def grad_fn(self, w: np.ndarray, X: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
        """ compute gradient of the loss function """
        grad_margin = np.vstack((-w[[0]], w[1:]))
        z = y.reshape(-1, 1) * (- w[[0]] * X[:, [0]] + X[:, 1:] @ w[1:])
        misclass = (np.arcsinh(1.) - np.arcsinh(- z)) > 0
        arcsinh_term = -1 / np.sqrt(z ** 2 + 1)
        mink_prod_term = y.reshape(-1, 1) * np.hstack((X[:, [0]], - X[:, 1:]))
        grad_misclass = misclass * arcsinh_term * mink_prod_term
        grad_w = grad_margin + C * np.sum(grad_misclass, axis=0, keepdims=True).T 
        return grad_w 

    def is_feasible(self, w: np.ndarray):
        """ check if w is in the feasible region """
        feasibility = - w[0, 0] ** 2 + np.dot(w[1:].T, w[1:]).squeeze().item()
        return feasibility > 0
    
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
    
    # =========== init ==================
    def init_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ 
        initizialize based on linear svm 
        """
        # note that here we could only use l2 penalty, since l1 + squared hinge is not supported 
        linear_svm = LinearSVC(loss='hinge', C=self.C, fit_intercept=False, max_iter=self.init_epochs)
        init_coef = linear_svm.fit(X, y).coef_.T 
        init_coef[0, 0] = - init_coef[0, 0]  # corrected initialization
        return init_coef

    # ========= train ========
    def train(self, X: np.ndarray, y: np.ndarray, p: np.ndarray=None):
        """ train LSVM. p here is useless """
        assert p is None, "LSVM only supports raw training"

        # initialize 
        init_w = self.init_weights(X, y)
        if not self.is_feasible(init_w):
            init_w = self.proj_boundary(init_w, alpha=0.01)
        
        # train 
        w_new = init_w
        best_w = init_w
        init_loss = self.loss_fn(init_w, X, y, self.C)
        min_loss = init_loss
        for _ in range(self.epochs):
            current_loss = 0
            # full batch GD 
            grad_w = self.grad_fn(w_new, X, y, self.C)
            w_new = w_new - self.lr * grad_w
            # if not in feasible region, need to use projection
            if not self.is_feasible(w_new):
                # solve optimization problem for nearest feasible point
                alpha_opt = self.alpha_search(w_new)
                # project w to feasible sub-space
                w_new = self.proj_boundary(w_new, alpha_opt)
            current_loss = self.loss_fn(w_new, X, y, self.C)

            # update loss and estimate 
            if current_loss < min_loss:
                min_loss = current_loss
                best_w = w_new
    
        self.coef = best_w 

    def decision(self, X: np.ndarray, p: np.ndarray=None) -> Tuple[np.ndarray]:
        """ make decisions """
        decision_vals = (X[:, [0]] * self.coef[[0]] - X[:, 1:] @ self.coef[1:]).flatten()
        decisions = ((decision_vals > 0) * 2 - 1).astype(int)
        return decision_vals, decisions
