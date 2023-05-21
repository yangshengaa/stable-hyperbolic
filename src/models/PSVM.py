"""
Poincare SVM, proposed in 
[Highly Scalable and Provably Accurate Classification in Poincare Balls](https://arxiv.org/pdf/2109.03781.pdf)
"""

# load packages 
from typing import Tuple
import numpy as np
from sklearn.svm import LinearSVC

# constants 
EPS = 1e-15

class PSVM:
    """ puoya's Poincare SVM """
    def __init__(self,
            epochs: int, 
            lr: float, 
            C: float, 
            penalty: str, 
            loss: str,
            init_epochs: int=3000
        ) -> None:
        self.epochs = epochs 
        self.lr = lr 
        self.C = C 
        self.penalty = penalty 
        self.loss = loss 
        self.init_epochs = init_epochs  # not used 

        self.model = None 

    # ========== utils ==========
    def log_map(self, p: np.ndarray, X: np.ndarray): 
        """ vectorized logarithm map to the tangent space of p """
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
    
    # ======== train ============
    def train(self, X: np.ndarray, y: np.ndarray, p: np.ndarray):
        """ train PSVM """
        p = np.array([p]).T
        # project to log map 
        X_projected = self.log_map(p, X)
        linear_svm = LinearSVC(penalty=self.penalty, loss=self.loss, max_iter=self.epochs, fit_intercept=False, C=self.C)
        linear_svm.fit(X_projected, y)

        self.model = linear_svm
    
    def decision(self, X: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray]:
        """ make decisions """
        p = np.array([p]).T
        coef = self.model.coef_.T 
        X_projected = self.log_map(p, X)
        decision_vals = (X_projected @ coef).flatten()
        decisions = ((decision_vals > 0) * 2 - 1).astype(int)
        return decision_vals, decisions
