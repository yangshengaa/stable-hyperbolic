"""
euclidean SVM
"""

# load packages 
from typing import Tuple
import numpy as np
from sklearn.svm import LinearSVC

class ESVM:
    """ a wrapper for euclidean svm """
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
        :param init_epochs: the number of epochs to init. not used in ESVM
        """
        self.epochs = epochs
        self.lr = lr 
        self.C = C 
        self.penalty = penalty 
        self.loss = loss
        self.init_epochs = init_epochs

        self.model = None 

    def train(self, train_X: np.ndarray, train_y: np.ndarray, p_arr: np.ndarray=None):
        """ train a svm model """
        assert p_arr is None, "ESVM does not accept a precomputed reference point"
        
        model = LinearSVC(self.penalty, self.loss, C=self.C, fit_intercept=True, max_iter=self.epochs)
        model.fit(train_X, train_y)
        self.model = model

    def decision(self, X: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray]:
        """ make decision, give both decision values and decisions themselves """
        w = self.model.coef_[0]
        b = self.model.intercept_
        decision_vals = (X @ w.T + b).flatten()
        decisions = ((decision_vals > 0) * 2 - 1).astype(int)
        return decision_vals, decisions
