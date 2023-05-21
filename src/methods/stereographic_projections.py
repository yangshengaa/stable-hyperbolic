"""
functions for stereographic projections
"""

# load packages 
import numpy as np

def poincare_to_loretnz(X: np.ndarray) -> np.ndarray:
    """ project poincare features to lorentz """
    X_norm_square = np.square(np.linalg.norm(X, axis=-1, keepdims=True))
    x0 = (X_norm_square + 1) / (1 - X_norm_square)
    xr = 2 / (1 - X_norm_square) * X 
    X_projected = np.hstack([x0, xr])
    return X_projected

def lorentz_to_poincare(X: np.ndarray) -> np.ndarray:
    """ project lorentz features to poincare """
    xr = X[:, 1:]
    X_projected = xr / (1 + X[:, [0]])
    return X_projected
