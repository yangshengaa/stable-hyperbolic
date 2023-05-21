"""
Platt scaling, with an improved implementation

Distributed under terms of the MIT license.
Modified from http://www.work.caltech.edu/~htlin/program/libsvm/doc/platt.py
"""

# load packages 
from typing import Tuple
import numpy as np

def platt_train(
        decision_vals: np.ndarray, 
        labels: np.ndarray, 
        prior0: float=None, 
        prior1: float=None,
        max_iteration: int=1000,
        min_step: float=1e-10,
        sigma: float=1e-12,
        eps: float=1e-5
    ) -> Tuple[float]:
    """ 
    the training process of platt 
    
    :param max_iteration: maximum iterations to train platt
    :param min_step: the minimum stepsize in the line search
    :param sigma: for numerical PD 
    :param eps: convergence guarantee threshold
    """
    # Count prior0 and prior1 if needed
    n = len(labels)
    if prior0 is None and prior1 is None:
        prior1 = sum(labels > 0)
        prior0 = n - prior1
    
    # count target support
    hi_target = (prior1 + 1.0) / (prior1 + 2.0)
    lo_target = 1 / (prior0 + 2.0)
    t = np.where(labels > 0, hi_target, lo_target)

    # initial point and initial function value
    a, b = 0, np.log((prior0 + 1) / (prior1 + 1))
    
    fapb = decision_vals * a + b
    t_transformed = np.where(fapb >= 0, t, t - 1)
    fval = np.sum(
        t_transformed * fapb + np.log(1 + np.exp(- np.abs(fapb)))
    )

    for _ in range(max_iteration):
        # update gradient and hessian 
        h11 = h22 = sigma
        h21 = g1 = g2 = 0 
        
        fapb = decision_vals * a + b

        p = np.where(fapb >= 0, np.exp(-fapb), 1) / (1 + np.exp(-np.abs(fapb)))
        q = 1 - p

        d2 = p * q 
        h11 += np.sum(decision_vals * decision_vals * d2)
        h22 += np.sum(d2)
        h21 += np.sum(decision_vals * d2) 

        d1 = t - p
        g1 += np.sum(decision_vals * d1)
        g2 += np.sum(d1)

        # stopping criterion
        if abs(g1) < eps and abs(g2) < eps:
            break
        
        # finding Newton's Direction 
        det = h11 * h22 - h21 * h21
        dA = -(h22 * g1 - h21 * g2) / det
        dB = -(-h21 * g1 + h11 * g2) / det
        gd = g1 * dA + g2 * dB

        # line seearch 
        stepsize = 1
        while stepsize >= min_step:
            new_a = a + stepsize * dA
            new_b = b + stepsize * dB

            # new function values
            fapb = decision_vals * new_a + new_b 
            t_transformed = np.where(fapb >= 0, t, t - 1)
            new_f = np.sum(
                t_transformed * fapb + np.log(1 + np.exp(- np.abs(fapb)))
            )

            # check sufficient decrease 
            if new_f < fval + 0.0001 * stepsize * gd:
                a, b, fval = new_a, new_b, new_f
                break 
            else:
                stepsize = stepsize / 2.
            
        # failure management 
        if stepsize < min_step:
            print("line search failed")
            return a, b
        
    return a, b


def platt_test(decision_vals: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """ make decisions """
    a, b = ab 
    fapb = decision_vals * a + b 
    decisions = np.where(fapb >= 0, np.exp(-fapb), 1) / (1 + np.exp(-np.abs(fapb)))
    return decisions
