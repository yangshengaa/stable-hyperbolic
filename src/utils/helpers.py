"""
helper functions
"""

# load packages 
from typing import Tuple
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt 


# ====================
# ----- metrics ------
# ====================
# all asusmes that test_y ranges from 0 to num_class - 1

def multiclass_acc_from_predictions(test_decision_mat: np.ndarray, test_y: np.ndarray) -> float:
    """ no calibration, average of the accuracies """
    num_classes = test_decision_mat.shape[1]
    accs = []
    for col in range(num_classes):
        test_binarized_y = ((test_y == col) * 2 - 1).astype(int)
        cur_acc = accuracy_score(test_binarized_y, test_decision_mat[:, col])
        accs.append(cur_acc)
    acc = np.mean(accs)
    return acc

def multiclass_acc_from_probabilities(test_decision_mat: np.ndarray, test_y: np.ndarray) -> float:
    """ with platt claibration, give prediction with argmax """
    predictions = np.argmax(test_decision_mat, axis=1)
    acc = accuracy_score(test_y, predictions)
    return acc 

def multiclass_f1_from_predictions(test_decision_mat: np.ndarray, test_y: np.ndarray) -> float:
    """ macro f1 from predictions """
    num_classes = test_decision_mat.shape[1]
    precisions, recalls = [], []
    for col in range(num_classes):
        test_binarized_y = ((test_y == col) * 2 - 1).astype(int)
        cur_precision = precision_score(test_binarized_y, test_decision_mat[:, col])
        cur_recall = recall_score(test_binarized_y, test_decision_mat[:, col])
        
        precisions.append(cur_precision)
        recalls.append(cur_recall)
    average_precision = np.mean(precisions)
    average_recall = np.mean(recalls)
    macro_f1 = 2 * average_precision * average_recall / (average_recall + average_precision)
    return macro_f1

def multiclass_f1_from_probabilities(test_decision_mat: np.ndarray, test_y: np.ndarray) -> float:
    """ macro f1 from probabilities """
    predictions = np.argmax(test_decision_mat, axis=1)
    f1 = f1_score(test_y, predictions, average="macro")
    return f1

# main report 
def metric_report(test_decision_mat: np.ndarray, test_y: np.ndarray) -> Tuple[float]:
    """ create a metric report and return custom metrics """
    num_classes = test_decision_mat.shape[1]
    if num_classes == 1: # binary classification
        predictions = test_decision_mat.flatten()
        acc = accuracy_score(test_y, predictions)
        f1 = f1_score(test_y, predictions)
    else:
        # if no calibration 
        if test_decision_mat.dtype == int:
            acc = multiclass_acc_from_predictions(test_decision_mat, test_y)
            f1 = multiclass_f1_from_predictions(test_decision_mat, test_y)
        # with calibration
        else:
            acc = multiclass_acc_from_probabilities(test_decision_mat, test_y)
            f1 = multiclass_f1_from_probabilities(test_decision_mat, test_y)
    
    return acc, f1
