"""
create hierarchical synthetic dataset
"""

# load packages 
import os 
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split

# for repreducibility 
np.random.seed(500)

# paths
data_path = '../../data'

# constants
RADIUS = 1
STD = 0.5
TOTAL_SAMPLES = 600

def generate_points(k: int) -> Tuple[np.ndarray]:
    """ generate points in euclidean and project to hyperbolic """
    Xs = []
    ys = []
    
    thetas = np.random.uniform(0, 2 * np.pi, size=(k, 1))
    points = np.hstack([np.cos(thetas) * RADIUS, np.sin(thetas) * RADIUS]) 
    for i in range(k):
        cur_point = points[[i]]
        noise = np.random.normal(0, STD, size=(TOTAL_SAMPLES // k, 2))
        cur_point_with_noise = cur_point + noise 
        Xs.append(cur_point_with_noise)
        ys.append([i] * (TOTAL_SAMPLES // k))
    
    X = np.vstack(Xs)
    y = np.hstack(ys)
    return X, y

def expmap(X: np.ndarray) -> np.ndarray:
    """ expmap of Poincare at the origin """
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    poincare_features = np.tanh(X_norm) * X / X_norm
    return poincare_features

def main():
    """ generate points """
    ks = [2, 3, 5, 10, 15]
    idx_s = [6, 7, 8, 9, 10]
    for idx, k in zip(idx_s, ks):
        # create folder 
        folder_path = os.path.join(data_path, f'sim_data_{idx}')
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        # get point 
        X, y = generate_points(k)
        X_poincare = expmap(X / 2)

        # train test split 
        X_train, X_test, y_train, y_test = train_test_split(X_poincare, y, test_size=0.25, stratify=y)
        np.savez(os.path.join(folder_path, f"sim_data_{idx}_poincare_embedding.npz"), x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)

        # vis 
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.scatter(X_poincare[:, 0], X_poincare[:, 1], c=y)
        plt.savefig(os.path.join(folder_path, 'vis.png'), facecolor='white', bbox_inches='tight', dpi=100)

if __name__ == '__main__':
    main()
