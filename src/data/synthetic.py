""" 
create synthetic data and do train test split
simulate Gaussian Mixture model 
procedure:
- fix k, the number of labels
- for each label, generate gaussian
"""

# load packages 
import os 
from typing import Tuple
import sys 
sys.path.append('../..')
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
plt.style.use('ggplot')

import torch
torch.set_default_dtype(torch.float64)

torch.manual_seed(442)

from geoopt.manifolds import Lorentz

# load files 
from src.methods import lorentz_to_poincare

# constants 
META_MEAN = 0
META_STD = 1.5
TOTAL_SAMPLES = 1200

c = 1
lorentz = Lorentz(k=c)

dataset_path = '../../data/'

# ================ lorentz and poincare features ====================

def generate_lorentz_features(k: int, d: int=2) -> Tuple[torch.Tensor]:
    """ 
    generate k cluster features 
    
    :param k: the number of clusters 
    :param d: the dimension 
    """
    # features 
    lorentz_features = []
    num_samples_per_k = TOTAL_SAMPLES // k
    for _ in range(k):
        samples = torch.normal(0, 1, (num_samples_per_k, d))
        zeros = torch.zeros((num_samples_per_k, 1))
        mu = torch.normal(META_MEAN, META_STD, (1, d))

        # find reference
        mu_augmented = torch.cat([torch.zeros(1, 1), mu], dim=-1)
        p = lorentz.expmap0(mu_augmented)

        # PT 
        samples_augmented = torch.cat([zeros, samples], dim=-1)
        samples_tangent = lorentz.transp0(p, samples_augmented)

        # lorentz_features 
        sample_lorentz_features = lorentz.expmap(p, samples_tangent)
        lorentz_features.append(sample_lorentz_features)
    
    lorentz_features = torch.cat(lorentz_features, dim=0)

    # labels 
    labels = torch.cat([torch.tensor([i for _ in range(num_samples_per_k)]).long() for i in range(k)])
    return lorentz_features, labels

# ======================= driver ======================

def main():
    """ generate different dataset and perform IO """
    # dataset 1 2 3 
    d = 2 
    for i, k in enumerate([2, 3, 5, 10, 15]):
        lorentz_features, lorentz_labels = generate_lorentz_features(k, d)
        poincare_features = lorentz_to_poincare(lorentz_features)

        y = lorentz_labels.detach().cpu().numpy().flatten().astype(int)
        X = poincare_features.detach().cpu().numpy()
        
        folder_path = os.path.join(dataset_path, f'sim_data_{i + 1}')
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        # train test split 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
        np.savez(os.path.join(folder_path, f"sim_data_{i + 1}_poincare_embedding.npz"), x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
        
        # vis 
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.scatter(poincare_features[:, 0], poincare_features[:, 1], c=lorentz_labels)
        plt.savefig(os.path.join(folder_path, 'vis.png'), facecolor='white', bbox_inches='tight', dpi=100)

if __name__ == '__main__':
    main()
