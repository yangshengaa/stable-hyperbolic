# Stable Hyperbolic

This repository holds all experiments supporting [The Numerical Stability of Hyperbolic Representation Learning](https://arxiv.org/abs/2211.00181), ICML 2023.

(TODO: use PMLR link)

Two functionalities are available:

1. shallow tree embeddings in different manifolds (Euclidean, Poincare, Lorentz, and Reparemtrized Euclidean;
2. stable hyperbolic SVM.

## Running instructions

### Set up enviornmenet

with conda installed, we may install the folloing env to run the codes.

```bash
conda create -n shyp
conda activate shyp
conda install python==3.8
pip install pytorch torchvision torchaudio pytorch
pip install autopep8 jupyterlab toml timebudget tensorboard rich torch-tb-profiler
pip install --no-cache-dir geoopt==0.4.1
pip install --no-cache-dir statsmodels seaborn scipy pillow networkx tqdm
pip install -e .
```

### Specify Paths

create a file named [config.toml](config.toml) that contains the path for data and results. An example is shown below:

```toml
['simulation']
data_dir = '../data'
result_dir = '../results'
```

## Authors

Authors: Sheng Yang, Zhengchao Wan.

Please contact [Sheng Yang](mailto:shengyang@g.harvard.edu) for any questions on running the repository.
