# Stable Hyperbolic

This repository holds all experiments supporting [The Numerical Stability of Hyperbolic Representation Learning](https://arxiv.org/abs/2211.00181), ICML 2023.

(TODO: use PMLR link)

Two functionalities are available:

1. shallow tree embeddings in different manifolds (Euclidean, Poincare, Lorentz, and Reparemtrized Euclidean;
2. stable hyperbolic SVM: compare the performances among

   - ESVM: Euclidean SVM
   - LSVM: Lorentz SVM with non-convex constraints
   - PSVM: Poincare SVM with precomputed reference points
   - LSVMPP: Lorentz SVM with reparametrized loss functions without the non-convex constraints. (PP stands for plus plus, as inspired by [Hyperbolic Neural Network++](https://openreview.net/forum?id=Ec85b0tUwbA))

## Running instructions

To run stable hyperbolic SVM, [train_svm.py](src/train_svm.py) contains all programs to train four models. An example training script is as follows: at root,

```bash
python src/train_svm.py --model LSVMPP --C 5 --epochs 5000 --data cifar --refpt raw
```

See [commands](commands) folder for more running scripts.

### Set up enviornmenet

with conda installed, we may install the folloing env to run the codes.

```bash
conda create -n shyp
conda activate shyp
conda install python==3.8
pip install torch torchvision torchaudio
pip install autopep8 jupyterlab toml timebudget tensorboard rich torch-tb-profiler
pip install --no-cache-dir geoopt==0.4.1
pip install --no-cache-dir statsmodels seaborn scipy pillow networkx tqdm gpustat scikit-learn
```

### Specify Paths

create a file named [config.toml](config.toml) that contains the path for data and results. An example is shown below:

```toml
['tree']
data_dir = 'data/tree/'
result_dir = 'results/tree/'

['svm']
data_dir = 'data/svm/'
result_dir = 'results/svm/'
```

change ```tag``` arccordingly to redirect reading and saving paths.

## Authors

Authors: Sheng Yang, Zhengchao Wan.

Please contact [Sheng Yang](mailto:shengyang@g.harvard.edu) for any questions on running the repository.
