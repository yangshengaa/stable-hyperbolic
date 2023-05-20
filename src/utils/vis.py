"""
visualize embeddings
"""
# load packages 
import os
from re import L 
import numpy as np
import networkx as nx 

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# constants
EPS = 1e-15

# ========= utils ========
def root_translation(x: np.ndarray, model_type, c=1., root_idx=0):
    # translate points w.r.t. root
    if model_type in ['TreePoincare', 'TreeEuclideanLiftPoincare', 'TreeMLPPoincare']:
        dim = x.shape[1]
        x = inverse_direct_map_np(c, x)  # map back to euclidean 
        x0 = np.sqrt(np.linalg.norm(x, axis=1, keepdims=True) ** 2 + 1 / c)  # the augmented coordinate
        xr = x 
        root = x[[root_idx]]
    elif model_type in ['TreeLorentz', 'TreeEuclideanLiftLorentz', 'TreeMLPLorentz', 'TreeMLPLorentzPP']:
        dim = x.shape[1] - 1
        x0 = x[:, [0]]
        xr = x[:, 1:]
        root = xr[[root_idx]]
    else:
        raise NotImplementedError()
    root_norm_sq = np.linalg.norm(root) ** 2
    # construct transformation matrix
    W_right = np.identity(dim) + (np.sqrt(1 / c + root_norm_sq) - 1) / (root_norm_sq + EPS) * root.T @ root

    # perform translation
    translated_x = - x0 @ root + xr @ W_right.T
    translated_x = direct_map_np(c, translated_x)
    return translated_x

def direct_map_np(c, x):
    mapped_x = x / (1 + np.sqrt(1 + c * np.linalg.norm(x, axis=1, keepdims=True) ** 2))
    return mapped_x

def inverse_direct_map_np(c, x):
    """ inverse of direct map, from poincare to euclidean """
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    euc_features = 2 * x / (1 - c * x_norm ** 2)
    return euc_features

# ========= vis =========
def visualize_embeddings(
        trained_emb, 
        edges, 
        shortest_path_mat, 
        model_type,
        c=1,
        root_idx=0,
        thr=0.9999,
        **kwargs
    ):
    """ 
    plot embeddings, along with the hard boundary

    :param trained_emb: the training embeddings 
    :param edges: the edge list
    :param model_type: the name of the model 
    :param loss: the loss at a particular epoch,
    :param diameter: the diameter of the embedding
    :param thr: the threshold for hard boundary 
    :param distortion: the training distortion 
    :return a fig 
    """
    # build graph        
    g = nx.Graph()
    g.add_edges_from(edges)

    # node colors 
    # ! fixed for now
    node_colors = (1 - shortest_path_mat[0] / 10).tolist()

    # translation w.r.t. the origin
    if model_type in ['TreePoincare', 'TreeEuclideanLiftPoincare' 'TreeMLPPoincare'] or model_type in ['TreeLorentz', 'TreeEuclideanLiftLorentz', 'TreeMLPLorentz', 'TreeMLPLorentzPP']:
        trained_emb = root_translation(trained_emb, model_type, c=c, root_idx=root_idx)
    
    # read node embeddings 
    fig = Figure(figsize=(6, 6))
    ax = fig.gca()
        
    nx.draw(
        g, 
        pos=dict(zip(range(g.number_of_nodes()), trained_emb)),
        node_size=15,
        width=0.1,
        node_color=node_colors,
        cmap='Blues',
        ax=ax
    )
    ax.scatter([trained_emb[root_idx][0]], [trained_emb[root_idx][1]], color='red')
    # set title 
    num_args = len(kwargs)
    title = model_type + '\n'
    for i, (key, value) in enumerate(kwargs.items()):
        title += f'{key}: {value:.4f}'
        if i < num_args - 1:
            title += ', '
        if (i + 1) % 2 == 0:
            title += '\n'
        
    ax.set_title(title, fontsize=10)

    # visualize hard boundary 
    if 'Poincare' in model_type or 'Lorentz' in model_type:
        t = np.linspace(0, 2 * np.pi, 100)
        ax.plot(thr * np.cos(t), thr * np.sin(t), linewidth=1, color='darkred')

    return fig

def convert_fig_to_array(fig):
    """ convert a matplotlib fig to numpy array (H, W, C) """
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img_arr = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return img_arr
