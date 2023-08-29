import ripser as rp
import numpy as np
import scipy.io
from scipy.stats import rankdata
import matplotlib.pyplot as plt

def plot_betti_curves(ax, BCs, edge_densities,colors,title_string):
    for j in range(np.shape(BCs)[1]):
        ax.plot(edge_densities,BCs[:,j],c = colors[j],drawstyle='steps-post')
    ax.set_xlabel('edge density')
    ax.set_ylabel('# cycles')
    ax.set_title('%s' % title_string)
    return ax

