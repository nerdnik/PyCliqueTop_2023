import ripser as rp
import numpy as np
import scipy.io
from scipy.stats import rankdata
import matplotlib.pyplot as plt

# Written on August 28, 2023 by Nikki Sanderson
# 
# THIS FUNCTION populates a panel of a plot with the
# precomputed Betti curves using the python package matplotlib.
# The x-axis is labeled with "edge density" and the y-axis is labeled with "# cycles".
#
# INPUTS:
#  ax - matplotlib object, defined outside of this function using matplotlib's subplots routine, 
# this is the subplot in which to plot the Betti curves.
#
#  BCs - numpy array, size (N,max_dim+1), where the ith column is the 
# (i-1)-dimensional Betti curve (ex. the first column, BCs[:,0], is Betti-0) 
# and max_dim is the maximum persistent homology dimension computed and N is the 
# number of steps in the filtration
# 
#  edge_densities - list (CHECK!), length N, the fraction of edges of the complete graph
# included in the graph at each step of the sequence used to make the order complex
#
#  colors - list, length is max_dim + 1, a list of colors to plot each of the Betti curves in
#
#  title_string - string, a title for the panel
#
# RETURNS:
#  ax - matplotlib object, the subplot in which to plot the Betti curves, defined outside of this function and
# provided as an input, now updated to have the Betti curves and labels.
#

def plot_betti_curves(ax, BCs, edge_densities,colors,title_string):
    for j in range(np.shape(BCs)[1]):
        ax.plot(edge_densities,BCs[:,j],c = colors[j],drawstyle='steps-post')
    ax.set_xlabel('edge density')
    ax.set_ylabel('# cycles')
    ax.set_title('%s' % title_string)
    return ax

