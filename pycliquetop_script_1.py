# ________________________________________________________________
# This script 'pycliquetop_script_1.py' loads a precomputed
# example correlation matrix from 'tester_correlations_1.mat' and
# computes and plots clique topology for this correlation matrix. 
# ________________________________________________________________
import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt 
# ________________________________________________________________
# Import functions from the PyCliqueTop_20203 module
# ________________________________________________________________
# When writing a script that is contained inside the PyCliqueTop_2023 module, 
# these functions can be imported directly from their own scripts
# and called in the script using the notation ex. 'compute_betti_curves()' and 'matrix_2_betti_curves()'
from compute_betti_curves import compute_betti_curves
from matrix_2_order_matrix import matrix_2_order_matrix
from matrix_2_persistence_diagrams import matrix_2_persistence_diagrams
from persistence_diagrams_2_betti_curves import persistence_diagrams_2_betti_curves
from plot_betti_curves import plot_betti_curves

'''
# When writing a script that is NOT contained inside the PyCliqueTop_20203 module, 
# these functions must be imported from the module and called from their own script using the notation 
# ex. 'compute_betti_curves.compute_betti_curves()' and 'matrix_2_betti_curves.matrix_2_betti_curves()'
from pyCliqueTop_2023_apr_2_2023 import compute_betti_curves
from pyCliqueTop_2023_apr_2_2023 import matrix_2_betti_curves
from pyCliqueTop_2023_apr_2_2023 import matrix_2_order_matrix
from pyCliqueTop_2023_apr_2_2023 import matrix_2_persistence_diagrams
from pyCliqueTop_2023_apr_2_2023 import persistence_diagrams_2_betti_curves
from pyCliqueTop_2023_apr_2_2023 import plot_betti_curves
'''
# ________________________________________________________________
# Dependencies: numpy, scipy, matplotlib, Ripser (python package)
#
# THIS SCRIPT loads a precomputed correlation matrix from the 
# file 'tester_correlations_1.mat'.  
#
# It then calls 'compute_betti_curves()' for this matrix to return Betti curves,
# and calls 'plot_betti_curves()' to plot the Betti curves for this matrix.  
#
# Running this script 'pyclique_top_script_1.py' will produce a figure with a single panel.
# This panel will have a set of Betti curves.
# ________________________________________________________________
# (1) Load a precomputed symmetric matrix.  
# ________________________________________________________________
mdict = scipy.io.loadmat('tester_correlations_1.mat',simplify_cells=True)
A = mdict['correlations']
# ________________________________________________________________
# (2) Call 'compute_betti_curves()' on the matrix A.  
# ________________________________________________________________
[betti_curves, edge_densities] = compute_betti_curves(A) # Here we are assuming that A is a correlation (similarity) matrix, so we leave the optional input parameter 'similarity' as its default 'True'.
# ________________________________________________________________
# (3) Call 'plot_betti_curves()'.
# ________________________________________________________________
colors = ['black', 'blue','red','green']
title_str = 'tester_correlations_1'
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,7))
ax = plot_betti_curves(ax, betti_curves, edge_densities, colors, title_string = '%s' % title_str)
plt.suptitle('pycliquetop_script_1.py')
plt.show()
###############