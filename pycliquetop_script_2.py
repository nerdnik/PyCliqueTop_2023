# ________________________________________________________________
# This script 'pycliquetop_script_2.py' computes and plots clique topology
# for three example symmetric matrices.
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
# THIS SCRIPT constructs three random matrices
#           - a random symmetric iid matrix (random matrix)
#           - a pairwise distance matrix of random points in Euclidean space (difference matrix)
#           - a correlation matrix of random vectors in Euclidean space (similarity matrix)
#
# It then calls 'compute_betti_curves()' for each of these matrices to return Betti curves,
# and calls 'plot_betti_curves()' to plot the Betti curves for each of these matrices.
#
# Running this script 'pycliquetop_script_2.py' will produce a figure with three panels.
# Each panel will have a set of Betti curves for one of the three random matrices.
# ________________________________________________________________
# (1) Set the size of the (n,n) symmetric matrix 
# ________________________________________________________________
n = 40 
# ________________________________________________________________
# (2) Make a random symmetric iid matrix of size (n,n)
# ________________________________________________________________
A_random_iid = np.random.uniform(size=(n,n))
A_random_iid = (A_random_iid +  np.transpose(A_random_iid))/2
for i in range(n):
    A_random_iid[i,i]=0
# ________________________________________________________________
# (3) Make a uniformly random sample of 'n' points in a 'geometric_dim'-dimensional 
# unit cube in Euclidean space
# ________________________________________________________________
geometric_dim = 3
xy_coords = np.random.uniform(size=(n,geometric_dim))
# ________________________________________________________________
# (4) Make a random geometric matrix of size (n,n) 
# where entries are the distances between the points sampled 
# uniformly in a 'geometric_dim'-dimensional unit cube in Euclidean space
# ________________________________________________________________
A_distances = np.zeros((n,n))
for i in range(n):
    for j in range(i+1,n):
        A_distances[i,j] = A_distances[j,i] = np.linalg.norm((xy_coords[i,:]-xy_coords[j,:]))
# ________________________________________________________________
# (5) Make a random correlation matrix of size (n,n)
# where entries are the correlations between points the sampled 
# uniformly in a 'geometric_dim' dimensional unit cube in Euclidean space
#  scipy.spatial.distance.cosine(u, v
# ________________________________________________________________
A_correlations = np.ones((n,n))
for i in range(n):
    for j in range(i+1,n):
        A_correlations[i,j] = A_correlations[j,i] = -1*scipy.spatial.distance.cosine(xy_coords[i,:],xy_coords[j,:]) + 1 #scipy.stats.pearsonr(xy_coords[i,:],xy_coords[j,:])[0]  
# ________________________________________________________________
# (6) Call the function 'compute_betti_curves()' to get Betti curves for
# each of the three symmetric matrices made above: A_distances, A_correlations, A_random_iid.
#
# Notice that for distance matrices we set the input parameter 'similarity' to False, as this is a dissimilarity matrix.
# For correlation matrices we set the input parameter 'similarity' to True, as this is a similarity matrix.
# For random symmetric iid matrices, either option is appropriate. 
# Here we also set the input parameter 'max_dim' to 3, meaning we will compute the persistent homology
# in dimensions 0, 1, 2 and 3 and get a Betti curve for each of these dimensions.
# In particular, the column xxx_betties[:,i] of xxx_bettis is the ith Betti curve.   
# ________________________________________________________________
[distance_bettis, distance_edge_densities] = compute_betti_curves(A_distances, max_dim = 3, similarity = False)
[correlation_bettis, correlation_edge_densities] = compute_betti_curves(A_correlations, max_dim = 3, similarity = True)
[random_bettis, random_edge_densities] = compute_betti_curves(A_random_iid, max_dim = 3, similarity = True)
# ________________________________________________________________
# (7) Plot the Betti curves by calling 'plot_betti_curves()'.
# This function populates a panel in a figure  that is already created
# before the function call with the specific panel as both an input and the output
# of the function.  
# ________________________________________________________________
fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (12,7))
colors = ['black','blue','red','green']
#
ax[0,0].imshow(A_distances, cmap='jet')
ax[0,1].imshow(A_correlations,cmap='jet')
ax[0,2].imshow(A_random_iid,cmap='jet')
ax[1,0] = plot_betti_curves(ax[1,0], distance_bettis, distance_edge_densities, colors, title_string = 'distances')
ax[1,1] = plot_betti_curves(ax[1,1], correlation_bettis, correlation_edge_densities, colors, title_string = 'correlations')
ax[1,2] = plot_betti_curves(ax[1,2], random_bettis, random_edge_densities, colors, title_string = 'random iid')
#
plt.suptitle('pycliquetop_script_2.py: compute_betti_curves() with n = %d, dim = %d' % (n,geometric_dim))
plt.show()
