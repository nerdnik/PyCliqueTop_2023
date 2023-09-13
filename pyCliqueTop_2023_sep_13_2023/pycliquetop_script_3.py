# ________________________________________________________________
# This script 'pycliquetop_script_3.py' computes and plots clique topology
# and intermediate data for three example symmetric matrices.  
# ________________________________________________________________
import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt 
# ________________________________________________________________
# Import functions from the PyCliqueTop_20203 module
# ________________________________________________________________
#
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
#
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
# Running this script 'pyclique_top_script_29aug2023.py' will produce a figure with three panels.
# Each panel will have a set of Betti curves for one of the three random matrices.
#
# Additionally, this script will 
#
# Running this script 'pyclique_top_script_12sept2023.py' will 
# ________________________________________________________________
# (1) Set the size of the (n,n) symmetric matrix (equivalently, the number of vertices in the order complex)
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
#
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
fig1, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12,4))
colors = ['black','blue','red','green']
#
ax[0] = plot_betti_curves(ax[0], distance_bettis, distance_edge_densities, colors, title_string = 'distances')
ax[1] = plot_betti_curves(ax[1], correlation_bettis, correlation_edge_densities, colors, title_string = 'correlations')
ax[2] = plot_betti_curves(ax[2], random_bettis, random_edge_densities, colors, title_string = 'random iid')
#
plt.suptitle('pycliquetop_script_3.py: compute_betti_curves() --- n = %d, dim = %d' % (n,geometric_dim))
#plt.show()
# ________________________________________________________________
# ________________________________________________________________
#  * * * *  LOOKING AT THE INTERNAL FUNCTIONS  * * * * 
#   Internally, the function 'compute_betti_curves()' calls on three other functions:
#       matrix_2_order_matrix()
#       matrix_2_persistence_diagrams()
#       persistence_diagrams_2_betti_curves()
#
# These intermediate functions can be called to return the intermediate data generated to get Betti curves from matrices.
# ________________________________________________________________
# ________________________________________________________________
# (8) Calling 'matrix_2_order_matrix()' on a matrix returns an integer matrix, 
# called the order matrix, that ranks the input matrix entries with the 
# lowest matrix value getting a rank of 1.
#
# Notice that this function ranks the matrix entries from low to high.
# Additionally, the persistent homology computation in the function 
# matrix_2_persistence_diagrams() adds edges into the filtration 
# of order complexes from low to high. 
# (This is hardcoded in the call to ripser() inside of matrix_2_persistence_diagrams()
# where the input parameter 'distance_matrix' is set to True.)
#
# So if we next want to compute the persistent homology of 
# the order matrix we get from 'matrix_2_order_matrix()'
# corresponding to a similarity (ex. correlation) matrix - 
# where high matrix values reflect greater similarity between the points - 
# we must invert the ordering of entries in the similarity matrix 
# BEFORE getting the order matrix with 'matrix_2_order_matrix()' so that
# the edges representing high similarity (high correlation) between points 
# are added into the filtration of the oder complex first. 
#
# Here we do this by multiplying the similarity (ex. correlation) matrix by -1.
# (Notice this is done internally when using the wrapper function 'compute_betti_curves()'
# if the input parameter 'similarity' is set to True.)
#
# Alternatively, one can get the order matrix of the similarity (ex. correlation)
# matrix without any modification of the input matrix and 
# AFTERWARDS multiply the order matrix itself by -1
# before calling 'matrix_2_persistence_diagrams()' to get the persistent homology.  
# ________________________________________________________________
distances_order_matrix =  matrix_2_order_matrix(A_distances)
correlations_order_matrix =  matrix_2_order_matrix(-1*A_correlations)
random_iid_order_matrix =  matrix_2_order_matrix(A_random_iid)
# ________________________________________________________________
# (9) Plotting the three distance matrices above the three corresponding order matrices.
# ________________________________________________________________
fig2, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (12,7))
ax[0,0].imshow(A_distances,cmap='jet')
ax[1,0].imshow(distances_order_matrix,cmap='jet')
ax[0,1].imshow(A_correlations,cmap='jet')
ax[1,1].imshow(correlations_order_matrix,cmap='jet')
ax[0,2].imshow(A_random_iid,cmap='jet')
ax[1,2].imshow(random_iid_order_matrix,cmap='jet')
#
ax[0,0].set_title('distances')
ax[1,0].set_title('order distances')
ax[0,1].set_title('correlations')
ax[1,1].set_title('order correlations')
ax[0,2].set_title('random iid')
ax[1,2].set_title('order random iid')
#
plt.suptitle('pycliquetop_script_3.py: matrix_2_order_matrix() --- n = %d, dim = %d' % (n,geometric_dim))
#plt.show()
# ________________________________________________________________
# (10) Calling 'matrix_2_persistence_diagrams()' on an order matrix returns a list of persistence diagrams,
# one for each dimension from 0 through optional input parameter 'max_dim'. 
# The other optional input parameter 'thresh' determines what value of matrix entries 
# below which to include in the persistent homology computations.  
# By default, these are set to 'max_dim = 3' and 'thresh = np.inf'. 
#
# Each persistence diagram in the list is an array of size (N, 2) where N is the number of homology
# cycles.  The first column has the matrix entry value at which the homology cycle represented
# by that row is "born" and the second column has the matrix entry value at which the homology cycle "dies".
# ________________________________________________________________
distances_DGMs = matrix_2_persistence_diagrams(distances_order_matrix)
correlations_DGMs = matrix_2_persistence_diagrams(correlations_order_matrix)
random_iid_DGMs = matrix_2_persistence_diagrams(random_iid_order_matrix)
# _______________________________________________________________
#(11) Plot persistence diagrams for the three order matrices.
# ________________________________________________________________
max_dim = len(distances_DGMs) 
colors = ['black','blue','red','green']
fig3, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12,4))
for i in range(max_dim):
    ax[0].plot([np.min(distances_order_matrix),np.max(distances_order_matrix)],[np.min(distances_order_matrix),np.max(distances_order_matrix)], color = 'grey')
    ax[0].scatter(distances_DGMs[i][:,0],distances_DGMs[i][:,1],c=colors[i],s=5)
    #
    ax[1].plot([np.min(correlations_order_matrix),np.max(correlations_order_matrix)],[np.min(correlations_order_matrix),np.max(correlations_order_matrix)], color = 'grey')
    ax[1].scatter(correlations_DGMs[i][:,0],correlations_DGMs[i][:,1],c=colors[i],s=5)
    #
    ax[2].plot([np.min(random_iid_order_matrix),np.max(random_iid_order_matrix)],[np.min(random_iid_order_matrix),np.max(random_iid_order_matrix)], color = 'grey')
    ax[2].scatter(random_iid_DGMs[i][:,0],random_iid_DGMs[i][:,1],c=colors[i],s=5)
#
ax[0].set_title('distances')
#
ax[1].set_title('correlations')
#
ax[2].set_title('random iid')
#
plt.suptitle('pycliquetop_script_3.py: matrix_2_persistence_diagrams() --- n = %d, dim = %d' % (n,geometric_dim))
#plt.show()
# ________________________________________________________________
# (12) Calling 'persistence_diagrams_2_betti_curves()' on a list of persistence diagrams
# returns Betti curves.  It requires as an additional input 'num_bins'. 
# This is number of edges added in the order complex filtration, equivalently, entries   
# in the upper triangle of the input matrix.  
# For a symmetric matrix of size (n,n), this is given by (n**2 - n)/2) 
# ________________________________________________________________
num_bins = int((n**2 - n)/2) 
# 
[distances_BCs, distances_EDs] = persistence_diagrams_2_betti_curves(distances_DGMs, num_bins)
[correlations_BCs, correlations_EDs] = persistence_diagrams_2_betti_curves(correlations_DGMs, num_bins)
[random_iid_BCs, random_iid_EDs] = persistence_diagrams_2_betti_curves(random_iid_DGMs, num_bins)
# ________________________________________________________________
# (13) Plot the Betti curves by calling 'plot_betti_curves()'.  
# These should match the Betti curves output in step (7) from when we directly called 'compute_betti_curves()'! 
# ________________________________________________________________
colors = ['black','blue','red','green']
fig4, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12,4))
ax[0] = plot_betti_curves(ax[0], distances_BCs, distances_EDs, colors, title_string = 'distances')
ax[1] = plot_betti_curves(ax[1], correlations_BCs, correlations_EDs, colors, title_string = 'correlations')
ax[2] = plot_betti_curves(ax[2], random_iid_BCs, random_iid_EDs, colors, title_string = 'random iid')
plt.suptitle('pycliquetop_script_3.py: persistence_diagrams_2_betti_curves() --- n = %d, dim = %d' % (n,geometric_dim))
# ________________________________________________________________
# (14) Show all of the plots generated by this script. 
# ________________________________________________________________
plt.show()

