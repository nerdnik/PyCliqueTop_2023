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
from matrix_2_betti_curves import matrix_2_betti_curves
from matrix_2_order_matrix import matrix_2_order_matrix
from matrix_2_persistence_diagrams import matrix_2_persistence_diagrams
from persistence_diagrams_2_betti_curves import persistence_diagrams_2_betti_curves
from plot_betti_curves import plot_betti_curves

'''
# *** NOTICE:
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

# NEED TO WRITE SCRIPT PREAMBLE !*!*!*!
# script to compute and plot clique topology 
# to be used AAA BBB CCC 
# calls: XXXXXX YYYYY ZZZZZ 
#
# THIS SCRIPT does BLAH BLAH BLAH
#
# 

#
# ________________________________________________________________
# Set the size of the (n,n) symmetric matrix (equivalently, the number of vertices in the order complex)
# ________________________________________________________________
n = 40 
# ________________________________________________________________
# Make a random symmetric iid matrix of size (n,n)
# ________________________________________________________________
A_random_iid = np.random.uniform(size=(n,n))
A_random_iid = (A_random_iid +  np.transpose(A_random_iid))/2
for i in range(n):
    A_random_iid[i,i]=0
#
# ________________________________________________________________
# Make a uniformly random sample of 'n' points in a 'geometric_dim'-dimensional 
# unit cube in Euclidean space
# ________________________________________________________________
geometric_dim = 3
#
#
xy_coords = np.random.uniform(size=(n,geometric_dim))
# ________________________________________________________________
# Make a random geometric matrix of size (n,n) 
# where entries are the distances between the points sampled 
# uniformly in a 'geometric_dim'-dimensional unit cube in Euclidean space
# ________________________________________________________________
#
A_distances = np.zeros((n,n))
for i in range(n):
    for j in range(i+1,n):
        A_distances[i,j] = A_distances[j,i] = np.linalg.norm((xy_coords[i,:]-xy_coords[j,:]))
#
# ________________________________________________________________
# Make a random correlation matrix of size (n,n)
# where entries are the correlations between points the sampled 
# uniformly in a 'geometric_dim' dimensional unit cube in Euclidean space
#  scipy.spatial.distance.cosine(u, v
# ________________________________________________________________
A_correlations = np.ones((n,n))
for i in range(n):
    for j in range(i+1,n):
        A_correlations[i,j] = A_correlations[j,i] = -1*scipy.spatial.distance.cosine(xy_coords[i,:],xy_coords[j,:]) + 1#scipy.stats.pearsonr(xy_coords[i,:],xy_coords[j,:])[0] 
#   
# ________________________________________________________________
# 
# Call the function 'compute_betti_curves()' to gets Betti curves for
# each of the three symmetric matrices made above: A_distances, A_correlations, A_random_iid.
#
# Notice that for distance matrices we set the input parameter 'similarity' to False, as this is a dissimilarity matrix.
# For correlation matrices we set the input parameter 'similarity' to True, as this is a similarity matrix.
# For random symmetric iid matrices, either option is appropriate. 
# Here we also set the input parameter 'max_dim' to 3, meaning we will compute the persistent homology
# in dimensions 0, 1, 2 and 3 and get a Betti curve for each of these dimensions.
# In particular, the column xxx_betties[:,i] of xxx_bettis is the ith Betti curve.   
#
# ________________________________________________________________
#
[distance_bettis, distance_edge_densities] = compute_betti_curves(A_distances, max_dim = 3, similarity = False)
#
[correlation_bettis, correlation_edge_densities] = compute_betti_curves(A_correlations, max_dim = 3, similarity = True)
#
[random_bettis, random_edge_densities] = compute_betti_curves(A_random_iid, max_dim = 3, similarity = True)
#
# ________________________________________________________________
# 
# Plot the Betti curves by calling 'plot_betti_curves()'.
# This function populates a panel in a figure  that is already created
# before the function call with the specific panel as both an input and the output
# of the function.  
#
# ________________________________________________________________
#
fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12,7))
colors = ['black','blue','red','green']
#
ax[0] = plot_betti_curves(ax[0], distance_bettis, distance_edge_densities, colors, title_string = 'distances')
ax[1] = plot_betti_curves(ax[1], correlation_bettis, correlation_edge_densities, colors, title_string = 'correlations')
ax[2] = plot_betti_curves(ax[2], random_bettis, random_edge_densities, colors, title_string = 'random iid')
#
plt.suptitle('compute_betti_curves() \n n = %d, dim = %d' % (n,geometric_dim))
plt.show()
# ________________________________________________________________
# 
# The function 'compute_betti_curves()' calls on three other functions:
#   matrix_2_order_matrix()
#   matrix_2_persistence_diagrams()
#   persistence_diagrams_2_betti_curves()
#
# These functions can be called to return the intermediate data generated to get Betti curves from matrices.
# ________________________________________________________________
#
# Calling 'matrix_2_order_matrix()' on a matrix returns an integer matrix, 
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
# corresponding to a similarity (ex. correlation) matrix,
# where high matrix values reflect greater similarity between the points, 
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
#
# ________________________________________________________________
#
distances_order_matrix =  matrix_2_order_matrix(A_distances)
correlations_order_matrix =  matrix_2_order_matrix(-1*A_correlations)
random_iid_order_matrix =  matrix_2_order_matrix(A_random_iid)
#
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
plt.suptitle('matrix_2_order_matrix() \n n = %d, dim = %d' % (n,geometric_dim))
plt.show()
# ________________________________________________________________
# 
# Calling 'matrix_2_persistence_diagrams()' on a matrix returns a list of persistence diagrams,
# one for each dimension from 0 through input parameter 'max_dim'. The other input parameter 'thresh'
# determines what value of matrix entries below which to include in the persistent homology
# computations.  By default, these are set to 'max_dim = 3' and 'thresh = np.inf'. 
# Each persistence diagram in the list is an array of size (N, 2) where N is the number of homology
# cycles.  The first column has the matrix entry value at which the homology cycle represented
# by that row is "born" and the second column has the matrix entry value at which the homology cycle "dies".
#
# While 'compute_betti_curves()' and 'matrix_2_betti_curves()' get the persistence diagrams 
# of the order matrix, not directly of the input matrix, here we get the persistence diagrams
# of both to show how the "birth" and "death" values in the persistence diagram reflect the matrix entries.
#
# ________________________________________________________________
#
max_dim = 3 
thresh = np.inf 
#
distances_DGMs = matrix_2_persistence_diagrams(A_distances,max_dim,thresh)
order_distances_DGMs = matrix_2_persistence_diagrams(distances_order_matrix,max_dim,thresh)
#
correlations_DGMs = matrix_2_persistence_diagrams(-1*A_correlations,max_dim,thresh)
order_correlations_DGMs = matrix_2_persistence_diagrams(correlations_order_matrix,max_dim,thresh)
#
random_iid_DGMs = matrix_2_persistence_diagrams(A_random_iid,max_dim,thresh)
order_random_iid_DGMs = matrix_2_persistence_diagrams(random_iid_order_matrix,max_dim,thresh)
#
colors = ['black','blue','red','green']
fig3, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (12,7))
for i in range(max_dim+1):
    ax[0,0].plot([np.min(A_distances),np.max(A_distances)],[np.min(A_distances),np.max(A_distances)], color = 'grey')
    ax[0,0].scatter(distances_DGMs[i][:,0],distances_DGMs[i][:,1],c=colors[i],s=5)
    ax[1,0].plot([np.min(distances_order_matrix),np.max(distances_order_matrix)],[np.min(distances_order_matrix),np.max(distances_order_matrix)], color = 'grey')
    ax[1,0].scatter(order_distances_DGMs[i][:,0],order_distances_DGMs[i][:,1],c=colors[i],s=5)
    #
    ax[0,1].plot([np.min(-1*A_correlations),np.max(-1*A_correlations)],[np.min(-1*A_correlations),np.max(-1*A_correlations)], color = 'grey')
    ax[0,1].scatter(correlations_DGMs[i][:,0],correlations_DGMs[i][:,1],c=colors[i],s=5)
    ax[1,1].plot([np.min(correlations_order_matrix),np.max(correlations_order_matrix)],[np.min(correlations_order_matrix),np.max(correlations_order_matrix)], color = 'grey')
    ax[1,1].scatter(order_correlations_DGMs[i][:,0],order_correlations_DGMs[i][:,1],c=colors[i],s=5)
    #
    ax[0,2].plot([np.min(A_random_iid),np.max(A_random_iid)],[np.min(A_random_iid),np.max(A_random_iid)], color = 'grey')
    ax[0,2].scatter(random_iid_DGMs[i][:,0],random_iid_DGMs[i][:,1],c=colors[i],s=5)
    ax[1,2].plot([np.min(random_iid_order_matrix),np.max(random_iid_order_matrix)],[np.min(random_iid_order_matrix),np.max(random_iid_order_matrix)], color = 'grey')
    ax[1,2].scatter(order_random_iid_DGMs[i][:,0],order_random_iid_DGMs[i][:,1],c=colors[i],s=5)
#
ax[0,0].set_title('distances')
ax[1,0].set_title('order distances')
#
ax[0,1].set_title('correlations')
ax[1,1].set_title('order correlations')
#
ax[0,2].set_title('random iid')
ax[1,2].set_title('order random iid')
#
plt.suptitle('matrix_2_persistence_diagrams() \n n = %d, dim = %d' % (n,geometric_dim))
plt.show()
#
# ________________________________________________________________
#
# Calling 'persistence_diagrams_2_betti_curves()' on a list of persistence diagrams
# returns Betti curves
#
# ________________________________________________________________
#
# num_bins is the number of entries in the upper triangle 
# of the input matrix for which persistence diagrams were computed 
# - THIS ONLY WORKS FOR THE ORDER COMPLEX !!! OTHERWISE should be the maximum entry of the matrix?
# could it always be that? NO, but should be able to adjust code by adding an extra parameter
# one that is num_bins which is the upper triangle as before, and
# an extra input parameter that is genuinely the maximum value of the matrix (or thresholded value)
#
num_bins = int(np.max(np.max(distances_order_matrix))) #int((n**2 - n)/2) 

# 
[distances_BCs, distances_EDs] = persistence_diagrams_2_betti_curves(distances_DGMs, num_bins)
[order_distances_BCs, order_distances_EDs] = persistence_diagrams_2_betti_curves(order_distances_DGMs, num_bins)
#
[correlations_BCs, correlations_EDs] = persistence_diagrams_2_betti_curves(correlations_DGMs, num_bins)
[order_correlations_BCs, order_correlations_EDs] = persistence_diagrams_2_betti_curves(order_correlations_DGMs, num_bins)
#
[random_iid_BCs, random_iid_EDs] = persistence_diagrams_2_betti_curves(random_iid_DGMs, num_bins)
[order_random_iid_BCs, order_random_iid_EDs] = persistence_diagrams_2_betti_curves(order_random_iid_DGMs, num_bins)
#
#
fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (12,7))
colors = ['black','blue','red','green']
#
ax[0,0] = plot_betti_curves(ax[0,0], distances_BCs, distances_EDs, colors, title_string = 'distances')
ax[1,0] = plot_betti_curves(ax[1,0], order_distances_BCs, order_distances_EDs, colors, title_string = 'order distances')
#
ax[0,1] = plot_betti_curves(ax[0,1], correlations_BCs, correlations_EDs, colors, title_string = 'correlations')
ax[1,1] = plot_betti_curves(ax[1,1], order_correlations_BCs, order_correlations_EDs, colors, title_string = 'order correlations')
#
ax[0,2] = plot_betti_curves(ax[0,2], random_iid_BCs, random_iid_EDs, colors, title_string = 'random iid')
ax[1,2] = plot_betti_curves(ax[1,2], order_random_iid_BCs, order_random_iid_EDs, colors, title_string = 'order random iid')
#
plt.suptitle('persistence_diagrams_2_betti_curves() \n n = %d, dim = %d' % (n,geometric_dim))
plt.show()


#
'''
# NEED TO STILL WRITE OUT ABOUT THIS AND WRITE SCRIPT TO RUN AND PLOT AND COMPARE:
# MAYBE ALSO WORTH SHOWING WHAT HAPPENS WHEN YOU PUT IT IN BACKWARDS AND A LITTLE
# WRITE UP ABOUT WHY DOESN'T MAKE MUCH SENSE THOUGH STILL COULD SHOW DIFFERENCES
# MAYBE WOULD MAKE MORE CONNECTED QUICKEST IF WERE TO CONNECT THIS WAY? MAYBE NOT :/
#
matrix_2_betti_curves()
'''

###############
# SECOND SCRIPT - LOADING CORRELATION MATRIX,
# COMPUTING AND PLOTTING BETTI CURVES

# ____________
# ____________

'''
# This has previously been run and saved
# in the repository.
# It's copied here now so you can cross-check 
# what's loaded or re-save if needed.

A = [[1,7,8,3,8.2,5],[2,5.4,3.6,1.8,4,2.1],[9,4,6,3.3,4.5,5.7],
            [2.9,5.2,3.4,8.6,7.8,9.1],[6.3,3.5,5.9,4.7,3.9,6.2],[7.4,8.6,9.8,5.1,8.3,3.7]]
mdict = {}
mdict['correlations'] = A
scipy.io.savemat('tester_correlations_1.mat', mdict)
'''
#
#
mdict = scipy.io.loadmat('tester_correlations_1.mat',simplify_cells=True)
A = mdict['correlations']
[betti_curves, edge_densities] = compute_betti_curves(A,max_dim=3,similarity=True)
#
#
colors = ['black', 'blue','red','green']
title_str = 'tester_correlations_1'
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(12,7))
ax[0].imshow(A,cmap='jet')
ax[1] = plot_betti_curves(ax[1], betti_curves, edge_densities, colors, title_string = '%s' % title_str)
plt.suptitle('tester correlations 1')
plt.show()
###############
