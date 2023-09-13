import ripser as rp
import numpy as np
import scipy.io
from scipy.stats import rankdata
from matrix_2_order_matrix import matrix_2_order_matrix
from matrix_2_persistence_diagrams import matrix_2_persistence_diagrams
from persistence_diagrams_2_betti_curves import persistence_diagrams_2_betti_curves

# Written August 28, 2023 by Nikki Sanderson
#
# Takes as input a real symmetric matrix, 
# constructs its order complex, a sequence of graphs
# with edges added based on the ordering of the input matrix entries
# and simplices added at the emergence of the clique on its' vertices,
# using Ripser to compute the persistent homology of the resulting 
# filtration of clique complexes from the order matrix. Returns the Betti curves and edge densities.
#
# THIS FUNCTION computes Betti curves for a symmetric matrix
# by calling the additional functions:
# (1) matrix_2_order_matrix()
# (2) matrix_2_persistence_diagrams()
# (3) persistence_diagrams_2_betti_curves()
# It uses the python package Ripser for the persistent homology computations.
#
# SYNTAX: 
# compute_betti_curves(matrix)
# compute_betti_curves(matrix, max_dim = 3, similarity = True)
#
# INPUTS:
# matrix - numpy array, (a symmetric matrix), often pairwise correlations or distances
# max_dim - integer, (default 3), the maximum dimension of persistent homology to compute 
# similarity - boolean, (default True), True means that the matrix entries 
# represent similarities, like correlations, where a higher value means more similar, 
# and False means that the matrix entries represent dissimilarities, like distances, 
# where a higher value means less similar
# 
# RETURNS: 
# [BCs, edge_densities] 
#
# BCs - numpy array, size (N,max_dim+1), where the ith column is the 
# (i-1)-dimensional Betti curve (ex. the first column, BCs[:,0], is Betti-0)
# edge_densities - list (CHECK!), length N, the fraction of edges of the complete graph
# included in the graph at each step of the sequence used to make the order complex
# 

def compute_betti_curves(matrix, max_dim = 3, similarity = True):
    if similarity == True:
        matrix = -1*matrix
    order_matrix = matrix_2_order_matrix(matrix)
    DGMS = matrix_2_persistence_diagrams(order_matrix,max_dim)
    num_bins = int(np.shape(order_matrix)[0]*(np.shape(order_matrix)[0]-1)/2)
    [BCS,edge_densities] = persistence_diagrams_2_betti_curves(DGMS, num_bins)

    return [BCS,edge_densities]

    