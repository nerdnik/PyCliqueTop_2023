import ripser as rp
import numpy as np
import scipy.io
from scipy.stats import rankdata
from pyCliqueTop_2023_apr_2_2023 import matrix_2_order_matrix
from pyCliqueTop_2023_apr_2_2023 import matrix_2_persistence_diagrams
from pyCliqueTop_2023_apr_2_2023 import persistence_diagrams_2_betti_curves
import matplotlib.pyplot as plt

# Written on August 28, 2023 by Nikki Sanderson
#
# THIS FUNCTION takes a matrix
# and computes the Betti curves for the order complex
# corresponding to this matrix.  It assumes that the matrix is 
# a dissimilarity matrix where smaller matrix entries reflect greater similarity
# and therefore smaller matrix entries represent edges added earlier in the
# filtration to the order complex
#
# INPUTS:
#  matrix - numpy array, a symmetric dissimiliarity (distance) matrix
#  max_dim - integer, (default 3), the maximum dimension of persistent homology to compute 
#  thresh - thresh - real value or np.inf, (default = np.inf), the value at which to stop the Rips complex construction.
# All entries of the input matrix above this value will not be added as edges in the Rips complex.  
# When this is set to np.inf ("infinity") the Rips complex terminates with all edges between pairs of data points added. 
#
# RETURNS:
#[BCs, edge_densities] 
#
#  BCs - numpy array, size (N,max_dim+1), where the ith column is the 
# (i-1)-dimensional Betti curve (ex. the first column, BCs[:,0], is Betti-0)
#  edge_densities - list (CHECK!), length N, the fraction of edges of the complete graph
# included in the graph at each step of the sequence used to make the order complex
# 


def matrix_2_betti_curves(matrix,max_dim=3,thresh=np.inf):
    order_matrix = matrix_2_order_matrix.matrix_2_order_matrix(matrix) 
    DGMS = matrix_2_persistence_diagrams.matrix_2_persistence_diagrams(order_matrix,max_dim,thresh) 
    num_bins = int(np.shape(order_matrix)[0]*(np.shape(order_matrix)[0]-1)/2) # Nikki 8/29/23: removed a + 1 to get rid of incongruence with 'compute_betti_curves()' function, MUST CHECK THIS WAS CORRECT DIRECTION OF CHANGE! 
    [BCS,edge_densities] = persistence_diagrams_2_betti_curves.persistence_diagrams_2_betti_curves(DGMS, num_bins)

    return [BCS,edge_densities] 