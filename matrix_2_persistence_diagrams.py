import ripser as rp
import numpy as np
import scipy.io

# Written August 28, 2023 by Nikki Sanderson
#
# THIS FUNCTION takes a matrix and uses the python package Ripser
# to compute the persistence diagram of Rips complex (clique complex)
# corresponding to the matrix.
#
# SYNTAX:
# matrix_2_persistence_diagrams(matrix,maxdim=3,thresh=np.inf)
#
# INPUTS:
# matrix - numpy array, (symmetric), assumed to be a distance/dissimilarity matrix where low values means high similarity 
# maxdim - integer, (default = 3), the maximum dimension of persistent homology to compute
# thresh - real value or np.inf, (default = np.inf), the value at which to stop the Rips complex construction.
# All entries of the input matrix above this value will not be added as edges in the Rips complex.  
# When this is set to np.inf ("infinity") the Rips complex terminates with all edges between pairs of data points added. 
#
# RETURNS:
# DGMS - list, length "maxdim" list of of arrays, [A_0, A_1,...A_{max_dim}] where the ith array corresponds to the (i-1)st persistent homology (ex. the first array, DGMS[0], is the 0th persistent homology)
# The arrays are of shape (n_pts, 2) where n_pts is the number of persistent homology cycles in that persistent homology dimension 
# and the first column, A_k[:,0], is the input matrix value at which the homological cycle is "born" 
# and the second column, A_k[:,1],  is the input matrix value at which the homological cycle "dies"

def matrix_2_persistence_diagrams(matrix,maxdim=3,thresh=np.inf):
    dgms = rp.ripser(matrix, distance_matrix=True, maxdim = maxdim, thresh=thresh)['dgms']
    DGMS = []
    n_dgms = len(dgms)
    for dim in range(n_dgms):
        dgm = dgms[dim]
        n_pts = len(dgm)
        dg = np.zeros((n_pts,2))
        for k in range(n_pts):
            dg[k,:] = [dgm[k][0],dgm[k][1]]
        DGMS.append(dg)
    return DGMS