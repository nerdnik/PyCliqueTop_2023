import ripser as rp
import numpy as np
import scipy.io

# Written on August 28, 2023 by Nikki Sanderson

# THIS FUNCTION takes persistence diagrams 
# and computes Betti curves from them by summing the
# number of persistent homology cycles that are "alive"
# at each step of the filtration.  
#
# SYNTAX:
#  persistence_diagrams_2_betti_curves(DGMS, num_bins)
#
# INPUTS:
#  DGMS - list of arrays, length is the maximum dimension of persistent homology computed plus one, 
# [A_0, A_1,...A_{max_dim}] where the ith array corresponds to the (i-1)st persistent homology (ex. the first array, DGMS[0], is the 0th persistent homology)
# The arrays are of shape (n_pts, 2) where n_pts is the number of persistent homology cycles in that persistent homology dimension 
# and the first column, A_k[:,0], is the input matrix value at which the homological cycle is "born" 
# and the second column, A_k[:,1],  is the input matrix value at which the homological cycle "dies"
#
#  num_bins - integer, the number of edges added in the filtration that produced the persistence diagrams DGMS.
# When all possible edges are added in the filtration for a symmetric matrix, this is given by "int(np.shape(order_matrix)[0]*(np.shape(order_matrix)[0]-1)/2)".
#
# RETURNS:
#[BCs, edge_densities] 
#
#  BCs - numpy array, size (N,max_dim+1), where the ith column is the 
# (i-1)-dimensional Betti curve (ex. the first column, BCs[:,0], is Betti-0)
#  edge_densities - list (CHECK!), length N, the fraction of edges of the complete graph
# included in the graph at each step of the sequence used to make the order complex
#
#
# Edited Nikki 10/9/22 - note: because using <= on line 26, 
# we are including points that die "at" a value in the sum of points that are "alive" at that value
#

def persistence_diagrams_2_betti_curves(DGMS, num_bins):

    BCS = np.zeros((int(num_bins+1),len(DGMS)))
    d = 0
    for dgm in DGMS:
        bc = np.zeros(int(num_bins+1))
        dividers = np.linspace(0.0, int(num_bins), int(num_bins+1))
        n_pts = np.shape(dgm)[0]
        for tt in range(len(dividers)-1): 
            living_pts = []
            for pt in range(n_pts):
                if dgm[pt,0]<=dividers[tt] and dividers[tt+1]<=dgm[pt,1]:
                    living_pts.append(1)
            living_points_sum = sum(living_pts)
            bc[tt] = living_points_sum 
        unbounded =0 
        for pt in range(n_pts):
            if dgm[pt,1]==np.inf:
                unbounded = unbounded+1
        bc[-1]= unbounded
        BCS[:,d] = bc
        d = d+1
    edge_densities = np.linspace(0,1,num_bins+1) 
    

    return [BCS,edge_densities]
