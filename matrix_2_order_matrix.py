import ripser as rp
import numpy as np
import scipy.io
from scipy.stats import rankdata

# Written on August 28, 2023 by Nikki Sanderson

# Takes a symmetric matrix and returns the corresponding
# integer matrix whose entries are the rank ordering 
# of the input matrix entries.
#
# SYNTAX :
# matrix_2_order_matrix(Matrix)
#
# INPUT:
# Matrix - numpy array, symmetric matrix
#
# RETURNS:
# OrderMatrix - numpy array, integer symmetric matrix with zeros on the diagonal 

def matrix_2_order_matrix(Matrix):


    num_samples = np.shape(Matrix)[0]
    # Curto et al use Order Complex: idea, order the entries of Cov matrix from highest being first to lowest being last
    OrderMatrix = np.zeros((num_samples,num_samples))

    entry_list = []
    for i in range(num_samples):
        for j in range(i+1,num_samples):
            entry_list.append(Matrix[i,j])
    ranked_entries = rankdata(entry_list,method='min') # This scipy function ranks entries from lowest to highest, with the lowest entry having a rank of 1
    # The method 'min' means that all tied entries in the matrix will be given the lowest rank that would have been assigned should the ties have been broken at random

    # This breaks ties in the matrix randomly 
    while len(np.unique(ranked_entries)) < len(ranked_entries):
        noise_list = np.random.uniform(low=-0.001, high=0.001,size=(np.shape(entry_list))) # ASK: should the low/high be adaptive based on matrix size?
        entry_list = ranked_entries + noise_list
        ranked_entries = rankdata(entry_list,method='min')
        
    k = 0
    for i in range(num_samples):
        for j in range(i+1,num_samples):
            OrderMatrix[i,j] = OrderMatrix[j,i] = ranked_entries[k]
            k=k+1
    return OrderMatrix


