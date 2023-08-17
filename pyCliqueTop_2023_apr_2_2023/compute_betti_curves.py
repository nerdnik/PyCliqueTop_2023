import ripser as rp
import numpy as np
import scipy.io
from scipy.stats import rankdata
from matrix_2_order_matrix import matrix_2_order_matrix
from matrix_2_persistence_diagrams import matrix_2_persistence_diagrams
from persistence_diagrams_2_betti_curves import persistence_diagrams_2_betti_curves

def compute_betti_curves(matrix, max_dim = 3, similarity = True):
    order_matrix = matrix_2_order_matrix(matrix)
    DGMS = matrix_2_persistence_diagrams(order_matrix,max_dim)
    num_bins = int(np.shape(order_matrix)[0]*(np.shape(order_matrix)[0]-1)/2)
    [BCS,edge_densities] = persistence_diagrams_2_betti_curves(DGMS, num_bins)

    return [BCS,edge_densities]