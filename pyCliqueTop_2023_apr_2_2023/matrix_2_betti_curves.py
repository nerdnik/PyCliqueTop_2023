import ripser as rp
import numpy as np
import scipy.io
from scipy.stats import rankdata
#from matrix_2_order_matrix import matrix_2_order_matrix
#from matrix_2_persistence_diagrams import matrix_2_persistence_diagrams
#from persistence_diagrams_2_betti_curves import persistence_diagrams_2_betti_curves
from pyCliqueTop_2023_apr_2_2023 import matrix_2_order_matrix
from pyCliqueTop_2023_apr_2_2023 import matrix_2_persistence_diagrams
from pyCliqueTop_2023_apr_2_2023 import persistence_diagrams_2_betti_curves
import matplotlib.pyplot as plt
#from plot_betti_curves import plot_betti_curves
#from shuffle_matrix import shuffle_matrix
#from compute_betti_curves import compute_betti_curves

# Nikki: 2/27/23 - added +1 to num_bins to make edge density correct since starts at 0 and ends at 1, needed to include 
 
def matrix_2_betti_curves(matrix,max_dim=3,thresh=np.inf):
    # matrix = -1*matrix - ** COMMENTED OUT SEPT 2 2022, will need to update scripts and wrapper functions accordingly **
    order_matrix = matrix_2_order_matrix.matrix_2_order_matrix(matrix) # Nikki 4/2/23: changed from just matrix_2_order_matrix(matrix) bc now in a python module
    DGMS = matrix_2_persistence_diagrams.matrix_2_persistence_diagrams(order_matrix,max_dim,thresh) # Nikki 4/2/23: updated bc needed input of thresh
    num_bins = int(np.shape(order_matrix)[0]*(np.shape(order_matrix)[0]-1)/2) + 1
    [BCS,edge_densities] = persistence_diagrams_2_betti_curves.persistence_diagrams_2_betti_curves(DGMS, num_bins)

    return [BCS,edge_densities] 