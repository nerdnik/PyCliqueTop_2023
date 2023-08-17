import ripser as rp
import numpy as np
import scipy.io
from scipy.stats import rankdata


# updated on 10/14/22 by nikki in order to randomize tie-breaking
def matrix_2_order_matrix(Matrix):
    #print('*** new order matrix ****')

    num_samples = np.shape(Matrix)[0]
    # Curto et al use Order Complex: idea, order the entries of Cov matrix from highest being first to lowest being last
    OrderMatrix = np.zeros((num_samples,num_samples))

    entry_list = []
    for i in range(num_samples):
        for j in range(i+1,num_samples):
            entry_list.append(Matrix[i,j])
    ranked_entries = rankdata(entry_list,method='min')

    print('number of unique entries', len(list(np.unique(ranked_entries))))

    while len(np.unique(ranked_entries)) < len(ranked_entries):
        noise_list = np.random.uniform(low=-0.001, high=0.001,size=(np.shape(entry_list))) # QUESTION: should the low/high be adaptive based on matrix size?
        entry_list = ranked_entries + noise_list
        ranked_entries = rankdata(entry_list,method='min')
    
    print('number of unique entries, after', len(list(np.unique(ranked_entries))))
        
    k = 0
    for i in range(num_samples):
        for j in range(i+1,num_samples):
            OrderMatrix[i,j] = OrderMatrix[j,i] = ranked_entries[k]
            k=k+1
    return OrderMatrix


# BELOW WAS COMMENTED OUT BY NIKKI ON 10/24/22 TO BREAK TIES RANDOMLY, NOT USING RANKDATA'S ORDINAL METHOD 
# WHICH IS TOO STRUCTURED AND PRODUCING LOW RANK 
'''
def matrix_2_order_matrix(Matrix):

    num_samples = np.shape(Matrix)[0]
    # Curto et al use Order Complex: idea, order the entries of Cov matrix from highest being first to lowest being last
    OrderMatrix = np.zeros((num_samples,num_samples))
    entry_list = []
    for i in range(num_samples):
        for j in range(i+1,num_samples):
            entry_list.append(Matrix[i,j])
    ranked_entries = rankdata(entry_list,method='ordinal')

    k = 0
    for i in range(num_samples):
        for j in range(i+1,num_samples):
            OrderMatrix[i,j] = OrderMatrix[j,i] = ranked_entries[k]
            k=k+1
    
    return OrderMatrix
'''