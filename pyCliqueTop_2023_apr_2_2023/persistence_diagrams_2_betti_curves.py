import ripser as rp
import numpy as np
import scipy.io
# 4/2/2023 Nikki commented below out to get to run as a module
#from matrix_2_persistence_diagrams import matrix_2_persistence_diagrams
#from plot_betti_curves import plot_betti_curves
#from matrix_2_order_matrix import matrix_2_order_matrix
#from pyCliqueTop_2023_apr_2_2023 
#import matplotlib.pyplot as plt

# LAST EDITED: Nikki 10/9/22 - note: because using <= on line 26, 
# we are including points that die "at" a value in the sum of points that are "alive" at that value

def persistence_diagrams_2_betti_curves(DGMS, num_bins):
    #print('num bins:', num_bins)
    BCS = np.zeros((int(num_bins+1),len(DGMS)))
    d = 0
    for dgm in DGMS:
        bc = np.zeros(int(num_bins+1))# commented out 10/9/22: np.zeros(int(num_bins))#np.zeros(int(num_bins-1))
        dividers = np.linspace(0.0, int(num_bins), int(num_bins+1))
        #print('dividers first,last:', dividers)
        n_pts = np.shape(dgm)[0]
        for tt in range(len(dividers)-1): 
            living_pts = []
            for pt in range(n_pts):
                '''
                print('tt', tt)
                print('dividers[tt], dividers[tt+1]',dividers[tt],dividers[tt+1])
                print('pt:', dgm[pt,:])
                print('')
                '''
                if dgm[pt,0]<=dividers[tt] and dividers[tt+1]<=dgm[pt,1]:
                    living_pts.append(1)
            living_points_sum = sum(living_pts)
            bc[tt] = living_points_sum #sum([1 for pt in range(n_pts) if dgm[pt,0]<=dividers[tt] and dividers[tt+1]<=dgm[pt,1]])
        #bc[-1]=bc[-2] # commented out 10/9/22
        unbounded =0 
        for pt in range(n_pts):
            if dgm[pt,1]==np.inf:
                unbounded = unbounded+1
        bc[-1]= unbounded
        #
        BCS[:,d] = bc
        d = d+1
    edge_densities = np.linspace(0,1,num_bins+1) # EDITED BY NIKKI ON 3/27/23 TO BE num+bins+1 instead of num_bins
    

    return [BCS,edge_densities]

# DEBUGGING 10/9/22: changing the matrix entry that is 6 to 5 shows the closed end point commented at the top of the script
'''
matrix = np.asarray([[0,1,6,2],[1,0,3,6],[6,3,0,4],[2,6,4,0]])
print('matrix', type(matrix), np.shape(matrix), matrix)
#matrix = matrix_2_order_matrix(matrix)
maxdim = 3
DGMS = matrix_2_persistence_diagrams(matrix,maxdim)
print('DGMS', DGMS)
num_bins = int(np.shape(matrix)[0]*(np.shape(matrix)[0]-1)/2)
print('num_bins',num_bins)
[BCs,edge_densities] = persistence_diagrams_2_betti_curves(DGMS, num_bins)
print('edge densities', edge_densities)
print('BCs', BCs)

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,7))
colors = ['black','gold','red','blue']
title_string = ''
ax = plot_betti_curves(ax, BCs, edge_densities, colors, title_string)
plt.show()
'''