import ripser as rp
import numpy as np
import scipy.io
import matplotlib.pyplot as plt 
from plot_betti_curves import plot_betti_curves
from compute_clique_topology import compute_clique_topology

# script to compute and plot clique topology for small matrices,
# to be used AAA BBB CCC 
# calls: XXXXXX YYYYY ZZZZZ 

# FIRST, create or load an nxn matrix .....................................
n = 25
d = 10
A1 = np.random.uniform(0,1,(n,n))
print('A1',np.shape(A1))

A2 = [[0, 1, 2, 3, 4, 5], [0, 0, 6, 7, 8, 9], [0, 0, 0, 10, 11, 12], [0, 0, 0, 0, 13, 14], [0, 0, 0, 0, 0, 15], [0, 0, 0, 0, 0, 0]]

A3 = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]

#choose A
A = A1

#symmetrize and put zeros on diagonal to get M
M = 0.5*(A + np.transpose(A)) - np.diag(np.diag(A))

# SECOND, compute clique topology of M - including betti 0 ................
#rho = 1; # max edge density in (0,1]
rho = 1 #0.6

# load pairwise correlation matrices for assemblies in a data set
# ** match the function 
[bettiCurves, edgeDensities, persistenceIntervals, unboundedIntervals] = compute_clique_topology(M, MaxEdgeDensity = rho, ComputeBetti0 =True, MaxBettiNumber=3)#(M,'MaxEdgeDensity',rho,'ComputeBetti0',true,'MaxBettiNumber',2);

print('eD:', len(edgeDensities), edgeDensities)
print('bC:', np.shape(bettiCurves), bettiCurves)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,7))
# plot matrix
ax[0].imshow(M)
# plot bettis
colors = ('black','blue','red','green')
title_string = 'test1'
ax[1] = plot_betti_curves(ax[1], bettiCurves, edgeDensities, colors, title_string)
ax[2] = plot_betti_curves(ax[2], persistenceIntervals, edgeDensities, colors, title_string)
plt.show()

