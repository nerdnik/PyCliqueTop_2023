import ripser as rp
import numpy as np
import scipy.io
from scipy.stats import rankdata
from matrix_2_order_matrix import matrix_2_order_matrix
from matrix_2_persistence_diagrams import matrix_2_persistence_diagrams
from persistence_diagrams_2_betti_curves import persistence_diagrams_2_betti_curves
from persistence_diagrams_2_lifetime_curves import persistence_diagrams_2_lifetime_curves

# function [bettiCurves, edgeDensities, persistenceIntervals, ...
# unboundedIntervals] = compute_clique_topology ( inputMatrix, varargin )
#
# --------------------------------------------------------------------------
# PyCOMPUTE CLIQUE TOPOLOGY
# re-written by Nikki Sanderson, 3/2023 as a python update
# to COMPUTE CLIQUE TOPOLOGY by Chad Giusti, 9/2014
#
# Takes as input a real symmetric matrix, 
# constructs its order complex, a sequence of graphs
# with edges added in increasing order of the input matrix entries
# and simplices added at the emergence of the clique on its' vertices,
# using Ripser to compute the persistent homology of the resulting 
# filtration of clique complexes from the order matrix. Returns
# both the Betti curves and the distribution of persistence lifetimes
# for the order complex of the imput matrix.
# 
# SYNTAX:
#  compute_clique_topology( inputMatrix )
#  compute_clique_topology (inputMatrix, 'ParameterName', param, ... )
# 
# INPUTS:
#   inputMatrix: an NxN symmetric matrix with real coefficients
# OPTIONAL PARAMETERS:
#   'ReportProgress': displays status and time elapsed in each stage
#           as a computation progresses (default: False)
#   'MaxBettiNumber': positive integer specifying maximum Betti
#       number to compute (default: 3)
#   'MaxEdgeDensity': maximum graph density to include in the 
#       order complex in range (0,1] (default: 0.6)
#   'FilePrefix': prefix for intermediate computation files,
#       useful for multipe simultaneous jobs (default: 'matrix')
#   'ComputeBetti0': boolean flag for keeping Betti 0
#       computations; this shifts the indexing of the 
#       outputs so that the nth column representes Betti (n-1). 
#          (default: False)
#   'KeepFiles': boolean flag indicating whether to keep intermediate
#       files when the computation is complete (default: False)
#
#   'WorkDirectory': directory in which to keep intermediate 
#       files during computation (default: current directory, '.')
#   'BaseDirectory': location of the CliqueTop python files
#       (default: detected by XX-which('compute_clique_topology')
#   '
#  
#
# REMOVED OPTIONAL PARAMETERS:
# 'WriteMaximalCliques' - no longer necessary since using Ripser 
#                           instead of Perseus
# 
# 'XXYYZZ' - whatever else I think might be nice to add and is helpful
#
# OUTPUTS:
#   bettiCurves: rectangular array of size
#       maxHomDim x floor(maxGraphDensity * (N choose 2))
#       whose rows are the Betti curves B_1 ... B_maxHomDim
#       across the order complex
#   edgeDensities: the edge densities of the graphs in the
#       order complex, useful for x-axis labels when plotting
#   persistenceIntervals: rectangular array of size
#       maxHomDim x floor(maxGraphDensity * (N choose 2))
#       whose rows are counts of the persistence lifetimes
#       in each homological dimension.
#   unboundedIntervals: vectors of length maxHomDIm whose
#       entries are the number of unbounded persistence intervals
#       for each dimension.  Here, unbounded should be interpreted as 
#       meaning that the cycle dissapears after maxGraphDensity
#       as all cycles dissapear by density 1.
#
# --------------------------------------------------------------------------
## ''' would still need to add these things
# defaultThreads = 1;
# functionLocation = which('compute_clique_topology');
# defaultBaseDirectory = fileparts(functionLocation);
#--------------------------------------------------------------------------
#
#
#
# --------------------------------------------------------------------------
# 
#
def compute_clique_topology(inputMatrix, MaxBettiNumber = 3, MaxEdgeDensity = .6, FilePrefix = 'matrix', WorkingDirectory='.',ComputeBetti0=True,ReportProgress = False):
#
# --------------------------------------------------------------------------
# Validate and set parameters
# --------------------------------------------------------------------------
#
#
#
#
#    defaultReportProgress = False
#    defaultMaxBettiNumber = 3
#    defaultEdgeDensity = .6
#    defaultFilePrefix = 'matrix'
#    dfaultKeepFiles = False
#    defaultWorkDirectory = '.'
#
#
#
#
# --------------------------------------------------------------------------
# Compute order matrix of input matrix 
#         * this zeros out the diagonal and 
#         * orders the entries in the lower triangle and 
#         * fills in the uppder triangle symmetrically
# --------------------------------------------------------------------------
# 
#
    order_matrix = matrix_2_order_matrix(inputMatrix)
#
    num_bins = int(np.shape(order_matrix)[0]*(np.shape(order_matrix)[0]-1)/2)
# 
# ----------------------------------------------------------------
# Move to working directoy and stop if files might be overwritten
# ----------------------------------------------------------------
# 
#   
# 
# ----------------------------------------------------------------
# Use Ripser to compute persistent homology of order matrix
#       * Vietoris Rips filtration with order matrix as distance matrix
# ----------------------------------------------------------------
# 
    thresh = int(np.floor(num_bins*MaxEdgeDensity))
    DGMS = matrix_2_persistence_diagrams(order_matrix,MaxBettiNumber,thresh)
# 
# ----------------------------------------------------------------
# Compute Betti curves and Lifetime curves from the persistent homology
#       * Since the number of pairwise correlations is n(n-1)/2,      
#       * the order complex consists of edges and sothe resolution of
#       * edge density along the x-axis of the betti curves we have
#       * persistent homology information at is 1 divided by that.
#       * In terms of number of equal spaced bins needed to go from 
#       * 0 to 1 edge density of the betti curves, that's num_bins.
# ----------------------------------------------------------------
#
# 
    [bettiCurves,bc_edge_densities] = persistence_diagrams_2_betti_curves(DGMS, num_bins)
# 
# *** NEED TO WRITE CODE TO GET LIFETIME CURVES ****
# 
    [persistenceIntervals, unboundedIntervals, lc_edge_densities] = persistence_diagrams_2_lifetime_curves(DGMS, num_bins)
# 
    edgeDensities = bc_edge_densities
#
# ----------------------------------------------------------------
# Save Betti curves and edge densities to a .mat file if desired
#       * set optional input 'FilePrefix' to change file name
# ----------------------------------------------------------------
# 
#
#
#
#
# ----------------------------------------------------------------
# Save intermediate files if desired
#       * (default) optional input boolean 'KeepFiles' set to False
#       * Set 'KeepFiles' = True to save intermediate files for 
#       * the order matrix and set of persistence diagrams.
# ----------------------------------------------------------------
# 
# 
# 
# 
# 
# ----------------------------------------------------------------
# Return Betti curves, lifetime curve sand edge densities so can be 
# used by whatever script may have called this function.
# ----------------------------------------------------------------
# 
    return bettiCurves, edgeDensities, persistenceIntervals, unboundedIntervals
#
#