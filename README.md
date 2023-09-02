# PyCliqueTop_2023
Python version of MATLAB CliqueTop package by Chad Giusti PNAS 2015.

Persisent homology (previously computed using Perseus) is now computed 
using the python package Ripser.  

Computes Betti curves from similarity (ex. correlation) or dissimilarity (ex. distance) matrices. The main functions are:

* (1) compute_betti_curves() 

* (2) matrix_2_betti_curves()

(3) matrix_2_order_matrix() 

(4) matrix_2_persistence_diagrams() 

(5) persistence_diagrams_2_betti_curves() 

(6) plot_betti_curves()

* Both (1) compute_betti_curves() and (2) matrix_2_betti_curves() take a symmetric matrix as input, call on (3), (4), and (5), and return Betti curves. The main difference is the flexibility provided by the optional input parameters to these two functions.  (1) compute_betti_curves() allows the user to specify whether the input matrix is a similarity (ex. correlation) or dissimilarity (ex. distance) matrix yet computes the Rips complex in its entirety. (2) matrix_2_betti_curves() assumes that the input is a dissimilarity (ex. distance) matrix yet allows the user to specify a threshold at which to terminate the construction of the Rips complex, alleviating computational demands.  Both (1) and (2) allow the user to set a maximum homological dimension up to which to compute the persistent homology.






