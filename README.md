# PyCliqueTop_2023
Written on August 28, 2023 by Nikki Sanderson

PyCliqueTop_2023 is a Python version of the MATLAB CliqueTop package by Chad Giusti corresponding to the 2015 PNAS paper "Clique topology reveals intrinsic geometric structure in neural correlations". 

Persisent homology (previously computed using the software Perseus) is now computed using the python package Ripser (https://ripser.scikit-tda.org). 

To run the functions in PyCliqueTop_2023, you'll need first install Ripser. To install Ripser, you'll need Cython. To install both of these, run the following in your terminal:

pip install Cython

pip install Ripser

PyCliqueTop_2023 computes Betti curves of clique complexs for an input similarity (ex. correlation) or dissimilarity (ex. distance) matrix.  The main functions are:

* compute_betti_curves() 

* matrix_2_order_matrix() 

* matrix_2_persistence_diagrams() 

* persistence_diagrams_2_betti_curves() 

* plot_betti_curves()

The wrapper function 'compute_betti_curves()' takes as input a symmetric matrix and outputs Betti curves.  It calls on the functions (1) 'matrix_2_order_matrix()', (2) 'matrix_2_persistence_diagrams()', and (3) 'persistence_diagrams_2_betti_curves()' internally. It also has two optional input parameters: 'max_dim' and 'similarity'.  The first optional parameter 'max_dim' allows the user to set a maximum homological dimension up to which to compute the persistent homology (default 'max_dim' = 3).  The second optional parameter 'similarity' is a boolean that allows the user to specify whether the input matrix is a similarity (ex. correlation) or dissimilarity (ex. distance) matrix (default 'similarity' = True). 

Dependencies: Python3, numpy, scipy, scipy.io, scipy.stats, matplotlib, Ripser
