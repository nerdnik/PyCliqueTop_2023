# PyCliqueTop_2023
Written on August 28, 2023 by Nikki Sanderson

PyCliqueTop_2023 is a python version of MATLAB CliqueTop package by Chad Giusti corresponding to the 2015 PNAS paper "Clique topology reveals intrinsic geometric structure in neural correlations". 

Persisent homology (previously computed using Perseus) is now computed using the python package Ripser (https://ripser.scikit-tda.org). 

To run the functions in PyCliqueTop_2023, you'll need first install Ripser. To install Ripser, you'll need Cython. To install both of these, run the following in your terminal:

pip install Cython

pip install Ripser

PyCliqueTop_2023 computes Betti curves of clique complexs for an input similarity (ex. correlation) or dissimilarity (ex. distance) matrices.  The main functions are:

* compute_betti_curves() 

* matrix_2_betti_curves()

* matrix_2_order_matrix() 

* matrix_2_persistence_diagrams() 

* persistence_diagrams_2_betti_curves() 

* plot_betti_curves()

Both 'compute_betti_curves()' and 'matrix_2_betti_curves()' take a symmetric matrix as input, call on 'matrix_2_order_matrix()', 'matrix_2_persistence_diagrams()', and 'persistence_diagrams_2_betti_curves()', and return Betti curves. Both 'compute_betti_curves()' and 'matrix_2_betti_curves()' also have an optional input parameter to allow the user to set a maximum homological dimension up to which to compute the persistent homology (default = 3). The main difference between the two functions is the flexibility provided by the additional optional input parameters to these two functions.  While 'compute_betti_curves()' allows the user to specify whether the input matrix is a similarity (ex. correlation) or dissimilarity (ex. distance) matrix (default similarity = True), it computes the Rips complex in its entirety. On the other hand, 'matrix_2_betti_curves()' assumes that the input is a dissimilarity (ex. distance) matrix, yet allows the user to specify a threshold at which to terminate the construction of the Rips complex to control computational demands (default = np.inf).  






