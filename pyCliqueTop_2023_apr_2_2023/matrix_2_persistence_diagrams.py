import ripser as rp
import numpy as np
import scipy.io

def matrix_2_persistence_diagrams(matrix,maxdim=3,thresh=np.inf): # 4/2/23: Nikki set defaults
    dgms = rp.ripser(matrix, distance_matrix=True, maxdim = maxdim, thresh=thresh)['dgms']

    # return len(maxdim) list of (n_pts, 2) arrays for ease of saving
    DGMS = []
    n_dgms = len(dgms)
    for dim in range(n_dgms):
        dgm = dgms[dim]
        n_pts = len(dgm)
        dg = np.zeros((n_pts,2))
        for k in range(n_pts):
            dg[k,:] = [dgm[k][0],dgm[k][1]]
        DGMS.append(dg)
    return DGMS