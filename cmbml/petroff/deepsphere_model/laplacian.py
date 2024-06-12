# From Petroff Code
# Available: https://zenodo.org/records/3963028
# Which is based on Defferard 2016 and 2020
"""
Based on DeepSphere (https://github.com/SwissDataScienceCenter/DeepSphere),
`deepsphere/utils.py` file, commit 262573f12c8a7eac058ac85f520401da77b380af.

Copyright (c) 2018 Nathanaël Perraudin, Michaël Defferrard
Copyright (c) 2019-2020 Matthew Petroff

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# JAmato Changes (2024): converted global variables to kernel_widths.py
# Put copied source code into laplacian_torch.py
# Modified for clarity
import time
import numpy as np
import healpy as hp
import scipy

from .laplacian_torch import prepare_laplacian
from .kernel_widths import kernel_width_optimal

from scipy import sparse
import numpy as np
import torch


# def prepare_laplacian(laplacian):
#     """Prepare a graph Laplacian to be fed to a graph convolutional layer.
#     """

#     def estimate_lmax(laplacian, tol=5e-3):
#         """Estimate the largest eigenvalue of an operator.
#         """
#         lmax = sparse.linalg.eigsh(laplacian, 
#                                    k=1, 
#                                    tol=tol, 
#                                    ncv=min(laplacian.shape[0], 10), 
#                                    return_eigenvectors=False)
#         lmax = lmax[0]
#         lmax *= 1 + 2 * tol  # Be robust to errors.
#         return lmax

#     def scale_operator(L, lmax, scale=1):
#         # TODO I'm not sure I like this as much as I like Kipf 2016's solution.
#         #       This forces the scale to the range [-1,1].
#         #       Petroff's version modified this to [-0.75, 0.75]
#         #           (Found in code, not in the paper)
#         #           (See Petroff's layers.py, line 168)
#         #           (Is 0.75 used because it's ~ 1/sqrt(2) ?)
#         #       Kipf's seems more ... rooted in some principle
#         #           (TODO: Research, verify)
#         """Scale the eigenvalues from [0, lmax] to [-scale, scale].
#         """
#         I = sparse.identity(L.shape[0], format=L.format, dtype=L.dtype)
#         L *= 2 * scale / lmax
#         L -= I
#         return L

#     lmax = estimate_lmax(laplacian)
#     laplacian = scale_operator(laplacian, lmax)
#     laplacian = scipy_csr_to_sparse_tensor(laplacian)
#     return laplacian


# def scipy_csr_to_sparse_tensor(csr_mat):
#     """Convert scipy csr to sparse pytorch tensor.

#     Args:
#         csr_mat (csr_matrix): The sparse scipy matrix.

#     Returns:
#         sparse_tensor :obj:`torch.sparse.FloatTensor`: The sparse torch matrix.
#     """
#     coo = sparse.coo_matrix(csr_mat)
#     values = coo.data
#     indices = np.vstack((coo.row, coo.col))
#     idx = torch.LongTensor(indices)
#     vals = torch.FloatTensor(values)
#     shape = coo.shape
#     sparse_tensor = torch.sparse.FloatTensor(idx, vals, torch.Size(shape))
#     sparse_tensor = sparse_tensor.coalesce()
#     return sparse_tensor


def compute_healpix_weightmatrix(nside=16, dtype=np.float32, nest=True):
    """Return an unnormalized weight matrix for a graph using the HEALPIX sampling of the full sphere.

    Parameters
    ----------
    nside : int
        The healpix nside parameter, must be a power of 2, less than 2**30.
    nest : bool, optional
        if True, assume NESTED pixel ordering, otherwise, RING pixel ordering.
    dtype : data-type, optional
        The desired data type of the weight matrix.
    """
    if not nest:
        raise NotImplementedError("Only NESTED pixel ordering is implemented.")

    if nside < 1 or nside & (nside - 1) != 0 or nside >= 2**30:
        raise ValueError("nside must be a power of 2 and less than 2**30.")

    npix = hp.nside2npix(nside)
    indexes = np.arange(npix)

    # Get the coordinates of each pixel in 3D space.
    x, y, z = hp.pix2vec(nside, indexes, nest=nest)
    coords = np.vstack([x, y, z]).transpose().astype(dtype)

    # Get the 7-8 neighbors for each pixel.
    neighbors = hp.pixelfunc.get_all_neighbours(nside, indexes, nest=nest)
    col_index = neighbors.T.reshape(npix * 8)
    row_index = np.repeat(indexes, 8)

    assert col_index.all() < npix, "This check was part of the definition of keep in the source code. It should be an error."
    # Probably, anyways.

    # get_all_neighbors always returns an 8-tuple. When there are 7 neighbors, it includes -1 for the eight missing neighbor
    keep = col_index >= 0
    col_index, row_index = col_index[keep], row_index[keep]

    # Compute Euclidean distances between all neighboring pixels.
    distances = np.linalg.norm(coords[row_index] - coords[col_index], axis=1) ** 2

    # Compute similarities / edge weights. Per Defferard 2020, equation (5) 
    kernel_width = kernel_width_optimal(nside) ** 2 / (4 * np.log(2))
    weights = np.exp(-distances / (4 * kernel_width))

    W = scipy.sparse.csr_matrix((weights, (row_index, col_index)), shape=(npix, npix), dtype=dtype)
    return W


def compute_degree_matrix(W, dtype=np.float32):
    """
    Compute the degree matrix for a graph represented by the adjacency matrix W.
    
    Parameters:
    - W (scipy.sparse matrix): The adjacency matrix of the graph, expected to be a sparse matrix.
    - dtype (data type): The desired data type for the degree matrix.
    
    Returns:
    - scipy.sparse matrix: The degree matrix in sparse diagonal format.
    """
    # Assuming W.sum(axis=1) returns a matrix where each row contains the sum of the corresponding row in W
    D = W.sum(axis=1)
    # Flatten degree
    D = D.A1
    # Convert to sparse matrix
    D = scipy.sparse.diags(D, 0, dtype=dtype)
    return D


def build_laplacian(W, lap_type="normalized", dtype=np.float32):
    """
    Build a Laplacian matrix from a given weighted adjacency matrix.
    See Defferard, 2016, #Graph Fourier Transform for definitions of the
    combinatorial and normalized Laplacians. More information on both can be found in


    Parameters:
    - W (scipy.sparse matrix): Weighted adjacency matrix of the graph.
    - lap_type (str): Type of Laplacian to construct ('combinatorial' or 'normalized').
    - dtype: Data type of the output matrix.
    
    Returns:
    - scipy.sparse.csc_matrix: The constructed Laplacian matrix in CSC format.
    
    Raises:
    - ValueError: If W is not a scipy sparse matrix.
    - ValueError: If an unknown Laplacian type is specified.
    """
    if not scipy.sparse.issparse(W):
        raise ValueError("Input W must be a scipy sparse matrix.")
    if lap_type not in ["combinatorial", "normalized"]:
        # Catch error before waiting for calculations.
        raise ValueError("Unknown Laplacian type {}".format(lap_type))

    D = compute_degree_matrix(W, dtype=dtype)

    if lap_type == "combinatorial":
        L = D - W
    elif lap_type == "normalized":
        # Use the inverse square root of D
        d = D.diagonal()
        D_sqrt_inv = scipy.sparse.diags(np.power(d, -0.5), 0, format='csc', dtype=dtype)

        # Calculate the normalized Laplacian
        L = scipy.sparse.identity(W.shape[0], dtype=dtype) - D_sqrt_inv @ W @ D_sqrt_inv
    else:
        # Should have been caught before calculations; something must have gone very wrong...
        #    Is your computer in a location with adquate shielding from high energy particles?
        raise ValueError("Unknown Laplacian type {}".format(lap_type))
    return L


def healpix_laplacian(
    nside=16, nest=True, lap_type="normalized", indexes=None, dtype=np.float32
):
    """Build a Healpix Laplacian."""
    W = compute_healpix_weightmatrix(nside=nside, nest=nest, dtype=dtype)
    csr_L = build_laplacian(W, lap_type=lap_type, dtype=dtype)
    L = prepare_laplacian(csr_L)
    return L
