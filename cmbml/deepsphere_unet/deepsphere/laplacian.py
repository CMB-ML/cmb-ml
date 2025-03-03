from scipy import sparse
from scipy.sparse import coo_matrix
import numpy as np
import healpy as hp
import torch
import math 

def healpix_weightmatrix(nside=16, nest=True, indexes=None, dtype=np.float32):
    """Return an unnormalized weight matrix for a graph using the HEALPIX sampling.

    Parameters
    ----------
    nside : int
        The healpix nside parameter, must be a power of 2, less than 2**30.
    nest : bool, optional
        if True, assume NESTED pixel ordering, otherwise, RING pixel ordering
    indexes : list of int, optional
        List of indexes to use. This allows to build the graph from a part of
        the sphere only. If None, the default, the whole sphere is used.
    dtype : data-type, optional
        The desired data type of the weight matrix.
    """
    if not nest:
        raise NotImplementedError()

    if indexes is None:
        indexes = range(nside**2 * 12)
    npix = len(indexes)  # Number of pixels.
    if npix >= (max(indexes) + 1):
        # If the user input is not consecutive nodes, we need to use a slower
        # method.
        usefast = True
        indexes = range(npix)
    else:
        usefast = False
        indexes = list(indexes)

    # Get the coordinates.
    x, y, z = hp.pix2vec(nside, indexes, nest=nest)
    coords = np.vstack([x, y, z]).transpose()
    coords = np.asarray(coords, dtype=dtype)
    # Get the 7-8 neighbors.
    neighbors = hp.pixelfunc.get_all_neighbours(nside, indexes, nest=nest)
    col_index = neighbors.T.reshape((npix * 8))
    row_index = np.repeat(indexes, 8)

    # Remove pixels that are out of our indexes of interest (part of sphere).
    if usefast:
        keep = (col_index < npix)
        # Remove fake neighbors (some pixels have less than 8).
        keep &= (col_index >= 0)
        col_index = col_index[keep]
        row_index = row_index[keep]
    else:
        col_index_set = set(indexes)
        keep = [c in col_index_set for c in col_index]
        inverse_map = [np.nan] * (nside**2 * 12)
        for i, index in enumerate(indexes):
            inverse_map[index] = i
        col_index = [inverse_map[el] for el in col_index[keep]]
        row_index = [inverse_map[el] for el in row_index[keep]]

    # Compute Euclidean distances between neighbors.
    distances = np.sum((coords[row_index] - coords[col_index])**2, axis=1)
    # slower: np.linalg.norm(coords[row_index] - coords[col_index], axis=1)**2

    # Compute similarities / edge weights.
    kernel_width = np.mean(distances)
    weights = np.exp(-distances / (2 * kernel_width))

    # Similarity proposed by Renata & Pascal, ICCV 2017.
    # weights = 1 / distances

    # Build the sparse matrix.
    W = sparse.csr_matrix(
        (weights, (row_index, col_index)), shape=(npix, npix), dtype=dtype)

    # if use_4:
    #     W = (W+W.T)/2

    return W


def build_laplacian(W, lap_type='normalized', dtype=np.float32):
    """Build a Laplacian (tensorflow)."""
    d = np.ravel(W.sum(1))
    if lap_type == 'combinatorial':
        D = sparse.diags(d, 0, dtype=dtype)
        return (D - W).tocsc()
    elif lap_type == 'normalized':
        d12 = np.power(d, -0.5)
        D12 = sparse.diags(np.ravel(d12), 0, dtype=dtype).tocsc()
        return sparse.identity(d.shape[0], dtype=dtype) - D12 * W * D12
    else:
        raise ValueError('Unknown Laplacian type {}'.format(lap_type))


def healpix_graph(nside=16,
                  nest=True,
                  lap_type='normalized',
                  indexes=None,
                  use_4=False,
                  dtype=np.float32):
    """Build a healpix graph using the pygsp from NSIDE."""
    from pygsp import graphs

    if indexes is None:
        indexes = range(nside**2 * 12)

    # 1) get the coordinates
    npix = hp.nside2npix(nside)  # number of pixels: 12 * nside**2
    pix = range(npix)
    x, y, z = hp.pix2vec(nside, pix, nest=nest)
    coords = np.vstack([x, y, z]).transpose()[indexes]
    # 2) computing the weight matrix
    if use_4:
        raise NotImplementedError()
        W = build_matrix_4_neighboors(nside, indexes, nest=nest, dtype=dtype)
    else:
        W = healpix_weightmatrix(
            nside=nside, nest=nest, indexes=indexes, dtype=dtype)
    # 3) building the graph
    G = graphs.Graph(
        W,
        lap_type=lap_type,
        coords=coords)
    return G

def scipy_csr_to_sparse_tensor(csr_mat):
    """Convert scipy csr to sparse pytorch tensor.

    Args:
        csr_mat (csr_matrix): The sparse scipy matrix.

    Returns:
        sparse_tensor :obj:`torch.sparse.FloatTensor`: The sparse torch matrix.
    """
    coo = coo_matrix(csr_mat)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    idx = torch.LongTensor(indices)
    vals = torch.FloatTensor(values)
    shape = coo.shape
    sparse_tensor = torch.sparse_coo_tensor(idx, vals, torch.Size(shape), dtype=torch.float32)
    sparse_tensor = sparse_tensor.coalesce()
    return sparse_tensor


def prepare_laplacian(laplacian):
    """Prepare a graph Laplacian to be fed to a graph convolutional layer.
    """

    def estimate_lmax(laplacian, tol=5e-3):
        """Estimate the largest eigenvalue of an operator.
        """
        lmax = sparse.linalg.eigsh(laplacian, k=1, tol=tol, ncv=min(laplacian.shape[0], 10), return_eigenvectors=False)
        lmax = lmax[0]
        lmax *= 1 + 2 * tol  # Be robust to errors.
        return lmax

    def scale_operator(L, lmax, scale=1):
        """Scale the eigenvalues from [0, lmax] to [-scale, scale].
        """
        I = sparse.identity(L.shape[0], format=L.format, dtype=L.dtype)
        L *= 2 * scale / lmax
        L -= I
        return L

    lmax = estimate_lmax(laplacian)
    laplacian = scale_operator(laplacian, lmax)
    laplacian = scipy_csr_to_sparse_tensor(laplacian)
    return laplacian

def resolution_calculator(nodes):
    resolution = int(math.sqrt(nodes / 12))
    return resolution

def get_laplacians(nodes, depth, laplacian_type):
    """Get the healpix laplacian list for a certain depth.
    Args:
        nodes (int): initial number of nodes.
        depth (int): the depth of the UNet.
        laplacian_type ["combinatorial", "normalized"]: the type of the laplacian.
    Returns:
        laps (list): increasing list of laplacians.
    """
    laps = []
    for i in range(depth):
        pixel_num = nodes
        subdivisions = int(resolution_calculator(pixel_num)/2**i)
        G = healpix_graph(subdivisions, lap_type=laplacian_type)
        G.compute_laplacian(laplacian_type)
        laplacian = prepare_laplacian(G.L)
        laps.append(laplacian)
    return laps[::-1]