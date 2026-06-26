import numpy as np
from scipy.sparse import csr_matrix

def _nearest_neighbor_indices(
    distances: np.ndarray, 
    k: int, 
    i: int
) -> np.ndarray:
    nn_idx = np.argpartition(distances, k + 1)[:k + 1]
    return nn_idx[nn_idx != i][:k]

def adaptive_sigma2_median(
    distances2: np.ndarray, 
    k: int
) -> float:
    n = distances2.shape[0]
    kth_dists = np.empty(n)

    for i in range(n):
        d2_i = distances2[i]
        nn_idx = _nearest_neighbor_indices(d2_i, k, i)
        kth_dists[i] = d2_i[nn_idx].max()

    return np.median(kth_dists)

def knn_affinity_from_squared_distance_matrix(
    distances2: np.ndarray, 
    k: int, 
    sigma2: float
) -> csr_matrix:
    n = distances2.shape[0]
    nnz = n * k

    index_dtype = np.int32 if nnz <= np.iinfo(np.int32).max else np.int64
    indptr = np.arange(n + 1, dtype=index_dtype) * k
    indices = np.empty(nnz, dtype=index_dtype)
    data = np.empty(nnz, dtype=np.float64)

    for i in range(n):
        d2_i = distances2[i]  # one memmap row
        nn_idx = _nearest_neighbor_indices(d2_i, k, i)
        assert nn_idx.shape[0] == k, f"row {i}: expected {k} neighbors, got {nn_idx.shape[0]}"

        s = i * k
        nn_idx = np.sort(nn_idx)
        indices[s:s + k] = nn_idx
        data[s:s + k] = np.exp(-d2_i[nn_idx] / (2.0 * sigma2))

    a = csr_matrix((data, indices, indptr), shape=(n, n))
    return a.maximum(a.T)

def thresholded_affinity_from_squared_distance_matrix(
    distances2: np.ndarray, 
    sigma2: float,
    threshold: float = 1e-3
) -> csr_matrix:
    n = distances2.shape[0]

    indptr = np.zeros(n+1, dtype=np.int64)
    indices = []
    data = []

    for i in range(n):
        d2_i = distances2[i]  # one memmap row
        affinity_i = np.exp(-d2_i / (2*sigma2))
        significant = np.argwhere(affinity_i > threshold)[:,0]
        indices.append(significant)
        data.append(affinity_i[significant])
        indptr[i+1] = indptr[i] + len(significant)

    a = csr_matrix(
        (np.concat(data), np.concat(indices), indptr), 
        shape=(n, n)
    )
    return a.maximum(a.T)