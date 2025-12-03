import numpy as np
import scipy
from numba import jit
import time
import scipy.sparse as sparse
from scipy.sparse.linalg import lsqr
from scipy.optimize import LinearConstraint, milp
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.special import comb
from scipy import sparse
from ripser import ripser
import warnings
from dreimac_utils import *
from dreimac_combinatorial import *
import time
import matplotlib.pyplot as plt
import math
from persim import plot_diagrams
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as patches
import plotly.graph_objs as go


def sample_torus(n_points, R, r):
    theta = 2 * np.pi * np.random.random(n_points)
    phi = 2 * np.pi * np.random.random(n_points)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.column_stack((x, y, z))


def rips_filtration_diagram(data: np.array = np.random.normal(size=(50, 3)), \
                            coeff: int = 2, \
                            maxdim: int = 2, \
                            do_cocycles: bool = True, \
                            distance_matrix: bool = True, \
                            n_landmarks: Optional[int] = None, \
                            thresh: Optional[float] = None) -> Dict:
    """
    Computs persistence diagrams and representative cocycles for input data using
    Rips filtration from ripser library
    Args:
      data = either point cloud (if distance_matrix=False) or distance matrix (otherwise)
      coeff = field coefficients used in homology computation
      do_cocycles = True if you want to compute representative cocycles
      distance_matrix = True if input data is distance matrix
      n_land = optional number of landmarks used
      thresh = optional filtration threshold
    Returns:
      rips_result = dictionary containing persistence diagrams and cocycles.
      See https://ripser.scikit-tda.org/en/latest/reference/stubs/ripser.ripser.html#ripser.ripser for details
    """
    begin = time.time()
    if thresh == None:
        rips_result = ripser(data, coeff=coeff, maxdim=maxdim, do_cocycles=do_cocycles, distance_matrix=distance_matrix,
                             n_perm=n_landmarks)
    else:
        rips_result = ripser(data, coeff=coeff, maxdim=maxdim, do_cocycles=do_cocycles, distance_matrix=distance_matrix,
                             thresh=thresh, n_perm=n_landmarks)
    end = time.time()
    print(f'Computed rips filtration in {(end - begin) / 60:.2f} mins')
    return rips_result


def compute_annotated_barcodes(ripser_res: Dict, k: int = 2) -> Tuple[float, float, List]:
    """
    Args:
      ripser_res = resulting dictionary returned by ripser, which contains diagrams and cocycles
      k = cohomology dimension bound
    Returns:
      B_1_k = barcodes annotated by representative cocycles
    """
    # iterate over dims from 1 to k
    annotated_barcodes = []
    dgms = ripser_res['dgms']
    cocycles = ripser_res['cocycles']

    for dim in range(1, k + 1):
        for i in range(dgms[dim].shape[0]):
            cocycle = cocycles[dim][i][:, :-1].tolist()  # we drop the last column which gives the value on the cocycle
            annotated_barcodes.append((dgms[dim][i, 0], dgms[dim][i, 1], cocycle))

    annotated_barcodes = sorted(annotated_barcodes, key=lambda annotated_barcodes: (annotated_barcodes[1], annotated_barcodes[0]))

    return annotated_barcodes




def simplex_exists(new_a: List, distance_matrix: np.array, thresh: float) -> bool:
    """
    checks if simplex new_a exists at filtration time thresh
    """
    edges = create_pairs_(new_a)

    edge_distances = [distance_matrix[edge[0], edge[1]] for edge in edges]   

    # Find the maximum distance
    max_distance = max(edge_distances)

    # Compare with threshold
    if max_distance <= thresh:
        return True
    else:
        return False


def compute_dict_of_birth_times(distance_matrix: np.array) -> Dict:
    """
    compute the birth time of all simplices in the filtration
    """
    birth_times = {}
    for i in range(distance_matrix.shape[0]):
        birth_times[tuple([i])] = 0.0

    int_list = list(range(distance_matrix.shape[0]))
    edges = create_pairs_(int_list)

    for edge in edges:
        birth_times[tuple(edge)] = compute_birth_time(edge, distance_matrix)

    triangles = create_triples_(int_list)
    for triangle in triangles:
        birth_times[tuple(triangle)] = compute_birth_time(triangle, distance_matrix)

    return birth_times

def create_pairs_(numbers: List[int]) -> List[List[int]]:
    # all possible pairs of integers in the original list
    return [[a, b] for i, a in enumerate(numbers) for b in numbers[i + 1:]]


def create_triples_(numbers: List[int]) -> List[List[int]]:
    # all possible triples of integers in the original list
    return [[a, b, c] for i, a in enumerate(numbers)
            for j, b in enumerate(numbers[i + 1:], start=i + 1)
            for c in numbers[j + 1:]]

def count_alive_simplices(distance_matrix: np.array, thresh: float) -> int:
    """
    counts the number of simplices alive at time thresh in input distance matrix distance_matrix
    """
    # get total number of vertices
    num_of_vertices = distance_matrix.shape[0]
    # compute all possible pairs (edges) and triples (triangles) in the filtration
    edges = create_pairs_(range(num_of_vertices))
    triangles = create_triples_(range(num_of_vertices))
    # check if simplex has birth time <= thresh, i.e. its diameter is <= thresh
    count_alive_simplices = 0
    for edge in edges:
        if simplex_exists(edge, distance_matrix, thresh):
            count_alive_simplices += 1

    for triangle in triangles:
        if simplex_exists(triangle, distance_matrix, thresh):
            count_alive_simplices += 1

    return count_alive_simplices

def cup_product(cocycle1_rep: List, cocycle2_rep: List, birth_times_row: np.array, lookup_table: np.ndarray, thresh: float,
                max_dim_X: int = 2) -> List:
    """
    Compute the cup product of two cocycles over Z mod 2 in a filtered simplicial complex.

    Args:
        cocycle1_rep (List): List of simplices representing the first cocycle.
        cocycle2_rep (List): List of simplices representing the second cocycle.
        birth_times_row (np.array): Array of birth times for simplices, indexed by combinatorial number system encoding.
        lookup_table (np.ndarray): Precomputed lookup table used by combinatorial number system.
        thresh (float): Filtration threshold; only cup products existing at this filtration value are retained.
        max_dim_X (int, optional): The dimension of the simplicial complex. Defaults to 2.

    Returns:
        List: A tuple consisting of the cup product and the two cocycles (cocycle1_rep, cocycle2_rep) that generated it.
    """
    size_of_cocycle1 = len(cocycle1_rep)
    size_of_cocycle2 = len(cocycle2_rep)
    p = size_of_cocycle1 + size_of_cocycle2

    # compute the dimension of the cohomology group the cocycle belongs to
    dim_of_cocycle1 = len(cocycle1_rep[0]) - 1
    dim_of_cocycle2 = len(cocycle2_rep[0]) - 1

    sigma = []
    if dim_of_cocycle1 + dim_of_cocycle2 <= max_dim_X:
        for i in range(size_of_cocycle1):
            for j in range(size_of_cocycle2):

                a, b = cocycle1_rep[i], cocycle2_rep[j]
                new_a = a.copy()
                if a[-1] == b[0]:
                    if len(b[1:]) > 0:
                        new_a = new_a + b[1:]
                        new_a_index = combinatorial_number_system_forward(np.array(new_a[::-1]), lookup_table)
                        if birth_times_row[new_a_index] <= thresh:
                            sigma.append(new_a)
    return sigma, cocycle1_rep, cocycle2_rep

def convert_indices(indices, mapping):
    # mapping original indices to subsampled indices. Used in computations involving landmarks
    return [mapping.get(idx, None) for idx in indices if mapping.get(idx, None) is not None]


def convert_annotated_barcodes(annotated_barcodes, original_to_subsampled):
    # Convert indices in annotated_barcodes to indices used in the subsample
    converted_barcodes = []
    for barcode in annotated_barcodes:
        # Unpack the tuple
        first, second, index_lists = barcode

        # Convert each list of indices
        converted_lists = [convert_indices(indices, original_to_subsampled) for indices in index_lists]

        # Create new tuple with converted indices
        converted_barcode = (first, second, converted_lists)
        converted_barcodes.append(converted_barcode)
    return converted_barcodes


def gf2_gaussian_elimination(A, b):
    A = np.array(A, dtype=int) % 2
    b = np.array(b, dtype=int) % 2
    M = np.column_stack((A, b))
    m, n = A.shape

    row = 0
    for col in range(n):
        # Find a row with a leading 1 in the current column
        pivot = None
        for r in range(row, m):
            if M[r, col] == 1:
                pivot = r
                break

        if pivot is None:
            continue  # No pivot found in this column, move to next

        # Swap the current row with the pivot row
        M[[row, pivot]] = M[[pivot, row]]

        # Perform elimination on all rows below
        for r in range(row + 1, m):
            if M[r, col] == 1:
                M[r] ^= M[row]

        row += 1

    # Check for inconsistency (rows where all A columns are 0 but b is 1)
    for i in range(row, m):
        if M[i, :-1].sum() == 0 and M[i, -1] == 1:
            return False, None  # No solution exists

    # Back substitution to find a solution
    x = np.zeros(n, dtype=int)
    for i in range(row - 1, -1, -1):
        leading_col = np.where(M[i, :-1] == 1)[0]
        if len(leading_col) == 0:
            continue
        j = leading_col[0]
        x[j] = M[i, -1] ^ np.dot(M[i, j+1:n], x[j+1:n]) % 2

    # Verify the solution
    if np.array_equal(np.dot(A, x) % 2, b):
        return True, x
    else:
        return False, None

def has_binary_solution(A, b):
    # Check if A and b contain only 0 and 1
    if not np.all(np.logical_or(A == 0, A == 1)) or not np.all(np.logical_or(b == 0, b == 1)):
        raise ValueError("A and b must contain only 0 and 1")

    return gf2_gaussian_elimination(A, b)

def contains_nonzero(arr):
    return np.any(arr != 0)

def convert_sparse_cocycle_to_vector_csr(sparse_cocycle, lookup_table, n_vertices, dtype):
    """
    Convert a sparse cocycle representation to a CSR-format sparse vector.

    Given a list of simplices representing a cocycle over Z mod 2, this function maps each simplex
    to a unique index using a combinatorial number system and constructs a sparse vector
    (CSR matrix) of shape (n_simplices, 1), where each nonzero entry indicates the
    presence of the corresponding simplex in the cocycle.

    Args:
        sparse_cocycle (array-like): A 2D array where each row is a simplex in the cocycle.
        lookup_table (np.ndarray): Lookup table for the combinatorial number system encoder.
        n_vertices (int): Total number of vertices in the ambient simplicial complex.
        dtype (type): Data type for the sparse matrix values (typically np.int8 or np.float32).

    Returns:
        csr_matrix: A sparse column vector in Compressed Sparse Row (CSR) format,
                    representing the cocycle as a binary indicator vector over all simplices
                    of the appropriate dimension.
    """
    dimension = sparse_cocycle.shape[1] - 1
    n_simplices = number_of_simplices_of_dimension(dimension, n_vertices, lookup_table)

    # Collect simplex indices for each entry in the sparse cocycle.
    simplex_indices = []
    for entry in sparse_cocycle:
        unordered_simplex = np.array(entry, dtype=int)
        ordered_simplex, sign = CohomologyUtils.order_simplex(unordered_simplex)
        simplex_index = combinatorial_number_system_forward(ordered_simplex, lookup_table)
        simplex_indices.append(simplex_index)

    # Remove duplicates so that each simplex gets a value of 1 (mimicking cocycle_as_vector[simplex_index] = 1).
    unique_indices = np.unique(simplex_indices)
    data = np.ones(len(unique_indices), dtype=dtype)

    # Build a sparse column vector (shape: n_simplices x 1) in CSR format.
    cocycle_as_vector = csr_matrix((data, (unique_indices, np.zeros(len(unique_indices), dtype=int))),
                                   shape=(n_simplices, 1))
    return cocycle_as_vector

def convert_sparse_cocycle_to_vector(sparse_cocycle, lookup_table, n_vertices, dtype):
    """
    Convert a sparse cocycle representation to a vector representation.

    Given a list of simplices representing a cocycle over Z mod 2, this function maps each simplex
    to a unique index using a combinatorial number system and constructs a vector of shape
    (n_simplices, 1), where each nonzero entry indicates the presence of the corresponding
    simplex in the cocycle.

    Args:
        sparse_cocycle (array-like): A 2D array where each row is a simplex in the cocycle.
        lookup_table (np.ndarray): Lookup table for the combinatorial number system encoder.
        n_vertices (int): Total number of vertices in the ambient simplicial complex.
        dtype (type): Data type for the sparse matrix values (typically np.int8 or np.float32).

    Returns:
        cocycle_as_vector (np.ndarray): A vector representation of the cocycle
    """
    dimension = sparse_cocycle.shape[1] - 1
    n_simplices = number_of_simplices_of_dimension(
        dimension, n_vertices, lookup_table
    )
    cocycle_as_vector = np.zeros((n_simplices,), dtype=dtype)
    for entry in sparse_cocycle:
        unordered_simplex = np.array(entry, dtype=int)
        ordered_simplex, sign = CohomologyUtils.order_simplex(unordered_simplex)
        simplex_index = combinatorial_number_system_forward(
            ordered_simplex, lookup_table
        )
        cocycle_as_vector[simplex_index] = 1
    return cocycle_as_vector

def vector_rep_of_cochain(sparse_cocycle: List, lookup_table: np.array, birth_times_row: np.array, n_vertices: int, thresh: float, output_format: str = 'csr') -> np.array:
    """
    Compute the vector representation of a thresholded sparse cocycle over Z mod 2
    Args:
        sparse_cocycle (List): List of simplices in the cocycle (each simplex is a list of vertex indices).
        lookup_table (np.array): Lookup table for encoding simplices using combinatorial number system.
        birth_times_row (np.array): Array of birth times for simplices
        n_vertices (int): Number of vertices in the simplicial complex.
        thresh (float): Filtration threshold; only simplices born at or before this value are kept.
        output_format (str, optional): Output format of the result. Options are:
            - 'csr' (default): Returns a sparse vector in CSR format.
            - 'dense': Returns a dense NumPy vector.

    Returns:
         The vector representation of the thresholded cocycle, either as a dense NumPy array or sparse CSR matrix.

    Notes:
        - If no simplices survive the thresholding, returns a sparse CSR vector of length 1 with value 0.
    """
    # threshold sparse_cocycle, i.e. only keep simplices that are alive at time = threshold
    thresholded_sparse_cocycle = []
    for simplex in sparse_cocycle:
      simplex_index = combinatorial_number_system_forward(np.array(simplex[::-1]), lookup_table)
      if birth_times_row[simplex_index] <= thresh:
        thresholded_sparse_cocycle.append(simplex)
    if len(thresholded_sparse_cocycle) == 0:
      return csr_matrix(np.zeros(1, dtype=int)) #empty cocycle
    else:
      if output_format == 'csr':
        return convert_sparse_cocycle_to_vector_csr(np.array(thresholded_sparse_cocycle), lookup_table, n_vertices, int)
      else:
        return convert_sparse_cocycle_to_vector(np.array(thresholded_sparse_cocycle), lookup_table, n_vertices, int)


def decode_combinatorial_number(N, k):
    """
    Decode the number N from the combinatorial number system representation
    back to the k-combination (c_1, c_2, ..., c_k) - where c_1 < c_2 < ... < c_k - such that:
    
      N = (c_k choose k) + (c_{k-1} choose k-1) + ... + (c_1 choose 1)
      
    The greedy algorithm finds the unique sequence corresponding to N.
    
    Parameters:
      N (int): The nonnegative integer in the combinatorial number system.
      k (int): The number of elements in the combination. Note that in our case, this is simplex dimension + 1
      
    Returns:
      tuple: A tuple (c_1, c_2, ..., c_k) representing the combination.
    """
    # This will store the decoded digits in order: c_1, c_2, ..., c_k.
    combination = [0] * k
    
    # Process from the highest index k down to 1.
    for i in range(k, 0, -1):
        # Find the maximum c_i (starting from i) such that math.comb(c_i, i) <= N.
        c = i  # The minimum possible value for c_i is i.
        while math.comb(c, i) <= N:
            c += 1
        # When the loop exits, math.comb(c, i) > N, so the correct value is c - 1.
        c -= 1
        combination[i - 1] = c
        # Subtract the contribution of the current digit.
        N -= math.comb(c, i)
    
    return tuple(combination)
  
def compute_threshold_persistence(arr: np.array):
    """
    Compute a persistence-based threshold for selecting top persistent 1-cocycles.

    Given an input array arr of birth-death pairs (each row is [birth, death]) representing
    H_1 (first cohomology) features, this function computes the persistence (death - birth) of
    each feature and returns a threshold value equal to the mean of the second and third largest
    persistences.

    Args:
        arr (np.array): A 2D NumPy array of shape (n, 2), where each row contains a birth and
                        death time of a persistent cohomology class in H^1.

    Returns:
        float: The average of the second and third largest persistence values (death - birth).

    """
    differences = arr[:, 1] - arr[:, 0]
    top3 = np.partition(differences, -3)[-3:]
    top3_sorted = np.sort(top3)[::-1]
    return (top3_sorted[1] + top3_sorted[2]) / 2


def threshold_A_optimized(A: csr_matrix, birth_times_row: np.array, birth_times_col: np.array, threshold: float):
    """
    Efficiently threshold a sparse coboundary matrix by filtration time.

    This function filters a sparse coboundary matrix `A` (in CSR format), where:
    - Rows correspond to 2-simplices (e.g., triangles),
    - Columns correspond to 1-simplices (e.g., edges).

    The function removes columns and rows whose associated simplices are born after
    a given filtration threshold. Additionally, it ensures that retained rows are not
    entirely zero after column filtering (i.e., they must have at least one nonzero entry).

    Args:
        A (csr_matrix): Sparse coboundary matrix in Compressed Sparse Row (CSR) format.
        birth_times_row (np.array): Array of birth times for rows (triangles),
                                    indexed to match row indices of A.
        birth_times_col (np.array): Array of birth times for columns (edges),
                                    indexed to match column indices of A.
        threshold (float): Filtration threshold; simplices born after this value are excluded.

    Returns:
        Tuple[csr_matrix, np.array]:
            - A (csr_matrix): Filtered sparse matrix with only rows and columns alive at or before threshold.
            - row_mask (np.array): Boolean array indicating which rows from the original matrix were retained.

    """

    # 1. Filter columns by birth time threshold
    col_mask = (birth_times_col <= threshold)
    if not col_mask.all():  # Only slice if there are columns to drop
        A = A[:, col_mask]  # Keep only columns where mask is True (CSR format preserved)

    # 2. Compute non-zero counts for each row after column filtering
    row_nnz = A.getnnz(axis=1)  # Efficiently count non-zeros per row (returns numpy array)

    # 3. Build mask for rows: birth time <= threshold and not an empty row
    row_mask = (birth_times_row <= threshold) & (row_nnz > 0)
    if not row_mask.all():  # Only slice if there are rows to drop
        A = A[row_mask, :]  # Keep only rows where mask is True (CSR format preserved)

    # Resulting A is still a CSR matrix with filtered rows/columns
    return A, row_mask


def threshold_y_optimized(y, birth_times_row, threshold, row_mask):
    """
    Filter the sparse vector y (n x 1) by:
      1. Zeroing out entries whose birth_times_row > threshold.
      2. Applying the row_mask (obtained from threshold_A) to select only the rows that remain.

    Parameters:
      y (csr_matrix): A sparse vector (n x 1) with few nonzero entries.
      birth_times_row (np.array): A 1D array of length n with birth times.
      threshold (float): The threshold value.
      row_mask (np.array): A boolean mask for rows (length n) to keep.

    Returns:
      csr_matrix: The filtered vector in CSR format.
    """
    # Create a boolean mask for entries that meet the threshold condition.
    valid_mask = (birth_times_row <= threshold)

    # Process sparse vector without densifying:
    # Convert y to COO format to access nonzero coordinates.
    y_coo = y.tocoo()
    # For each nonzero entry, check if its row's birth time is <= threshold.
    keep = valid_mask[y_coo.row]
    # Filter out the entries that don't pass.
    new_row = y_coo.row[keep]
    new_data = y_coo.data[keep]

    # Build a new sparse vector of shape (n,1) with only the valid nonzero entries.
    y_filtered = csr_matrix((new_data, (new_row, np.zeros_like(new_row))),
                            shape=(y.shape[0], 1))

    # Now apply the row_mask to y_filtered.
    y_final = y_filtered[row_mask, :]
    #are we basically applying the row_mask twice here?

    return y_final


def to_hashable(barcode):
    """
    Convert a barcode of the form (float, float, list of lists)
    into a fully hashable representation.
    """
    # Convert the list of lists into a tuple of tuples.
    hashable_list = tuple(tuple(row) for row in barcode[2])
    # Return the barcode as a tuple with all hashable elements.
    return (barcode[0], barcode[1], hashable_list)


# CUP LENGTH ALGORITHM COMPTABILE WITH RIPSER
def compute_persistent_cup_length(k: int, \
                                        ripser_result: Dict, \
                                        distance_matrix_full: np.array, \
                                        min_persistence: float, \
                                        thresh: float = math.inf) -> Tuple[np.array, List[int], List[int]]:

  """
    Compute persistent cup-length diagram for a given point cloud.

    This function implements a persistent cup-length algorithm using cohomology over Z mod 2
    and is designed to work with output from the Ripser persistent cohomology library. It identifies
    persistent higher-order cohomological features by computing cup products between persistent
    1-cocycles and checking their nontriviality via linear constraints on coboundary matrices.

    Args:
        k (int): Maximum cup length to compute (i.e., maximum power of the cup product to check; default is 2).
        ripser_result (Dict): Dictionary containing the output from Ripser (typically includes barcodes,
                              cocycles, and landmark subsampling indices).
        distance_matrix_full (np.array): Full pairwise distance matrix of the point cloud (before landmark subsampling).
        min_persistence (float): Minimum persistence required for input cocycles to be considered in cup products.
        thresh (float, optional): Filtration threshold (defaults to infinity). Used to restrict cocycles to features
                                  born before or at this threshold.

    Returns:
        Tuple[np.array, List[int], List[int], List[Tuple[float, float]]]:
            - A_l (np.array): Persistent cup length matrix representation at level l.
                              in A_2, entry (i, j) contains a non-zero k (either 1 or 2) if a cup length of length k persists from
                              birth time b_time[i] to death time d_time[j].
            - b_time (List[int]): Sorted list of all birth times of degree-2 cohomology classes.
            - d_time (List[int]): Sorted list of all death times of cohomology classes (with `inf` appended).
            - cup_length_2_intervals (List[Tuple[float, float]]): List of cup length 2 intervals.

    Notes:
        - The function uses a combinatorial number system to encode simplices and index into filtration birth times.
        - Only pairs of cocycles with persistence ≥ `min_persistence` are considered for cup product testing.
  """
  cup_length_2_intervals = []
  # use subsampled points in the cloud
  idx_land = ripser_result["idx_perm"]  # indices of points in original point cloud that are in the subsample
  distance_matrix = distance_matrix_full[np.ix_(idx_land, idx_land)]

  n_vertices = distance_matrix.shape[0]
  maxdim = 2

  lookup_table = combinatorial_number_system_table(n_vertices, maxdim)  # maxdim = maximum homology dimension
  coboundary_matrix, birth_times_row, birth_times_col = CohomologyUtils.make_delta1(dist_mat=distance_matrix,
                                                                                                 threshold=math.inf,
                                                                                                 lookup_table=lookup_table)

  original_to_subsampled = {orig: sub for sub, orig in enumerate(idx_land)}
  annotated_barcodes = compute_annotated_barcodes(ripser_result)
  B_1_k = convert_annotated_barcodes(annotated_barcodes, original_to_subsampled)

  valid = [item for item in B_1_k if len(item[2][0]) == 2]
  # Invalid: at least one sublist does not have length 2.
  invalid = [item for item in B_1_k if not len(item[2][0]) == 2]
  # Sort the valid tuples in descending order by the difference (second - first)
  valid_sorted = sorted(valid, key=lambda t: t[1] - t[0], reverse=True)

  # Combine the sorted valid tuples with the invalid ones appended at the end
  B_1_k_sorted = valid_sorted + invalid

  b_time = sorted([item[0] for item in B_1_k if len(item[2][0]) == 3])  #all birth times in degree 2
  d_time = sorted([elt[1] for elt in B_1_k]) #all death times
  d_time.append(math.inf)

  l, B_1 = 1, B_1_k_sorted
  A_0 = np.zeros((len(b_time), len(d_time)))
  A_1 = np.zeros((len(b_time), len(d_time)))
  A = [A_0, A_1]

  B = [[], B_1]

  b_d_times = [(elt[0], elt[1]) for elt in B_1_k]

  for i in range(len(b_time)):
    for j in range(len(d_time)):
      if (b_time[i], d_time[j]) in b_d_times:
        A_1[i,j] = 1

  A[1] = A_1
  while ((not np.array_equal(A[l-1], A[l])) and (l <= k-1)):
    A.append(A[l].copy())
    B.append([]) # this is initialization for B_l_plus_1

    seen_pairs = set()

    for barcode_1 in B[1]:
      for barcode_2 in B[l]:
        hashable1 = to_hashable(barcode_1)
        hashable2 = to_hashable(barcode_2)

        # Create a canonical representation for the pair.
        # Sorting the two hashable barcodes ensures that the order does not matter.
        pair_key = tuple(sorted((hashable1, hashable2)))

        # If the pair was already processed, skip it.
        if pair_key in seen_pairs:
            continue

        # Mark this pair as seen.
        seen_pairs.add(pair_key)
        if ((barcode_1[1] - barcode_1[0]) < min_persistence) | ((barcode_2[1] - barcode_2[0]) < min_persistence):
          break #go to the next pair of barcodes whose persistence >= min_persistence
        else:

          sigma, sigma1, sigma2 = cup_product(barcode_1[2], barcode_2[2], birth_times_row, lookup_table, thresh = math.inf, max_dim_X = 2)
          y = vector_rep_of_cochain(sigma, lookup_table, birth_times_row, distance_matrix.shape[0], thresh, 'csr')
          if contains_nonzero(y.toarray()):
            d_min = min(barcode_1[1], barcode_2[1])

            i_primes_ = [i for i in range(len(b_time)) if b_time[i] <= d_min]
            if len(i_primes_) == 0: #unnecessary?
                break
                
            i_ = max(i_primes_)

            closest_birth_time =  b_time[i_]
            d_eval = (d_min + closest_birth_time) / 2

            thresholded_A, row_mask = threshold_A_optimized(coboundary_matrix, birth_times_row, birth_times_col, d_eval)
            thresholded_y = threshold_y_optimized(y, birth_times_row, d_eval, row_mask)

            t0 = time.time()
            if (has_binary_solution(thresholded_A.toarray(), thresholded_y.toarray().reshape(1,-1)[0])[0] == False):
              t1 = time.time()
              i_primes = [i for i in range(len(b_time)) if b_time[i] <= d_min]
              i = max(i_primes)

              b_time_eval = (b_time[i] + b_time[i - 1]) / 2
              b_time_i = b_time[i]

              thresholded_A, row_mask = threshold_A_optimized(coboundary_matrix, birth_times_row, birth_times_col, b_time_eval)
              thresholded_y = threshold_y_optimized(y, birth_times_row, b_time_eval, row_mask)

              while (has_binary_solution(thresholded_A.toarray(), thresholded_y.toarray().reshape(1,-1)[0])[0] == False):
                  if i == 0:
                    break

                  i = i-1
                  b_time_eval = (b_time[i] + b_time[i-1])/2
                  b_time_i = b_time[i]

                  thresholded_A, row_mask = threshold_A_optimized(coboundary_matrix, birth_times_row, birth_times_col, b_time_eval)
                  thresholded_y = threshold_y_optimized(y, birth_times_row, b_time_eval, row_mask)

                  if contains_nonzero(thresholded_y.toarray().reshape(1,-1)[0]) == False:
                    break

              if math.isinf(b_time_i) == False:  
                if (b_time_i < d_min):
                    print(f'cup length 2 interval: {b_time_i, d_min}')
                    cup_length_2_intervals.append((b_time_i, d_min))
                    B[l+1].append((b_time_i,  d_min, sigma))
                    d_min_index = d_time.index(d_min) 
                    A[l+1][i, d_min_index] = l + 1

      if ((barcode_1[1] - barcode_1[0]) < min_persistence):
          break
    l = l + 1
  return (A[l], b_time, d_time, cup_length_2_intervals)



 

def plot_persistent_homology_barcodes(ph_data, cup_interval=None, y_offset=2, bar_spacing=0.3):
    color_cycle = ['#FF8C00', 'green', 'black']

    plt.figure(figsize=(8, 4))
    max_x = 0

    # Dictionary to store endpoints of bars for vertical lines
    bar_ends = {
        'orange': [],  # store all H0 bars
        'green': []    # store all H1 bars
    }

    # Plot each homology dimension
    for dim, arr in enumerate(ph_data):
        # Sort by persistence
        if arr.size > 0:
            persistence = arr[:, 1] - arr[:, 0]
            idx_sorted = np.argsort(persistence)[::-1]
            sorted_arr = arr[idx_sorted]
        else:
            sorted_arr = arr

        n = len(sorted_arr)
        y_base = -dim * y_offset

        if n > 0:
            offsets = np.linspace(-bar_spacing * (n - 1) / 2,
                                  bar_spacing * (n - 1) / 2,
                                  n)
            y_positions = y_base + offsets

            color = color_cycle[dim % len(color_cycle)]
            for i, ((b, d), y) in enumerate(zip(sorted_arr, y_positions)):
                plt.hlines(y, b, d, color=color, linewidth=2)
                max_x = max(max_x, d)

                if dim == 0:
                    if i < 2:
                        label = r"$\alpha$" if i == 0 else r"$\beta$"
                        plt.text(d + 0.1, y, label, va='center', fontsize=16)
                    bar_ends['orange'].append((b, d, y))

                if dim == 1:
                    bar_ends['green'].append((b, d, y))

        plt.text(-1, y_base, f"$H_{{{dim + 1}}}$", va='center', fontsize=16, fontweight='bold')

    # Always reserve label position for Cup length 2
    y_cup = -len(ph_data) * y_offset
    plt.text(-2, y_cup, "Cup length 2", va='center', fontsize=16)

    if cup_interval is not None:
        cup_left, cup_right = cup_interval
        plt.hlines(y_cup, cup_left, cup_right, color='black', linewidth=2)
        max_x = max(max_x, cup_right)

        # Find matching orange bar ending at cup_right
        for (b, d, y) in bar_ends['orange']:
            if np.isclose(d, cup_right):
                plt.vlines(cup_right, y_cup, y, linestyle='dotted', color='black', linewidth=1.5)
                break

        # Find matching green bar starting at cup_left
        for (b, d, y) in bar_ends['green']:
            if np.isclose(b, cup_left):
                plt.vlines(cup_left, y_cup, y, linestyle='dotted', color='black', linewidth=1.5)
                break

    # Axes and limits

    top_offset = 0
    if ph_data and len(ph_data[0]) > 1:
        top_offset = bar_spacing * (len(ph_data[0]) - 1) / 2
    y_top = 0 + top_offset
    plt.ylim(y_cup - 0.5, y_top + 0.5)
    plt.xlabel("Lifespan", fontsize=16)  # Increase font size of x-axis label
    plt.xticks(fontsize=12)              # Increase font size of x-axis tick labels
    plt.yticks([], fontsize=12)
    plt.xlim(0, max_x + 1)
    plt.yticks([])

    plt.tight_layout()
    plt.show()


 

def plot_and_extract_staircase_polygons(matrix, h1_dgms, births, deaths,
                                        diagonal_range=None,
                                        death_max=None,
                                        facecolors=None,
                                        pointcolors=None,
                                        alpha=0.3,
                                        show_polygon_vertices=False,
                                        show_cup_length_diagram=True,
                                        title='Cup-Length Diagram and Cup-Length Function'):
    """
    Compute and plot staircase polygons grouped by cup-length values from a matrix and birth/death data.

    Returns:
        {
            cup_length_val: {
                'diagram_points': [...],
                'critical_points': [...],
                'polygon': [...]
            },
            ...
        }
    """
    output = {}
    rows, cols = matrix.shape

    # Step 1: Group by cup-length value
    for i in range(rows):
        for j in range(cols):
            val = matrix[i, j]
            if val == 0:
                continue
            b, d = births[i], deaths[j]
            if d <= b:
                continue
            output.setdefault(val, {'diagram_points': []})
            output[val]['diagram_points'].append((b, d))


    for bar in h1_dgms:
        output.setdefault(1, {'diagram_points': []})
        output[1]['diagram_points'].append(tuple(bar))

    # Step 2: Compute staircase critical points and polygons
    for val, data in output.items():
        points = sorted(data['diagram_points'], key=lambda x: x[0])
        crit = []
        idx = 0
        while idx < len(points):
            b, d = points[idx]
            crit.append((b, d))
            idx += 1
            while idx < len(points) and points[idx][1] <= d:
                idx += 1
        data['critical_points'] = crit

        # Build staircase polygons
        all_polygons = []
        i = 0
        while i < len(crit):
            polygon = []
            b, d = crit[i]
            polygon.append((b, b))
            polygon.append((b, d))
            while i + 1 < len(crit) and crit[i + 1][0] <= d:
                b_next, d_next = crit[i + 1]
                polygon.append((b_next, d))
                polygon.append((b_next, d_next))
                d = d_next
                i += 1
            polygon.append((d, d))
            all_polygons.append(polygon)
            i += 1
        data['polygon'] = all_polygons

    # Step 3: Determine plot range from data if not provided
    all_points = [pt for group in output.values() for pt in group['diagram_points']]
    if not all_points:
        print("No diagram points to plot.")
        return output

    max_val = max(max(b, d) for b, d in all_points)
    if diagonal_range is None:
        diagonal_range = (0, max_val + 0.2)
    if death_max is None:
        death_max = max_val + 0.2

    # Step 4: Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([*diagonal_range], [*diagonal_range], 'k--')

    label_added = set()
    for val, data in sorted(output.items()):
        if not data['diagram_points']:
            continue
        births_, deaths_ = zip(*data['diagram_points'])
        print('births_', births_)

        print('deaths_', deaths_)

        if show_cup_length_diagram:
            point_color = pointcolors.get(val, 'lightpink') if pointcolors else 'lightpink'
            label_pts = f'cup-length {int(val)} pts' if val not in label_added else None
            ax.scatter(births_, deaths_, s=15, color=point_color, label=label_pts)
        else:
            title = 'Cup-Length Function'

        for poly in data['polygon']:
            face_color = facecolors.get(val, 'pink') if facecolors else 'pink'
            label_poly = f'cup-length {int(val)}' if val not in label_added else None
            patch = patches.Polygon(poly, closed=True, facecolor=face_color,
                                    edgecolor='none', alpha=alpha, label=label_poly)
            ax.add_patch(patch)
            label_added.add(val)

        if show_polygon_vertices and data['critical_points']:
            crit_births, crit_deaths = zip(*data['critical_points'])
            ax.scatter(crit_births, crit_deaths, color='pink', s=10)

    ax.set_xlim(*diagonal_range)
    ax.set_ylim(0, death_max)
    ax.set_aspect('equal')
    # Set font sizes for labels and title
    ax.set_xlabel('Birth', fontsize=14)
    ax.set_ylabel('Death', fontsize=14)
    ax.set_title(title, fontsize=14)

    # Adjust tick label fonts
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Adjust legend font size
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=12)


    return output