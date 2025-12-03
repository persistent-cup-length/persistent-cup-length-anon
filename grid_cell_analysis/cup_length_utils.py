"""
Programmer: Katya Ivshina (ekaterina.s.ivshina@gmail.com) and Ling Zhou
Purpose: To provide a number of utility functions for cup length computation
"""
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

from ripser import ripser
import warnings
from dreimac_utils import *
from dreimac_combinatorial import *

import time
import numpy as np
import matplotlib.pyplot as plt
import math
from persim import plot_diagrams
from typing import List, Tuple, Dict, Optional
from scipy.special import comb
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances


def sample_torus(n_points, R, r):
    theta = 2 * np.pi * np.random.random(n_points)
    phi = 2 * np.pi * np.random.random(n_points)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.column_stack((x, y, z))


def generate_bunny_points(n_points=1000, sphere_ratio=0.6, circle_radius=0.3, seed=None):
    """
    Generate points for S¹ V S² V S¹ (sphere with two circles attached).

    Components:
    1. Sphere (S²): Main body
    2. Circle 1 (S¹): Attached at (1,0,0) in xy-plane
    3. Circle 2 (S¹): Attached at (0,0,1) in yz-plane

    The shape resembles a bunny with two "ears" (the circles).

    Args:
        n_points (int): Total number of points to generate
        sphere_ratio (float): Proportion of points to allocate to sphere (0-1)
        circle_radius (float): Radius of the attached circles
        seed (int, optional): Random seed for reproducibility

    Returns:
        np.ndarray: Array of shape (n_points, 3) containing point coordinates

    Raises:
        ValueError: If parameters are invalid
        MemoryError: If not enough memory to generate points
    """
    # Validate inputs
    if n_points < 1:
        raise ValueError("n_points must be positive")
    if not 0 < sphere_ratio < 1:
        raise ValueError("sphere_ratio must be between 0 and 1")
    if circle_radius <= 0:
        raise ValueError("circle_radius must be positive")

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    try:
        # Distribute points among components
        sphere_points = int(n_points * sphere_ratio)
        circle1_points = int(n_points * (1 - sphere_ratio) / 2)
        circle2_points = n_points - sphere_points - circle1_points

        # Generate sphere points with uniform distribution
        phi = np.arccos(2 * np.random.random(sphere_points) - 1)
        theta = np.random.uniform(0, 2 * np.pi, sphere_points)

        x_sphere = np.sin(phi) * np.cos(theta)
        y_sphere = np.sin(phi) * np.sin(theta)
        z_sphere = np.cos(phi)
        sphere = np.column_stack((x_sphere, y_sphere, z_sphere))

        # Generate first circle (xy-plane, attached at (1,0,0))
        theta1 = np.random.uniform(0, 2 * np.pi, circle1_points)
        x_circle1 = 1 + circle_radius * np.cos(theta1)
        y_circle1 = circle_radius * np.sin(theta1)
        z_circle1 = np.zeros_like(theta1)
        circle1 = np.column_stack((x_circle1, y_circle1, z_circle1))

        # Generate second circle (yz-plane, attached at (0,0,1))
        theta2 = np.random.uniform(0, 2 * np.pi, circle2_points)
        x_circle2 = np.zeros_like(theta2)
        y_circle2 = circle_radius * np.cos(theta2)
        z_circle2 = 1 + circle_radius * np.sin(theta2)
        circle2 = np.column_stack((x_circle2, y_circle2, z_circle2))

        # Combine all points
        points = np.vstack((sphere, circle1, circle2))

        # Add small random noise (commented out)
        # noise = np.random.normal(0, 0.01, points.shape)
        # points += noise

        return points

    except MemoryError:
        raise MemoryError("Not enough memory to generate points. Try reducing n_points.")

def makeSparseDM(X: np.array, thresh: float) -> sparse.coo_matrix:
    """
    create a sparse distance matrix from point cloud X. Only keep distances <= threshold
    """
    N = X.shape[0]
    D = pairwise_distances(X, metric='euclidean')
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    I = I[D <= thresh]
    J = J[D <= thresh]
    V = D[D <= thresh]
    return sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()


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
      n_land = optional number of landmarks
      thresh = optional filtration threshold; either should be used here or in makeSparseDM
    Returns:
      ripser_result = dictionary containing persistence diagrams and cocycles. Here is info
      for cocycles: For each dimension less than maxdim a list of representative cocycles.
      Each representative cocycle in dimension d is represented as a ndarray of (k,d+1) elements.
      Each non zero value of the cocycle is laid out in a row, first the d indices of the simplex
      and then the value of the cocycle on the simplex. The indices of the simplex reference the
      original point cloud, even if a greedy permutation was used.
      c.f. https://ripser.scikit-tda.org/en/latest/reference/stubs/ripser.ripser.html#ripser.ripser
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
      ripser_res = resulting dictionary returned by ripser(), which contains diagrams and cocycles
      k = cohomology dimension bound
    Returns:
      B_1_k (barcodes annotated by representative cocycles)
    """
    # iterate over dims from 1 to k
    annotated_barcodes = []
    dgms = ripser_res['dgms']
    cocycles = ripser_res['cocycles']

    for dim in range(1, k + 1):
        for i in range(dgms[dim].shape[0]):
            cocycle = cocycles[dim][i][:, :-1].tolist()  # we drop the last column which gives the
            # value of the cocycle on the simplex

            annotated_barcodes.append((dgms[dim][i, 0], dgms[dim][i, 1], cocycle))

    # sort annotated barcodes first in the increasing order of the death time
    # and then in the increasing order of the birth time
    #annotated_barcodes.sort(key=lambda x: x[0])
    #annotated_barcodes.sort(key=lambda x: x[1])
    annotated_barcodes = sorted(annotated_barcodes, key=lambda annotated_barcodes: (annotated_barcodes[1], annotated_barcodes[0]))

    return annotated_barcodes




def simplex_exists(new_a: List, distance_matrix: np.array, thresh: float) -> bool:
    """
    checks if simplex new_a exists at filtration time = thresh
    """
    edges = create_pairs_(new_a)

    edge_distances = [distance_matrix[edge[0], edge[1]] for edge in edges]  # !

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
    counts the number of simplices alive at time = thresh
    """
    # get total number of vertices
    num_of_vertices = distance_matrix.shape[0]
    # compute all possible pairs (edges) and triples (triangles) in the filtration
    edges = create_pairs_(range(num_of_vertices))
    triangles = create_triples_(range(num_of_vertices))
    # check if simplex has birth time <= thresh, i.e. its diameter is <= b_i
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
    Compute the cup product of two cochains over Z_2
    Args:
      cocycle1: first cocycle
      cocycle2: second cocycle
      X: simplicial complex
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


# def cup_product(cocycle1_rep: List, cocycle2_rep: List, distance_matrix: np.array, thresh: float,
#                 max_dim_X: int = 2) -> List:
#     """
#     Compute the cup product of two cochains over Z_2
#     Args:
#       cocycle1: first cocycle
#       cocycle2: second cocycle
#       X: simplicial complex
#     """
#     size_of_cocycle1 = len(cocycle1_rep)
#     size_of_cocycle2 = len(cocycle2_rep)
#     p = size_of_cocycle1 + size_of_cocycle2
#
#     # compute the dimension of the cohomology group the cocycle belongs to
#     dim_of_cocycle1 = len(cocycle1_rep[0]) - 1
#     dim_of_cocycle2 = len(cocycle2_rep[0]) - 1
#
#     sigma = []
#     if dim_of_cocycle1 + dim_of_cocycle2 <= max_dim_X:
#         for i in range(size_of_cocycle1):
#             for j in range(size_of_cocycle2):
#
#                 a, b = cocycle1_rep[i], cocycle2_rep[j]
#                 new_a = a.copy()
#                 if a[-1] == b[0]:
#                     if len(b[1:]) > 0:
#                         new_a = new_a + b[1:]
#                         if simplex_exists(new_a, distance_matrix, thresh):
#                             sigma.append(new_a)
#     return sigma, cocycle1_rep, cocycle2_rep


# Assuming you have the following:
# annotated_barcodes: list of tuples, where the third entry is a list of lists containing indices
# original_to_subsampled: dictionary mapping original indices to subsampled indices

def convert_indices(indices, mapping):
    # mapping original indices to subsampled indices
    # mapping = {orig: sub for sub, orig in enumerate(idx_land)}
    return [mapping.get(idx, None) for idx in indices if mapping.get(idx, None) is not None]


def convert_annotated_barcodes(annotated_barcodes, original_to_subsampled):
    # Convert indices in annotated_barcodes
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

# test
# annotated_barcodes = [(1.0,2.0, [[1,2,5]])]
# original_to_subsampled = {0:0, 1:2, 2:3, 5:1}
# converted_barcodes = convert_annotated_barcodes(annotated_barcodes, original_to_subsampled)
# # Print example of conversion (optional)
# print("Original annotated_barcodes (first item):", annotated_barcodes[0])
# print("Converted annotated_barcodes (first item):", converted_barcodes[0])

# # Now converted_barcodes contains the updated indices


def gf2_gaussian_elimination(A, b):
    A = np.array(A, dtype=int) % 2
    b = np.array(b, dtype=int) % 2
    M = np.column_stack((A, b))
    m, n = A.shape  # Number of rows and columns

    row = 0  # Tracks the current row
    for col in range(n):  # Iterate over columns
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
                M[r] ^= M[row]  # XOR operation for elimination

        row += 1  # Move to next row

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
# this function below assumes number of columns is >= number of rows
# def gf2_gaussian_elimination(A, b):
#     # Combine A and b into an augmented matrix
#     M = np.column_stack((A, b))
#     m, n = M.shape
#     print('m, n : ', m, n)
#
#     # Forward elimination
#     for i in range(min(m, n - 1)):
#         # Find pivot
#         pivot_row = i
#         for j in range(i + 1, m):
#             if M[j, i] == 1:
#                 pivot_row = j
#                 break
#
#         # Swap rows if necessary
#         if pivot_row != i:
#             M[i], M[pivot_row] = M[pivot_row].copy(), M[i].copy()
#
#         # Eliminate below
#         for j in range(i + 1, m):
#             if M[j, i] == 1:
#                 M[j] = np.logical_xor(M[j], M[i]).astype(int)
#
#     # Back substitution
#     x = np.zeros(n - 1, dtype=int)
#     for i in range(m - 1, -1, -1):
#         if M[i, i] == 1:
#             x[i] = M[i, -1]
#             for j in range(i + 1, n - 1):
#                 x[i] ^= (M[i, j] & x[j])
#         elif np.any(M[i, i+1:-1]) or M[i, -1] == 1:
#             return False, None  # No solution
#
#     # Check if solution is valid
#     if np.array_equal(np.dot(A, x) % 2, b):
#         return True, x
#     else:
#         return False, None

def has_binary_solution(A, b):
    # Check if A and b contain only 0 and 1
    if not np.all(np.logical_or(A == 0, A == 1)) or not np.all(np.logical_or(b == 0, b == 1)):
        raise ValueError("A and b must contain only 0 and 1")

    return gf2_gaussian_elimination(A, b)

def contains_nonzero(arr):
    return np.any(arr != 0)


def convert_sparse_cocycle_to_vector_csr(sparse_cocycle, lookup_table, n_vertices, dtype):
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
    # max_dimension = sparse_cocycle.shape[1] - 1
    # dimensions = list(range(1, max_dimension + 1))
    #
    # num_of_simplices_greater_than_deg_0 = sum(
    #     number_of_simplices_of_dimension(d, n_vertices, lookup_table)
    #     for d in dimensions)
    #
    # total_number_of_simplices  = n_vertices + num_of_simplices_greater_than_deg_0 #vertices, edges, triangles
    # cocycle_as_vector = np.zeros((total_number_of_simplices,), dtype=dtype) # is the size of the cocycle correct?

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
    Compute vector representation of thresholded sparse_cocycle, i.e find
    y such that the thresholded sparse_cocycle = S_star dot y where y has 0 or 1 entries.
    We don't have access to S_star filtration directly and rely on the lookup_table instead.
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
      k (int): The number of elements in the combination. Note that this is simplex dimension + 1
      
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


def threshold_y(y: np.array, birth_times: Dict, t0: float):
    """
    y: vector representation of cocycle (in our case consists of triangles)
    birht_times: dictionary of birth times for all simplices in the Rips filtration
    t0: filtration time
    Returns indices of all simplices in y born before or at t0
    """
    non_zero_indices = np.nonzero(y)
    indices_of_simplices_alive_before_t0 = []
    for i in non_zero_indices[0]:

        simplex = decode_combinatorial_number(i, 3)

        simplex_birth_time = birth_times.get(simplex, float('inf'))  # Default to inf if not found
        if simplex_birth_time <= t0:
            indices_of_simplices_alive_before_t0.append(i)

    # Create a mask and return the subset of y
    y_masked = np.zeros_like(y)
    y_masked[indices_of_simplices_alive_before_t0] = y[indices_of_simplices_alive_before_t0]

    return y_masked


def threshold_A(A: np.array, birth_times: Dict, t0: float):
    """
    A: coboundary matrix (rows are triangles and columns are edges ([i,j] index is 1 when triangle i has edge j))
    birht_times: dictionary of birth times for all simplices in the Rips filtration
    t0: filtration time
    Returns indices of all rows/columns in A born before or at t0

    When solving Ax=y, x consists of all edges bounding triangles in y????
    """
    indices_of_edges_alive_before_t0 = []
    for i in range(A.shape[1]):
        simplex = decode_combinatorial_number(i, 2)
        simplex_birth_time = birth_times.get(simplex, float('inf'))  # Default to inf if not found
        #print('i: ', i, 'simplex: ', simplex, 'birth time: ', simplex_birth_time)
        if simplex_birth_time <= t0:
            indices_of_edges_alive_before_t0.append(i)

    # Create column mask
    col_mask = np.zeros(A.shape[1], dtype=bool)
    col_mask[indices_of_edges_alive_before_t0] = True

    # Step 2: Find indices of rows that are not all zero in the alive columns
    A_alive_cols = A[:, col_mask]  # Restrict to alive columns
    row_mask1 = np.any(A_alive_cols != 0, axis=1)  # Rows with at least one nonzero entry
    # we also want to remove rows (triangles) corresponding to triangles with birth times > t0
    # note that if we had a triangle [a b c] and we removed its edges, then the triangle is not present automatically
    # that is what removing zero rows is for
    # but we might have a triangle [a b c] whose edges get born before t0 but the triangle itself is born at time > t0

    A_col_mask = A[:, col_mask]
    indices_of_triangles_alive_before_t0 = []
    for i in range(A.shape[0]):
        simplex = decode_combinatorial_number(i, 3)
        simplex_birth_time = birth_times.get(simplex, float('inf'))  # Default to inf if not found
        if simplex_birth_time <= t0:
            indices_of_triangles_alive_before_t0.append(i)

    row_mask2 = np.zeros(A.shape[0], dtype=bool)
    row_mask2[indices_of_triangles_alive_before_t0] = True
    row_mask = row_mask1 & row_mask2

    # print('row_mask1 ', row_mask1)
    # print('row_mask2 ', row_mask2)
    # print('row_mask ', row_mask)

    A_sub = A[row_mask, :][:, col_mask]
    return A_sub, row_mask


def compute_threshold_persistence(arr: np.array):
    """
    given input array of birth-death times in H^1,
    compute persistence of each point and return
    the mean between second and third largest persistences
    (so that when we use this threshold persistence,
    we choose top 2 largest 1-cycles)
    """
    # Compute differences for each row: second element minus first element
    differences = arr[:, 1] - arr[:, 0]

    # Use np.partition to get the three largest differences
    # We partition so that the three largest values are in the last three positions (in any order)
    top3 = np.partition(differences, -3)[-3:]

    # Sort these three values in descending order so that:
    # top3_sorted[0] is the largest, [1] is the second largest, and [2] is the third largest.
    top3_sorted = np.sort(top3)[::-1]

    # Compute the mean of the second and third largest differences
    return (top3_sorted[1] + top3_sorted[2]) / 2


def threshold_A_optimized(A: csr_matrix, birth_times_row: np.array, birth_times_col: np.array, threshold: float):
    # Ensure A is in CSR format (if not already)

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


# CUP LENGTH ALGORITHM WITH RIPSER
def compute_persistent_cup_length_ripser(k: int, \
                                        ripser_result: Dict, \
                                        distance_matrix_full: np.array, \
                                        min_persistence: float, \
                                        thresh: float = math.inf) -> Tuple[np.array, List[int], List[int]]:

  """
  Args:
    k: dimension bound
    S_star: the ordered list of cosimplices from dimension 1 to k+1
    coboundary_matrix: coboundary matrix (rows are triangles, columns are edges
    B_1_k: barcodes (b_sigma, d_sigma, sigma) annotated by representative
    cocycles, from dimension 1 to k.  (sigma_1,...,sigma_q1) is ordered first
    in the increasing order of the death time and then in the increasing order
    of the birth time.
    lookup_table: lookup table for combinatorial number system
    birth_times_row: birth times for each triangle (ordered by combinatorial number system)
    birth_times_col: birth times for each edge (ordered by combinatorial number system)
    min_persistence: minimum persistence to consider when computing cup product
    thresh: filtration threshold
  Return: a matrix representation of the persistent cup-length-diagram, and
  the lists of distinct birth times b_time and death times d_time
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
  # subsampled_to_original = {sub: orig for sub, orig in enumerate(idx_land)}
  annotated_barcodes = compute_annotated_barcodes(ripser_result)
  B_1_k = convert_annotated_barcodes(annotated_barcodes, original_to_subsampled)

  #B_1_k = sorted(B_1_k, key=lambda barcode: barcode[1] - barcode[0], reverse=True) #ok to sort like
  #this in order from largest to smallest persistence?

  valid = [item for item in B_1_k if len(item[2][0]) == 2]
  # Invalid: at least one sublist does not have length 2.
  invalid = [item for item in B_1_k if not len(item[2][0]) == 2]

  # Sort the valid tuples in descending order by the difference (second - first)
  valid_sorted = sorted(valid, key=lambda t: t[1] - t[0], reverse=True)

  # Combine the sorted valid tuples with the invalid ones appended at the end
  B_1_k_sorted = valid_sorted + invalid


  #B_1_k looks like [(pt1, cocycle1), (pt2, cocycle2)]
  b_time = sorted([elt[0] for elt in B_1_k]) #all birth times
  d_time = sorted([elt[1] for elt in B_1_k]) #all death times
  d_time.append(math.inf) # do we need this line?

  # m_k is the number of simplices with positive dimension in the (k + 1)-skeleton X_{k+1} of X
  # is m_k the number of triangles?
  # m_k = coboundary_matrix.shape[0]
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
  #print('min_persistence ', min_persistence)
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
        #print('another pair =====', barcode_1[1] - barcode_1[0], barcode_2[1] - barcode_2[0])
        #print('barcode_1', barcode_1)
        #print('barcode_2', barcode_2)
        #print('condition ', ((barcode_1[1] - barcode_1[0]) < min_persistence) | ((barcode_2[1] - barcode_2[0]) < min_persistence))
        if ((barcode_1[1] - barcode_1[0]) < min_persistence) | ((barcode_2[1] - barcode_2[0]) < min_persistence):
          break #go to the next pair of barcodes whose persistence >= min_persistence
        else:

          sigma, sigma1, sigma2 = cup_product(barcode_1[2], barcode_2[2], birth_times_row, lookup_table, thresh = math.inf, max_dim_X = 2)
          #print('barcode_1: ', barcode_1)
          #print('barcode_2: ', barcode_2)
          #print('sigma: ', sigma)
          # careful! do we need to reverse order in y? that's what [::-1] is doing...
          y = vector_rep_of_cochain(sigma, lookup_table, birth_times_row, distance_matrix.shape[0], thresh, 'csr') #.astype(int) #[::-1]

          #nonzero_count = y.nnz

          #print("Number of nonzero elements:", nonzero_count)

          if contains_nonzero(y.toarray()):
            #print('entered if contains_nonzero')
            d_min = min(barcode_1[1], barcode_2[1])

            thresholded_A, row_mask = threshold_A_optimized(coboundary_matrix, birth_times_row, birth_times_col, d_min)
            thresholded_y = threshold_y_optimized(y, birth_times_row, d_min, row_mask)

            t0 = time.time()
            if (has_binary_solution(thresholded_A.toarray(), thresholded_y.toarray().reshape(1,-1)[0])[0] == False):
              t1 = time.time()
              #print('Time took: ', (t1-t0)/60)
              # get max birth time <= d_min and count number of simplices
              # alive at that time
              i_primes = [i for i in range(len(b_time)) if b_time[i] <= d_min]
              i = max(i_primes)
              b_time_i = b_time[i] - 0.0000001
              #print('entered if has binary sol, b_time[i]:', b_time_i)

              thresholded_A, row_mask = threshold_A_optimized(coboundary_matrix, birth_times_row, birth_times_col, b_time_i)
              thresholded_y = threshold_y_optimized(y, birth_times_row, b_time_i, row_mask)

              while (has_binary_solution(thresholded_A.toarray(), thresholded_y.toarray().reshape(1,-1)[0])[0] == False):
                  if i == 0:
                    break

                  i = i-1
                  b_time_i = b_time[i] - 0.0000001
                  #print('in while loop, b_time[i]', b_time_i)

                  thresholded_A, row_mask = threshold_A_optimized(coboundary_matrix, birth_times_row, birth_times_col, b_time_i)
                  thresholded_y = threshold_y_optimized(y, birth_times_row, b_time_i, row_mask)

                  if contains_nonzero(thresholded_y.toarray().reshape(1,-1)[0]) == False:
                    #print(f'exited while loop because thresholded_y == 0, b_time_i = {b_time_i}')
                    break

              if math.isinf(b_time_i) == False: # do we need this line? birth time can not be inf, right?
                if (b_time_i < d_min):
                    print(f'cup length 2 interval: {b_time_i, d_min}')
                    cup_length_2_intervals.append((b_time_i, d_min))
                    B[l+1].append((b_time_i,  d_min, sigma))
                    d_min_index = d_time.index(d_min) #check!
                    A[l+1][i, d_min_index] = l + 1

                    # np.savetxt(f"results/sigma_{b_time_i}_{d_min}.csv", sigma, delimiter=",")  # Adjust precision as needed
                    # np.savetxt(f"results/barcode_1_{b_time_i}_{d_min}.csv", barcode_1[2], delimiter=",",fmt="%d")
                    # np.savetxt(f"results/barcode_2_{b_time_i}_{d_min}.csv", barcode_2[2], delimiter=",",fmt="%d")

      if ((barcode_1[1] - barcode_1[0]) < min_persistence):
          break
    l = l + 1
  return (A[l], b_time, d_time, cup_length_2_intervals)

 

def plot_ph_subplots_continuous_lines(ph_data, cup_interval=None):
    """
    Create 4 subplots (H0, H1, H2, Cup length 2) and draw vertical dotted lines
    at key x-positions continuously across the H1, H2, and Cup subplots.

    Parameters
    ----------
    ph_data : list of np.ndarray
        A list where element 0 is H0, element 1 is H1, element 2 is H2.
        Each is an array of shape (n,2) with [birth, death] pairs.
    cup_interval : tuple of two floats, optional
        If provided, a tuple (cup_left, cup_right) for the Cup length 2 interval.
    """
    # --- Sort each dimension's intervals by persistence (death - birth) descending ---
    for i in range(len(ph_data)):
        arr = ph_data[i]
        if arr.shape[0] > 0:
            persistence = arr[:, 1] - arr[:, 0]
            sorted_indices = np.argsort(persistence)[::-1]
            ph_data[i] = arr[sorted_indices]

    # --- Determine key x positions for the vertical lines ---
    best_H1_x = None
    if ph_data[1].shape[0] >= 2:
        arr1 = ph_data[1]
        # Choose the one (of the top two) with the earlier death time.
        if arr1[0, 1] < arr1[1, 1]:
            best_H1_x = arr1[0, 1]
        else:
            best_H1_x = arr1[1, 1]
    elif ph_data[1].shape[0] == 1:
        best_H1_x = ph_data[1][0, 1]

    best_H2_x = None
    if ph_data[2].shape[0] >= 1:
        best_H2_x = ph_data[2][0, 0]

    # --- Create 4 subplots (vertically stacked) with a shared x-axis ---
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 12))

    # A helper function to plot a barcode for one dimension.
    def plot_barcode(ax, arr, dim_label, add_alpha_beta=False):
        """
        Plot barcode intervals as horizontal lines in a given axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on.
        arr : np.ndarray
            Array of intervals (n x 2).
        dim_label : str
            Dimension label (e.g., "H0").
        add_alpha_beta : bool
            If True and if at least two intervals are present, label the first two as α and β.
        """
        n = arr.shape[0]
        if n == 0:
            ax.text(0.05, 0.5, r"${dim_label}$: No intervals", transform=ax.transAxes,
                    verticalalignment='center', fontsize=12)
            ax.set_ylim(0, 1)
            return

        # For H1, use increased spacing between bars and add extra top margin.
        if dim_label == "H_1":
            spacing_factor = 20.0  # Increase vertical spacing between bars
            top_margin = 10  # Extra space above the top bar
            y_positions = np.linspace(n * spacing_factor + top_margin, top_margin, n)
            ax.set_ylim(0, n * spacing_factor + 200)  # + 0.5)
        else:
            spacing_factor = 1.0
            y_positions = np.linspace(n * spacing_factor, spacing_factor, n)
            ax.set_ylim(0, n * spacing_factor + spacing_factor)

        for i, (birth, death) in enumerate(arr):
            ax.hlines(y=y_positions[i], xmin=birth, xmax=death, color='C0', linewidth=2)
            if add_alpha_beta and i < 2:
                label = r"$\alpha$" if i == 0 else r"$\beta$"
                # Place label a bit to the right of the death.
                ax.text(death + 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]), y_positions[i],
                        label, verticalalignment='center', fontsize=12)
        # ax.set_ylabel("$" + dim_label + "$", fontsize=12, fontweight='bold')
        ax.set_ylabel(fr"${dim_label}$", fontsize=12, fontweight='bold')

        ax.set_yticks([])

    # --- Plot H0 (top subplot) ---
    plot_barcode(axs[0], ph_data[0], "H_0", add_alpha_beta=False)
    axs[0].set_title("Barcode Plots", fontsize=14)

    # --- Plot H1 (second subplot) ---
    plot_barcode(axs[1], ph_data[1], "H_1", add_alpha_beta=True)

    # --- Plot H2 (third subplot) ---
    plot_barcode(axs[2], ph_data[2], "H_2", add_alpha_beta=False)

    # --- Plot Cup length 2 interval (bottom subplot) ---
    ax_cup = axs[3]
    if cup_interval is not None:
        cup_left, cup_right = cup_interval
        ax_cup.hlines(y=1, xmin=cup_left, xmax=cup_right, color='black', linewidth=2)
        ax_cup.set_ylabel("Cup length 2", fontsize=12, fontweight='bold')
        ax_cup.set_yticks([])
    else:
        ax_cup.set_ylabel("Cup length 2", fontsize=12, fontweight='bold')
        ax_cup.set_yticks([])
    ax_cup.set_ylim(0, 2)

    # --- Draw vertical dotted lines in H1, H2, and Cup subplots ---
    # These calls will draw the dotted lines at the same x coordinates in each subplot.
    for ax in [axs[1], axs[2], axs[3]]:
        if best_H1_x is not None:
            ax.axvline(x=best_H1_x, linestyle='dotted', color='black', linewidth=1.5)
        if best_H2_x is not None:
            ax.axvline(x=best_H2_x, linestyle='dotted', color='black', linewidth=1.5)

    # --- Place the label for α ⨟ β (alpha smilie beta) in the Cup subplot ---
    if best_H1_x is not None and best_H2_x is not None:
        mid_x = 0.5 * (best_H1_x + best_H2_x)
        ax_cup.text(mid_x, 1.7, r"$\alpha \smile \beta$", horizontalalignment='center', fontsize=12)

    axs[3].set_xlabel("Lifespan", fontsize=12)
    plt.xlim(left=0)
    plt.tight_layout()
    plt.show()




def plot_barcodes(ph_data, cup_interval=None, y_offset=2, bar_spacing=0.3):
    """
    Plot persistent homology barcodes for dimensions H1, H2, etc., skipping H0.
    (Other functionality remains the same.)
    """
    num_dims = len(ph_data)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Variables to store information for the dotted vertical lines.
    best_H1 = None  # For H1: (death_time, y_position)
    best_H2 = None  # For H2: (birth_time, y_position)

    plt.figure(figsize=(8, 5))
    max_x = 0  # To track maximum x for x-axis limit.

    # Loop over each homology dimension.
    # Skip H0 by checking the dimension index.
    for dim, arr in enumerate(ph_data):
        if dim == 0:
            continue  # Skip plotting for H0

        # Sort intervals by persistence (death - birth) in descending order.
        if arr.size > 0:
            persistence = arr[:, 1] - arr[:, 0]
            sort_indices = np.argsort(persistence)[::-1]
            sorted_arr = arr[sort_indices]
        else:
            sorted_arr = arr

        n_intervals = len(sorted_arr)
        # Define base y-level: H1 at y = (num_dims-1)*y_offset, H2 at (num_dims-2)*y_offset, etc.
        # (Since H0 is skipped, the labels will be for H1, H2, ...)
        y_base = (num_dims - dim) * y_offset

        # Compute y positions for the bars in this homology group.
        if n_intervals > 0:
            if n_intervals == 1:
                y_positions = [y_base]
            else:
                offsets = np.linspace(-bar_spacing * (n_intervals - 1) / 2,
                                        bar_spacing * (n_intervals - 1) / 2,
                                        n_intervals)
                y_positions = y_base + offsets

            color = color_cycle[dim % len(color_cycle)]

            # Draw each birth-death interval.
            for idx, ((birth, death), y) in enumerate(zip(sorted_arr, y_positions)):
                plt.hlines(y=y, xmin=birth, xmax=death, color=color, linewidth=2)
                max_x = max(max_x, death)

                # For H1 (dim == 1), label the two most persistent intervals.
                if dim == 1 and idx < 2:
                    label = r"$\alpha$" if idx == 0 else r"$\beta$"
                    plt.text(death + 0.1, y, label, verticalalignment='center', fontsize=12)

            # For H1: among the top two intervals, select the one with the earlier death time.
            if dim == 1 and n_intervals >= 2:
                death1 = sorted_arr[0, 1]
                death2 = sorted_arr[1, 1]
                if death1 < death2:
                    best_H1 = (death1, y_positions[0])
                else:
                    best_H1 = (death2, y_positions[1])

            # For H2: select the most persistent interval.
            if dim == 2 and n_intervals >= 1:
                best_H2 = (sorted_arr[0, 0], y_positions[0])

        # Label the homology dimension on the left.
        plt.text(-0.5, y_base, f"H{dim}", verticalalignment='center',
                 fontsize=12, fontweight='bold')

    # Draw the Cup length 2 interval at y=0 if provided.
    if cup_interval is not None:
        y_cup = 0
        cup_left, cup_right = cup_interval
        plt.hlines(y=y_cup, xmin=cup_left, xmax=cup_right, color='black', linewidth=2)
        plt.text(-1, y_cup, "Cup length 2", verticalalignment='center', fontsize=12, fontweight='bold')
        max_x = max(max_x, cup_right)

    # Draw dotted vertical lines from the chosen barcode ends down to y=0.
    if best_H1 is not None:
        death_time, y_pos = best_H1
        plt.vlines(x=death_time, ymin=0, ymax=y_pos, colors='black', linestyles='dotted', linewidth=1.5)
    if best_H2 is not None:
        birth_time, y_pos = best_H2
        plt.vlines(x=birth_time, ymin=0, ymax=y_pos, colors='black', linestyles='dotted', linewidth=1.5)

    # Place the label \alpha \smile \beta above the Cup length 2 bar,
    # centered between the two dotted vertical lines (if both exist).
    if best_H1 is not None and best_H2 is not None:
        x1 = best_H1[0]
        x2 = best_H2[0]
        mid_x = (x1 + x2) / 2
        plt.text(mid_x, 0.2, r"$\alpha \smile \beta$", horizontalalignment='center',
                 verticalalignment='bottom', fontsize=12)

    plt.xlabel("Lifespan (birth to death)")
    plt.xlim(left=0, right=max_x + 1)
    plt.yticks([])  # Hide y ticks; custom labels are used.
    plt.title("Persistent Homology Barcodes")
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Example persistent homology data:
    # H0: three intervals; H1: two intervals; H2: one interval.
    ph_data_example = [
        np.array([[0.0, 2.0], [0.0, 3.0], [1.0, 2.5]]),  # H0 (will be skipped)
        np.array([[1.5, 2.8], [2.0, 3.5]]),              # H1
        np.array([[2.2, 4.0]])                           # H2
    ]

    # Example cup_interval.
    b_times = np.array([0.5, 1.0, 1.5])
    d_times = np.array([3.0, 3.2, 4.0])
    row_indices = [0]  # so the left end is b_times[0]
    col_indices = [0]  # so the right end is d_times[0]
    cup_interval = (b_times[row_indices[0]], d_times[col_indices[0]])

    plot_barcodes(ph_data_example, cup_interval=cup_interval)
