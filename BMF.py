'''

Based on k-Greedy algorithm from https://github.com/kovacsrekaagnes/rank_k_Binary_Matrix_Factorisation


'''

import numpy as np
import pandas as pd
from math import ceil
from sklearn.metrics import confusion_matrix

# float comparison
EPS = np.finfo(np.float32).eps
# optimality tolerances... in some cases 1.0e-6 resulted in numerical precision errors, and infinite iterations for CG
TOL = 1.0e-4

def boolean_matrix_product(A, B):
    """
    Compute the Boolean matrix product of A and B

    :param A:       n x k binary matrix, if A is a 1 dim array it is assumed to be a column vector
    :param B:       k x m binary matrix, if B is a 1 dim array it is assumed to be a row vector

    :return:        n x m binary matrix, the Boolean matrix product of A and B
    """

    if len(A.shape) == 1 and len(B.shape) == 1:
        X = np.outer(A, B)

    elif len(A.shape) == 1 and len(B.shape) == 2:
        X = np.dot(A.reshape(len(A), 1), B)

    elif len(A.shape) == 2 and len(B.shape) == 1:
        X = np.dot(A, B.reshape(1, len(B)))

    else:
        X = np.dot(A, B)

    X[X > 0] = 1

    return X

def random_binary_matrix(n, m, rank, p_sparsity=0.50, p_noise=0.00, fixed_seed=0):
    """
    Generate a random binary matrix with specified Boolean rank, sparsity and noise

    :param n:           int, number of rows
    :param m:           int, number of columns
    :param rank:        int, max Boolean rank before noise is introduced
    :param p_sparsity:  float in [0,1], default=0.5, sparsity of X, as the probability of an entry x_ij is zero
    :param p_noise:     float in [0,1], default=0.0, noise in X, as the probability of an entry x_ij is flipped
    :param fixed_seed:  int, default=0, seed passed onto random number generator

    :return X:          n x m binary matrix X, that is at most p_noise * n * m squared Frobenius distance
                        away from an n x m binary matrix of Boolean rank=rank
    """

    # random number generation seed
    np.random.seed(fixed_seed)

    # sparsity
    q = 1 - np.sqrt(1 - p_sparsity**(1 / rank))  # probability of 0's in a Bernoulli trial
    p = 1 - q                                    # probability of 1's in a Bernoulli trial

    # initial X with Boolean rank at most = rank
    A = np.random.binomial(1, p, [n, rank])      # generating n x rank A with probability p for a 1
    B = np.random.binomial(1, p, [rank, m])
    X = boolean_matrix_product(A, B)

    # introducing noise
    for noise in range(int(p_noise * n * m)):    # at most p_noise * n * m entries perturbed in X
        i = np.random.choice(n)
        j = np.random.choice(m)                  # if same i, j comes up, noise is less

        X[i,j] = np.abs(X[i,j] - 1)

    return X

def preprocess(X):
    """
    Eliminate duplicate and zero rows/columns of a binary matrix but keep a record of them

    :param X:               n x m binary matrix with possibly missing entries set to np.nan

    Returns
    -------
    X_out :                 preprocessed matrix
    row_weights :           the number of times each unique row of X_out is repated in X
    col_weights :           the number of times each unique column of X_out is repated in X
    idx_rows_reconstruct :  X_out[idx_rows_reconstruct, :] puts row duplicates back
    idx_cols_reconstruct :  X_out[:, idx_cols_reconstruct] puts column duplicates back
    idx_zero_row :          index of zero row if any
    idx_zero_col :          index of zero col if any
    idx_rows_unique :       X[idx_rows_unique, :] eliminates duplicate rows of X
    idx_cols_unique :       X[:, idx_cols_unique] eliminates duplicate cols of X
    """

    X_in = np.copy(X)

    # if the matrix has missing entries, make them numerical so they can be subject to preprocessing
    if np.isnan(X_in).sum() > 0:
        is_missing = True
        X_in[np.isnan(X_in)] = -100
    else:
        is_missing = False

    #(1) operations on rows
    # delete row duplicates but record their indices and counts
    X_unique_rows, idx_rows_unique, idx_rows_reconstruct, row_weights = np.unique(X_in,
                                                                       return_index=True,
                                                                       return_inverse=True,
                                                                       return_counts=True,
                                                                       axis=0)
    is_zero_row = np.all(X_unique_rows == 0, axis=1)  # find the row of all zeros if it exists
    if is_zero_row.sum() > 0:
        X_unique_nonzero_rows = X_unique_rows[~is_zero_row]  # delete that row of zeros
        row_weights = row_weights[~is_zero_row]  # delete the count of zero rows
        idx_zero_row = np.where(is_zero_row)[0][0]  # get index of zero row
    else:
        X_unique_nonzero_rows = X_unique_rows
        idx_zero_row = None

    #(2) operations on columns
    # delete column duplicates but record their indices and counts
    X_unique_cols, idx_cols_unique, idx_cols_reconstruct, col_weights = np.unique(X_unique_nonzero_rows,
                                                                       return_index=True,
                                                                       return_inverse=True,
                                                                       return_counts=True,
                                                                       axis=1)

    is_zero_col = np.all(X_unique_cols == 0, axis=0)  # find the col of all zeros if it exists
    if is_zero_col.sum() > 0:
        X_unique_nonzero_cols = np.transpose(np.transpose(X_unique_cols)[~is_zero_col])
        # delete that col of zeros
        col_weights = col_weights[~is_zero_col]  # delete the count of zero columns
        idx_zero_col = np.where(is_zero_col)[0][0]  # store the index of zero column
    else:
        X_unique_nonzero_cols = X_unique_cols
        idx_zero_col = None
    X_out = X_unique_nonzero_cols

    # if the matrix had missing entries, turn these back to nan
    if is_missing:
        X_out[X_out == -100] = np.nan

    return X_out, row_weights, col_weights, idx_rows_reconstruct,\
           idx_cols_reconstruct, idx_zero_row, idx_zero_col, idx_rows_unique, idx_cols_unique

def un_preprocess(X, idx_rows_reconstruct, idx_zero_row, idx_cols_reconstruct, idx_zero_col):
    """
    Place back duplicate and zero rows and columns, the inverse fucntion of preprocess(X)

    :param X:                           preprocessed binary matrix
    :param idx_rows_reconstruct:        X_out[idx_rows_reconstruct, :] puts row duplicates back
    :param idx_zero_row:                index of zero row if any
    :param idx_cols_reconstruct:        X_out[:, idx_cols_reconstruct] puts column duplicates back
    :param idx_zero_col:                index of zero column if any

    :return X_out:                      binary matrix containing duplicate and zero rows and columns
    """

    #(1) operations on columns
    n, m = X.shape
    #(1.1) if there was a zero column removed add it back
    if idx_zero_col is None:
        X_zero_col = np.copy(X)
    else:
        X_zero_col = np.concatenate((X[:, 0:idx_zero_col],
                                     np.zeros([n, 1], dtype=int),
                                     X[:, idx_zero_col:m]), axis=1)
    # create duplicates of columns
    X_orig_cols = X_zero_col[:, idx_cols_reconstruct]

    #(2) operations on rows
    [n_tmp, m_tmp] = X_orig_cols.shape
    # if there was a zero row removed add it back
    if idx_zero_row is None:
        X_zero_row = X_orig_cols
    else:
        X_zero_row = np.concatenate((X_orig_cols[0:idx_zero_row, :],
                                     np.zeros([1, m_tmp], dtype=int),
                                     X_orig_cols[idx_zero_row:n_tmp, :]), axis=0)
    # create duplicates of rows
    X_out = X_zero_row[idx_rows_reconstruct, :]

    return X_out

def post_process_factorisation(A_in, B_in, idx_rows_reconstruct, idx_zero_row, idx_cols_reconstruct, idx_zero_col):
    """
    Place back duplicate and zero rows in A_in and duplicate and zero columns in B_in

    :param A_in:                        binary matrix to be extended with duplicate and zero rows
    :param B_in:                        binary matrix to be extended with duplicate and zero columns
    :param idx_rows_reconstruct:        A_in[idx_rows_reconstruct, :] puts row duplicates back
    :param idx_zero_row:                index of zero row if any
    :param idx_cols_reconstruct:        B_in[:, idx_cols_reconstruct] puts column duplicates back
    :param idx_zero_col:                index of zero column if any

    :return A:                          binary matrix with duplicate and zero rows
    :return B:                          binary matrix with duplicate and zero columns
    """

    #(1) operations on columns of B
    [n_tmp, m_tmp] = B_in.shape
    # if there was a zero column removed add it back
    if idx_zero_col is None:
        B_zero_col = np.copy(B_in)
    else:
        B_zero_col = np.concatenate((B_in[:, 0:idx_zero_col],
                                     np.zeros([n_tmp, 1], dtype=int),
                                     B_in[:, idx_zero_col:m_tmp]),
                                    axis=1)
    # create duplicates of columns
    B = B_zero_col[:, idx_cols_reconstruct]


    #(2) operations on rows of A
    [n_tmp, m_tmp] = A_in.shape
    # if there was a zero row removed add it back
    if idx_zero_row is None:
        A_zero_row = np.copy(A_in)
    else:
        A_zero_row = np.concatenate((A_in[0:idx_zero_row, :],
                                     np.zeros([1, m_tmp], dtype=int),
                                     A_in[idx_zero_row:n_tmp, :]),
                                    axis=0)
    # create duplicates of rows
    A = A_zero_row[idx_rows_reconstruct]

    return A, B


# heuristics
def BBQP_greedy_heur(Q_in, perturbed=False, transpose=False, revised=False, r_seed=None):
    """
    Greedy algorithm for the Bipartite Binary Quadratic Program: max_a,b a^T Q_in b where a, b are constrained binary
    Computing a rank-1 binary matrix ab^T that picks up the max weight of Q_in
    1. Sorts the rows of Q_in in decreasing order according to their sum of positive weights
    2. Aims to set a_i=1 for rows that increase the cumulative weight
    3. Sets b based on a

    :param Q_in:            n x m real matrix
    :param perturbed:       bool, default=False, perturb the original ordering of rows of Q_in
    :param transpose:       bool, default=False, use the transpose of Q_in
    :param revised:         bool, default=False, break ties in the original ordering by comparing negative sums
    :param r_seed:          int, default=None, use a random ordering

    :return a:              n dimensional binary vector
    :return b:              m dimensional binary vector
    """

    if transpose:
        # work on the transpose of the matrix
        Q = np.transpose(Q_in)
    else:
        Q = Q_in

    n, m = Q.shape

    if r_seed is None:
        # sum of positive weights in each row
        Q_pos = np.copy(Q)
        Q_pos[Q < 0] = 0
        w_pos = Q_pos.sum(axis=1)

        if perturbed:
            # slightly perturbed positive weights
            w_pos = w_pos * np.random.uniform(0.9, 1.1, n)

        if revised:
            # sum of negative weights in each row
            Q_neg = np.copy(Q)
            Q_neg[Q > 0] = 0
            w_neg = Q_neg.sum(axis=1)
            # sort w_pos in decreasing order, and resolve ties with sorting of w_neg in decreasing order
            sorted_i = np.lexsort((w_neg, w_pos))[::-1]
        else:
            # simply sort sort w_pos in decreasing order -- original ordering
            sorted_i = np.argsort(w_pos)[::-1]
    else:
        # use random ordering of rows
        np.random.seed(r_seed)
        sorted_i = np.random.permutation(n)

    a = np.zeros(n, dtype=int)
    s = np.zeros(m)
    for i in sorted_i:
        f_0 = np.sum(s[s >= 0])
        s_plus_Q = s + Q[i, :]
        f_1 = np.sum(s_plus_Q[s_plus_Q >= 0])
        if f_0 < f_1:
            a[i] = 1
            s = s + Q[i, :]

    b = np.zeros(m, dtype=int)
    b[s > 0] = 1

    if transpose:
        return b, a
    else:
        return a, b

def BBQP_alternating_heur(Q_in, a_in, b_in, transpose=False):
    """
    Alternating iterative algorithm for the Bipartite Binary Quadratic Program with a starting point a_in, b_in
    max_a,b a^T Q_in b where a, b are constrained binary
    Computing a rank-1 binary matrix ab^T that picks up the max weight of Q_in
    Sets a based on b_in or b based on a_in and then alternates

    :param Q_in:                        n x m real matrix
    :param a_in:                        n dimensional binary vector
    :param b_in:                        m dimensional binary vector
    :param transpose:                   bool, default=False, work on the transpose of Q_in

    :return a:                          n dimensional binary vector
    :return b:                          m dimensional binary vector
    """
    if transpose:
        # work on the transpose of the matrix
        Q = np.transpose(Q_in)
        a = b_in
        b = a_in
    else:
        Q = Q_in
        a = a_in
        b = b_in

    while True:
        idx_a = np.dot(Q, b) > 0
        a_new = np.zeros(Q.shape[0], dtype=int)
        a_new[idx_a] = 1
        if np.array_equal(a, a_new):
            break
        else:
            a = a_new

        idx_b = np.dot(a, Q) > 0
        b_new = np.zeros(Q.shape[1], dtype=int)
        b_new[idx_b] = 1
        if np.array_equal(b, b_new):
            break
        else:
            b = b_new

    if transpose:
        return b, a
    else:
        return a, b

def BBQP_mixed_heurs(Q, num_rand = 30):
    """
    Computes several variations of the Greedy and Alternating algorithms for the Bipartite Binary Quadratic Program

    :param Q_in:                        n x m real matrix
    :param num_rand:                    int, number of random ordering Greedy+Alternating algorithm to compute

    :return A:                          n x (8 + num_rand) binary matrix, each column for a different heur sol
    :return B:                          (8 + num_rand) x m binary matrix, each row for a different heur sol
    """

    (n,m) = Q.shape
    A = np.zeros((n, 8 + num_rand), dtype=int)
    B = np.zeros((8 + num_rand, m), dtype=int)

    # original
    a, b = BBQP_greedy_heur(Q)
    a, b = BBQP_alternating_heur(Q, a, b)
    A[:, 0] = a
    B[0, :] = b

    # original perturbed
    a, b = BBQP_greedy_heur(Q, perturbed=True)
    a, b = BBQP_alternating_heur(Q, a, b)
    A[:, 1] = a
    B[1, :] = b

    # original transpose
    a, b = BBQP_greedy_heur(Q, transpose=True)
    a, b = BBQP_alternating_heur(Q, a, b, transpose=True)
    A[:, 2] = a
    B[2, :] = b

    # original perturbed transpose
    a, b = BBQP_greedy_heur(Q, perturbed=True, transpose=True)
    a, b = BBQP_alternating_heur(Q, a, b, transpose=True)
    A[:, 3] = a
    B[3, :] = b

    # revised
    a, b = BBQP_greedy_heur(Q, revised=True)
    a, b = BBQP_alternating_heur(Q, a, b)
    A[:, 4] = a
    B[4, :] = b

    # revised perturbed
    a, b = BBQP_greedy_heur(Q, revised=True, perturbed=True)
    a, b = BBQP_alternating_heur(Q, a, b)
    A[:, 5] = a
    B[5, :] = b

    # revised transpose
    a, b = BBQP_greedy_heur(Q, revised=True, transpose=True)
    a, b = BBQP_alternating_heur(Q, a, b, transpose=True)
    A[:, 6] = a
    B[6, :] = b

    # revised perturbed transpose
    a, b = BBQP_greedy_heur(Q, revised=True, perturbed=True, transpose=True)
    a, b = BBQP_alternating_heur(Q, a, b, transpose=True)
    A[:, 7] = a
    B[7, :] = b

    # random
    for i in range(num_rand):
        a, b = BBQP_greedy_heur(Q, r_seed=i)
        a, b = BBQP_alternating_heur(Q, a, b)
        A[:, 8 + i] = a
        B[8 + i, :] = b

    return A, B

def BBQP_mixed_heur(Q, num_rand=30):
    """
    Computes several variations of the Greedy and Alternating algorithms for the Bipartite Binary Quadratic Program
    Returns the best one among them

    :param Q_in:                        n x m real matrix
    :param num_rand:                    int, number of random ordering Greedy+Alternating algorithm to compute

    :return a:                          n dimensional binary vector, with best objective value
    :return b:                          m dimensional binary vector, with best objective value
    """

    A, B = BBQP_mixed_heurs(Q, num_rand=num_rand)

    # compute objective of each heur sol
    obj = np.array([np.dot(A[:, l], np.dot(Q, B[l, :])) for l in range(A.shape[1])])
    l = np.argmax(obj)
    
    

    #================================================================================================================================
    """ obj2 = []
        for l in range(A.shape[1]):
            a = A[:, l]
            a.shape = a.shape[0], 1
            b = B[l, :]
            b.shape = b.shape[0], 1
            Q_new = a @ b.T
            Q_new[Q_new > 0] = 1
            Z = Q + 1
            Z[Z > 0] = 1
            cm = confusion_matrix(Z.ravel(), Q_new.ravel())
            TN, FP, FN, TP = confusion_matrix(Z.ravel(), Q_new.ravel()).ravel()
            metric = 0.5 * (1 + (TP / (TP + FN)) - (FP / (FP + TN)))
            obj2.append(metric)
        l = np.argmax(obj2)
    """
    #print('BEST', f"{l=}")
    #================================================================================================================================
    
    return A[:, l], B[l, :]

def BMF_k_greedy_heur(X, k, row_weights=None, col_weights=None,
                      mixed=True, revised=False, perturbed=False, transpose=False, r_seed=None, add = 0):
    """
    Greedy algorithm for the rank-k Binary Matrix Factorisation problem
    Computes k rank-1 binary matrices sequentially which give the 1-best coverage of X with least 0-coverage
    Uses the greeady and alternating heuristics for Bipartite Binary Quadratic Programs as subroutine

    Parameters
    ----------
    X :                 n x m binary matrix with possibly missing entries set to np.nan
    k :                 int, rank of factorisation
    row_weights :       n dimensional array, weights to put on rows
    col_weights :       m dimensional array, weights to put on columns
    mixed:              bool, default=True,
                        use BBQP_mixed_heur(Q, num_rand=30) as subroutine
    perturbed:          bool, default=False,
                        use BBQP_greedy_heur(Q, perturbed=True) + alternating heuristic as subroutine
    transpose:          bool, default=False,
                        use BBQP_greedy_heur(Q, transpose=True) + alternating heuristic as subroutine
    revised:            bool, default=False,
                        use BBQP_greedy_heur(Q, revised=True) + alternating heuristic as subroutine
    r_seed:             int, default=None,
                        use BBQP_greedy_heur(Q, r_seed=r_seed) + alternating heuristic as subroutine

    Returns
    -------
    A:                  n x k  binary matrix
    B:                  k x m  binary matrix
    """
    #print("HelloWorld"*10)
    (n, m) = X.shape

    if row_weights is None or col_weights is None:
        row_weights = np.ones(n, dtype=int)
        col_weights = np.ones(m, dtype=int)

    A = np.zeros([n, k], dtype=int)
    B = np.zeros([k, m], dtype=int)

    Q_sign = np.zeros((n, m), dtype=int)
    Q_sign[X == 1] = 1
    Q_sign[X == 0] = -1
    Weights = np.outer(row_weights, col_weights)
    Q_orig = Q_sign * Weights
    Q = np.copy(Q_orig)
    #print('pfff7')
    for i in range(k + add):
        if mixed:
            a, b = BBQP_mixed_heur(Q, 30)
            #a, b = BBQP_greedy_heur(Q, perturbed, transpose, revised, r_seed)
            #a, b = BBQP_alternating_heur(Q, a, b, transpose)
        else:
            a, b = BBQP_greedy_heur(Q, perturbed, transpose, revised, r_seed)
            a, b = BBQP_alternating_heur(Q, a, b, transpose)
        # ================================ Addon iteration =======================================================
        if i >= k:
            #print('ээээ чего?===============================================')
            l = np.zeros(k, dtype=int)
            for q in range(k):
                a_new = A[:, q]+a
                a_new[a_new > 0] = 1
                b_new = B[q, :]+b
                b_new[b_new > 0] = 1
                l[q] = np.dot(a_new, np.dot(Q, b_new))
            i_new = np.argmax(l)
            #print(i, i_new, l) #, A[:, i_new], a, sep='\n')
            a = A[:, i_new] + a
            a[a>0] = 1
            b = B[i_new, :]+b
            b[b>0] = 1
        else:
            i_new = i
        # ========================================================================================================
        A[:, i_new] = a
        B[i_new, :] = b
        idx_covered = boolean_matrix_product(A, B) == 1
        Q[idx_covered] = 0



        ## objective value in squared frobenius norm
        # print(Q_orig[Q_orig > 0].sum() - Q_orig[boolean_matrix_product(A, B) == 1].sum())

    return A, B


if __name__ == "__main__":

    # binary matrix generation, preprocessing, post-processing
    XX = random_binary_matrix(20, 10, 10, 0.5, 0, 0)
    print('The original dimension is \t\t\t', XX.shape)

    XX_out, row_weights, col_weights, idx_rows_reconstruct, idx_cols_reconstruct,\
    idx_zero_row, idx_zero_col, idx_rows_unique, idx_cols_unique = preprocess(XX)
    print('The preprocessed dimension is \t\t', XX_out.shape)

    YY = un_preprocess(XX_out, idx_rows_reconstruct, idx_zero_row, idx_cols_reconstruct, idx_zero_col)
    print('The un-preprocessed dimension is \t', YY.shape)
    print('The original and output matrix are equal ', np.array_equal(XX, YY))

    kk = 3

    # k-Greedy heuristic for rank-k binary matrix factorisation
    A_greedy, B_greedy = BMF_k_greedy_heur(XX, kk)


    print('\n\nThe rank-%s BMF via k-Greedy has error \t\t\t\t\t\t%s' %(kk, np.sum(np.abs(XX - boolean_matrix_product(A_greedy, B_greedy)))))


