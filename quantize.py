import numpy as np

def quantize_row(a, R, b):
    '''
    map each value to range [-R, R] using signed b-bit format where b geq 2
    
    the integer range for a signed b-bit value is [-(2^(b-1)), 2^(b-1) - 1] but the paper
    uses a uniform range [-(2^(b-1) - 1), 2^(b-1) - 1]
    '''
    r_min = -(2 ** (b - 1) - 1)
    r_max = (2 ** (b - 1) - 1)
    # scale factor:
    delta = R/(r_max)
    # divide each value by the scale factor and constrain values to the (uniform) range of a b-bit ints
    a = np.rint(np.divide(a,delta))
    a = np.clip(a, r_min, r_max)
    # a_hat is the quantized a, i.e. a multiplied back by the scale factor:
    a_hat = a * delta
    return a_hat

def quantize_by_block(A, b, s):
    """ quantize A block wise where s is the size of the block """
    A_hat = np.empty_like(A, dtype=float)
    m = A.shape[0]
    for i in range(0, m, s):
        # r is computed for the entire block of rows i though i+s:
        r = np.max(np.abs(A[i:i+s]))
        for j in range(i, min((i + s), m)):
            # each row in the block uses the same r
            row_hat = quantize_row(A[j], r, b)
            A_hat[j, :] = row_hat
    return A_hat
    
def quantize_by_row(A, b, s = 0):
    """ quantize A row wise """
    A_hat = np.empty_like(A, dtype=float)
    for index, row in enumerate(A):
        # r is the range to be used, in this case let r be the maximum magnitude of the row
        r = np.max(np.abs(row))
        row_hat = quantize_row(row, r, b)
        A_hat[index, :] = row_hat
    return A_hat

def quantize_product(A, B, b, by_scale, s = 0):
    """ quantize A and B according to specified scaling and return their quantized versions:"""
    A_hat = by_scale(A,b, s)
    B_hat = by_scale(B,b, s)
    return (A_hat, B_hat)

def range_rule(A,B):
    """ compute values h_k for each k according to range rule (section 4.3) """
    # h is a vector of dimension matching the number of columns of A
    n = A.shape[1]
    h = np.zeros(n)
    for k in range(n):
        max_col_A = np.max(np.abs(A[: , k]))
        max_row_B = np.max(np.abs(B[k , :]))
        # technically h_k may be proportional to the rhs, not necessarily equal
        h[k] = np.sqrt(max_row_B / max_col_A)
    return h

def norm_rule(A,B):
    """ compute h_k according to the norm rule (sectionm 4.3) """
    n = A.shape[1]
    h = np.zeros(n)
    for k in range(n):
        norm_col_A = np.linalg.norm(A[: , k])
        norm_row_B = np.linalg.norm(B[k , :])
        # again h_k might be scaled by some factor later, not necessarily equal
        h[k] = np.sqrt(norm_row_B / norm_col_A)
    return h

def apply_contraction(A, B, rule):
    """ compute and apply specified contraction to A and B"""
    T = np.diag(rule(A, B))
    A_prime = A @ T
    B_prime = np.linalg.inv(T) @ B
    return (A_prime, B_prime)

def variance(A, b):
    """ the entries in the variance field v^A depend of the range, in this case they are per row, not per 
    individual entry. the predicted variance of a row of A is equal to delta^2/12 where delta = R/(2^(b-1) - 1)
    (section 3.1)
    """
    var_A = np.empty_like(A, dtype=float)
    for index, row in enumerate(A):
        r = np.max(np.abs(row))
        delta = r / (2**(b - 1) - 1)
        var_predicted = (delta ** 2) / 12
        var_A[index, :] = var_predicted
    return var_A

def predicted_error(A, B, v_A, v_B):
    """ total squared error can be predicted using thm 3.3

    the row energies of B are the sums of the squares of the entries in the rows of B. the A_term
    is the sum over i,k of (v_A)_(ik) ||B_(k,:)||^2
     """
    row_energies_B = np.sum((B**2), axis=1)
    weighted_A = v_A * row_energies_B
    A_term = np.sum(weighted_A)

    """ the column energies of A are the sums of the squares of the entries in the /columns/ of A
    and therefore use axis=0. the result also needs to be converted to a column vector to match
    the dimension of the variance matrix. the B_term is the sum over k,j of (v_B)_(kj) ||A_(:,k)||^2 
    """
    col_energies_A = (np.sum((A**2), axis=0)).reshape(-1,1)
    weighted_B = v_B * col_energies_A
    B_term = np.sum(weighted_B)

    """ the third term is error contributed by both factors. take the product of the sum accross 
    the columns of v_A and accross the rows of v_B, then the total contribution is the sum of the 
    elements of that vector 
    """
    cols_vA = np.sum(v_A, axis=0)
    rows_vB = np.sum(v_B, axis=1)
    C_prod = cols_vA * rows_vB
    C_term = np.sum(C_prod)

    error = A_term + B_term + C_term
    return error
