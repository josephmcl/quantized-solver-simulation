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

def quantize_product(A, B, b):
    """ quantize A and B row wise and return their quantized versions:"""
    A_hat = np.empty_like(A)
    B_hat = np.empty_like(B)
    for index, row in enumerate(A):
        # r is the range to be used, in this case let r be the maximum magnitude of the row
        r = np.max(np.abs(row))
        row_hat = quantize_row(row, r, b)
        A_hat[index, :] = row_hat
    for index, row in enumerate(B):
        r = np.max(np.abs(row))
        row_hat = quantize_row(row, r, b)
        B_hat[index, :] = row_hat
    return (A_hat, B_hat)

def predicted_error(A, B, v_A, v_B):
    """ total squared error can be predicted using thm 3.3

    the column energies of B are the sums of the squares of the entries in the rows of B. the A_term
    is the sum over i,k of (v_A)_(ik) ||B_(k,:)|| 
     """
    row_energies_B = np.sum((B**2), axis=1)
    weighted_A = v_A * row_energies_B
    A_term = np.sum(weighted_A)

    """ the column energies of A are the sums of the squares of the entries in the /columns/ of A
    and therefore used axis=0. the result also needs to be converted to a column vector to match
    the dimension of the variance matrix. the B_term is the sum over k,j of (v_B)_(kj) ||A_(:,k)|| 
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

def main():
    np.random.seed(12)
    A = np.random.randn(128, 128) *2
    B = np.random.randn(128, 128) *2
    (A_hat, B_hat) = quantize_product(A, B, 8)
    exact = A @ B
    quantized = A_hat @ B_hat
    b = 8
    v_A = variance(A,b)
    v_B = variance(B,b)
    predicted = predicted_error(A,B,v_A,v_B)
    
    #print("Exact product AB = ", "\n", exact, "\n")
    #print("product of quantized A_hat B_hat = ", "\n" , quantized, "\n")
    print("Actual error = ", np.sum((exact - quantized)**2))
    print("Predicted error = ", predicted)
    

main()