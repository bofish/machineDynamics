import numpy as np
import numpy.matlib
from numpy.linalg import lstsq 
from scipy.optimize import fsolve
import pandas as pd
import matplotlib.pyplot as plt

def fsFun(x):
    return 2*np.sin(x) - np.exp(x)/4 - 1
    
def dfsFun(x):
    return 2*np.cos(x) - np.exp(x)/4

def make_coeff_table_df(km_array, p_array, diff_coeff):
    tuples = list(zip(*km_array))
    index = pd.MultiIndex.from_tuples(tuples, names=['k', 'm'])
    coeff_df = pd.DataFrame(diff_coeff, columns=p_array, index=index)
    coeff_df.columns.name = 'p'
    return coeff_df

def count_sample_number(km_array, diff_coeff):
    sample_number = np.count_nonzero(diff_coeff, axis=1)
    tuples = list(zip(*km_array))
    index = pd.MultiIndex.from_tuples(tuples, names=['k', 'm'])
    sample_number_s = pd.Series(sample_number, index=index)

    return sample_number_s

def cal_error(f, f_hat):
    return np.sqrt((f-f_hat)**2)

def forward_diff(x, f, k, m, diff_origin=None):
    '''
    Limitation:
    1. For `k`-rd derivative
    2. `h` is evenly spacing
    3. Output length less than len(f), sample number caused
    4. If len(df) = 100, len(f_diff)=95, means f_diff[0:94] is match to df[0:94]
    '''
    # Table of Coefficients 
    km_array = [[1, 1, 2, 2, 3, 3, 4, 4],
               [1, 2, 1, 2, 1, 2, 1, 2]]
    p_array = [0, 1, 2, 3, 4, 5]
    forward_diff_coeff = np.array([
        [-1, 1, 0, 0, 0, 0],
        [-3/2, 2, -1/2, 0, 0, 0],
        [1, -2, 1, 0, 0, 0],
        [2, -5, 4, -1, 0, 0],
        [-1, 3, -3, 1, 0, 0],
        [-5/2, 9, -12, 7, -3/2, 0],
        [1, -4, 6, -4, 1, 0],
        [3, -14, 26, -24, 11, -2],
    ])
    coeff_df = make_coeff_table_df(km_array, p_array, forward_diff_coeff)
    sample_number = count_sample_number(km_array, forward_diff_coeff)
    sn = sample_number.loc[(k,m)]

    M = len(x)
    f_diff = []
    h = np.sqrt(x[-1] - x[0])**2/(M-1)
    for j in range(M-sn+1):
        # single_pt = [coeff_df.loc[(k,m),p]*f[j+p] for p in range(6) if p<=M-j-1]
        single_pt = [coeff_df.loc[(k,m),p]*f[j+p] for p in range(sn)]
        dnf_j = sum(single_pt)/h**k
        f_diff.append(dnf_j)
    f_diff = np.array(f_diff)
    x_paired = np.array(x[0:M-sn+1])
    error = cal_error(diff_origin[0:M-sn+1], f_diff) if diff_origin is not None else ['Without derivative of f(x) inputed']
    return x_paired, f_diff, error

def backward_diff(x, f, k, m, diff_origin=None):
    '''
    Limitation:
    1. For `k`-rd derivative
    2. `h` is evenly spacing
    3. Output length less than len(f), sample number caused
    4. If len(df) = 100, len(f_diff)=95, means f_diff[5:99] is match to df[5:99]
    '''
    # Table of Coefficients 
    km_array = [[1, 1, 2, 2, 3, 3, 4, 4],
                [1, 2, 1, 2, 1, 2, 1, 2]]
    p_array = [-5, -4, -3, -2, -1, 0]
    backward_diff_coeff = np.array([
        [0, 0, 0, 0, -1, 1],
        [0, 0, 0, 1/2, -2, 3/2],
        [0, 0, 0, 1, -2, 1],
        [0, 0, -1, 4, -5, 2],
        [0, 0, -1, 3, -3, 1],
        [0, 3/2, -7, 12, -9, 5/2],
        [0, 1, -4, 6, -4, 1,],
        [-2, 11, -24, 26, -14, 3],
    ])
    coeff_df = make_coeff_table_df(km_array, p_array, backward_diff_coeff)
    sample_number = count_sample_number(km_array, backward_diff_coeff)
    sn = sample_number.loc[(k,m)]

    M = len(x)
    f_diff = []
    h = np.sqrt(x[-1] - x[0])**2/(M-1)
    for j in range(sn-1, M):
        single_pt = [coeff_df.loc[(k,m),p]*f[j+p] for p in range(0, -sn, -1)]
        dnf_j = sum(single_pt)/h**k
        f_diff.append(dnf_j)
    f_diff = np.array(f_diff)
    x_paired = np.array(x[sn-1:M])
    error = cal_error(diff_origin[sn-1:M], f_diff) if diff_origin is not None else ['Without derivative of f(x) inputed']
    return x_paired, f_diff, error

def central_diff(x, f, k, m, diff_origin=None):
    '''
    Limitation:
    1. For `k`-rd derivative
    2. `h` is evenly spacing
    '''
    # Table of Coefficients 
    km_array = [[1, 1, 2, 2, 3, 3, 4, 4],
               [2, 4, 2, 4, 2, 4, 2, 4]]
    p_array = [-3, -2, -1, 0, 1, 2, 3]
    central_diff_coeff = np.array([
        [0, 0, -1/2, 1.0e-310, 1/2, 0, 0],
        [0, 1/12, -2/3, 1.0e-310, 2/3, -1/12, 0],
        [0, 0, 1, -2, 1, 0, 0],
        [0, -1/12, 4/3, -5/2, 4/3, -1/12, 0],
        [0, -1/2, 1, 1.0e-310, -1, 1/2, 0],
        [1/8, -1, 13/8, 1.0e-310, -13/8, 1, -1/8],
        [0, 1, -4, 6, -4, 1, 0],
        [-1/6, 2, -13/2, 28/3, -13/2, 2, -1/6],
    ])
    coeff_df = make_coeff_table_df(km_array, p_array, central_diff_coeff)
    sample_number = count_sample_number(km_array, central_diff_coeff)
    sn = sample_number.loc[(k,m)]

    M = len(x)
    f_diff = []
    h = np.sqrt(x[-1] - x[0])**2/(M-1)
    side_len = int((sn-1)/2)
    for j in range(side_len, M - side_len):
        single_pt = [coeff_df.loc[(k,m),p]*f[j+p] for p in range(-side_len, side_len+1)]
        dnf_j = sum(single_pt)/h**k
        f_diff.append(dnf_j)
    f_diff = np.array(f_diff)
    x_paired = np.array(x[side_len: M - side_len])
    error = cal_error(diff_origin[side_len: M - side_len], f_diff) if diff_origin is not None else ['Without derivative of f(x) inputed']
    return x_paired, f_diff, error

def trapezoidal_integ(x, f, integ_origin=None):
    M = len(x)
    if x[-1] == x[0]:
        return 0
    h = np.sqrt((x[-1] - x[0])**2)/(M-1)
    f_integ = ((f[0] + f[-1])/2 + sum(f[1:-2]))*h
    error = cal_error(integ_origin, f_integ) if integ_origin is not None else ['Without derivative of f(x) inputed']
    return f_integ

def simpson13_integ(x, f, integ_origin=None):
    '''
    Limitation:
    1. `N` must be odd number
    2. `N` must be larger than or eqaul to 5
    '''
    M = len(x)
    h = np.sqrt((x[-1] - x[0])**2)/(M-1)
    mid = [2*f[2*k+1] + f[2*k+2] for k in range(int((M-5)/2))]
    f_integ = (f[0] + 4*f[-2] + f[-1] + 2*sum(mid))*h/3
    error = cal_error(integ_origin, f_integ) if integ_origin is not None else ['Without derivative of f(x) inputed']
    return f_integ

def gauss_integ(x, f, integ_origin=None):
    '''
    Limitation:
    1. w(x) = 1
    '''
    M = len(x)
    vandermonder_mat = np.matlib.ones((M,M))
    B = np.matlib.zeros((M,1))
    for j in range(M):
        vandermonder_mat[j][:] = [x[i]**j for i in range(M)]
        B[j] = (x[-1]**(j+1) - x[0]**(j+1))/(j+1)
    W = lstsq(vandermonder_mat, B, rcond=-1)[0]
    f_integ = sum(f[j]*W.item(j) for j in range(M))
    error = cal_error(integ_origin, f_integ) if integ_origin is not None else ['Without derivative of f(x) inputed']
    return f_integ

def bisection_root(x_l, x_u, f, max_iter_count=1e5, error_tol = 1e-15):
    a = x_l
    b = x_u
    c = 0
    iter_count = 0
    exit_message = 'Root found'

    while f(c)**2 > error_tol:
        c = (a + b)/2
        if f(a)*f(c) < 0:
            (a, b) = (a, c)
        elif f(c)*f(b) < 0:
            (a, b) = (c, b)
        iter_count += 1
        
        if iter_count >= max_iter_count:
            exit_message = 'Over max number of iterations'
            break
        
    output = {
        'x_root': c,
        'f_val': f(c),
        'iterations': iter_count,
        'exit_message': exit_message
    }
    return output
    
def secant_root(x_0, x_1, f, max_iter_count=1e5, error_tol = 1e-15):
    x_old = x_0
    x_new = x_1
    x_root = 0
    f_val = 0
    iter_count = 0
    exit_message = 'Root found'

    while f(x_root)**2 > error_tol:
        x_root = x_new - f(x_new)*((x_new - x_old) / (f(x_new) - f(x_old)))
        x_old = x_new
        x_new = x_root
        iter_count += 1

        if iter_count >= max_iter_count:
            exit_message = 'Over max number of iterations'
            break
        
    output = {
        'x_root': x_root,
        'f_val': f(x_root),
        'iterations': iter_count,
        'exit_message': exit_message
    }
    return output

def newton_root(x_0, f, df, max_iter_count=1e5, error_tol = 1e-15):
    x_n = x_0
    iter_count = 0
    exit_message = 'Root found'

    while f(x_n)**2 > error_tol:
        x_n = x_n - f(x_n)/df(x_n)
        iter_count += 1

        if iter_count >= max_iter_count:
            exit_message = 'Over max number of iterations'
            break
        
    output = {
        'x_root': x_n,
        'f_val': f(x_n),
        'iterations': iter_count,
        'exit_message': exit_message
    }
    return output







