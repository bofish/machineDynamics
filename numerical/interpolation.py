import numpy as np
import numpy.matlib
from numpy.linalg import lstsq
from math import pi, cos
import matplotlib.pyplot as plt
from functools import reduce

'''
TODO:
1. Re-written vandermode method 
'''

def prod(iterable):
    '''
    https://stackoverflow.com/questions/595374/whats-the-python-function-like-sum-but-for-multiplication-product
    >>> prod(range(1, 5))
    24
    '''
    return reduce(lambda x,y: x*y, iterable, 1)

def evenly_space_pt(a,b,N):
    x = np.linspace(a, b, N)
    f = np.cos(10*x**2)*np.exp(-x)
    return x, f

def chebyshev_pt(a, b, N):
    x = np.array([(a+b)/2 - (b-a)/2*cos(k*pi) for k in np.linspace(0,1,N)])
    f = np.cos(10*x**2)*np.exp(-x)
    return x, f

def origin_fun(x, a, b):
    f = np.cos(10*x**2)*np.exp(-x)
    return f

def polynomial_vandermode(x, f, x_sub):
    M = len(x)
    vandermonder_mat = np.matlib.ones((M,M))
    B = np.matlib.zeros((M,1))
    for j in range(M):
        vandermonder_mat[j][:] = [x[j]**i for i in range(M)]
        B[j] = [f[j]]
    coeff = lstsq(vandermonder_mat, B, rcond=-1)[0]
    HOT = [coeff.item(i)*x_sub**i for i in range(1,M)]
    f_hat = coeff.item(0)*np.ones(len(x_sub)) + np.sum(HOT, axis=0)
    return f_hat

def polynomial_lagrange(x, f):
    M = len(x)
    poly = np.poly1d(0.0)
    for j in range(M):
        single_item = np.poly1d(f[j])
        for k in range(M):
            if k == j:
                continue
            fac = x[j] - x[k]
            single_item *= np.poly1d([1.0, -x[k]])/fac
        poly += single_item
    return poly

def polynomial_newton(x, f):
    M = len(x)
    # Calculate Coefficient
    c = [f[0]]
    for i in range(1,M):
        fac = 0
        for j in range(1,i):
            fac += c[j]*prod(x[i] - x[0:j])
        c.append((f[i] - c[0] - fac)/prod(x[i] - x[0:i]))
    
    # Obtain polynomial
    poly = np.poly1d(c[0])
    for j in range(1,M):
        single_item = np.poly1d(1.0)
        for k in range(0,j):
            single_item *= np.poly1d([1.0, -x[k]])
        poly += c[j]*single_item
    return poly

def nature_spline_coeff(x, f):
    M = len(x)
    delta = []
    delta_f = []
    coeff_mat = np.matlib.zeros((M-2, M))
    f_mat = np.matlib.zeros((M-2, 1))
    for j in range(1,M):
        delta.append( x[j] - x[j-1])
        delta_f.append((f[j] - f[j-1])/(x[j] - x[j-1]))
    for j in range(1,M-1):
        row = j-1
        coeff_mat[row, j-1] = delta[j-1]/6
        coeff_mat[row, j] = (delta[j-1]+delta[j])/3
        coeff_mat[row, j+1] = delta[j]/6
        f_mat[row, 0] = delta_f[j] - delta_f[j-1]
    # For the nature spline, we given the conditin which is f1" = fn" = 0
    coeff_mat = np.delete(coeff_mat, M-1, 1)
    coeff_mat = np.delete(coeff_mat, 0, 1)

    res = lstsq(coeff_mat, f_mat, rcond=-1)
    ddF = [0]
    for i in range(res[2]):
        ddF.append(res[0].item(i))
    ddF.append(0)
    return ddF, delta, delta_f

def cubic_spline(x, f, sub_xs):
    M = len(x)
    (ddF, delta, delta_f) = nature_spline_coeff(x, f)
    f_vals = []
    for sub_x in sub_xs:
        for j in range(M-1):
            if sub_x <= x[j+1]:
                f_val = ((x[j+1] - sub_x)**3/(6*delta[j]) - delta[j]*(x[j+1] - sub_x)/6)*ddF[j] \
                + ((sub_x - x[j])**3/(6*delta[j]) - delta[j]*(sub_x - x[j])/6)*ddF[j+1] + delta_f[j]*(sub_x - x[j]) + f[j]
                f_vals.append(f_val)
                break
    return f_vals

def cal_error(f, f_hat):
    return np.sqrt((f-f_hat)**2)
