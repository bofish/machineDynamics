from math import pi, sqrt
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

def DFT(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)    
    return np.dot(M, x)

def IDFT(X):
    """Compute the discrete Fourier Transform of the 1D array x"""
    X = np.asarray(X)
    N = X.shape[0]
    k = np.arange(N)
    n = k.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return np.dot(M, X)/N

def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:(N // 2)] * X_odd,
                               X_even + factor[(N // 2):] * X_odd])

def IFFT(X):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    X = np.asarray(X)
    N = X.shape[0]
    
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return IDFT(X)
    else:
        x_even = IFFT(X[::2])*(N/2)
        x_odd = IFFT(X[1::2])*(N/2)
        factor = np.exp(2j * np.pi * np.arange(N) / N)
        return np.concatenate([x_even + factor[:(N // 2)] * x_odd,x_even + factor[(N // 2):] * x_odd])/N

def FFTshift(x):
    y = np.asarray(x)
    n = y.shape[0]
    p2 = (n+1)//2
    shiftlist = np.concatenate((np.arange(p2, n), np.arange(p2)))
    y = np.take(y, shiftlist)
    return y

def cal_fourier_transform(s, a):
    # cal G(s) by analytical way
    return sqrt(pi/a)*np.exp(-pi**2*s**2/a)

def cal_fourier_series_coeff(a, x_boundary, N):
    # cal f_hat by analytical way
    x_l = x_boundary[0]
    x_u = x_boundary[1]
    series_coeff = [] # f_hat
    for n in range(-N//2,N//2):
        integ_val = erf((x_u+n*1j/(2*a))*sqrt(a)) - erf((x_l+n*1j/(2*a))*sqrt(a))
        f_hat_approximation = np.exp(-n**2/(4*a))*sqrt(pi/a)*integ_val/(4*pi)

        series_coeff.append(f_hat_approximation)
    n_coeff = np.linspace(-N//2, N//2, N)
    return n_coeff, series_coeff

def fourier_series_interpolation(x, series_coeff, N):
    # cal Sf(x) by analytical way
    f_fourier_series= [] # Sf(x)
    for x_n in x:
        sum_temp = []
        for n in range(-N//2,N//2):
            sum_temp.append( series_coeff[n+N//2]*np.exp(1j*n*x_n) )
        f_fourier_series.append(np.sum(sum_temp))
    return f_fourier_series

def get_DFT_coeff(g, N):
    g_tilde = abs(FFT(g)/N)
    n = np.linspace(-N//2, N//2+1, N+1)
    g_tilde_shift = FFTshift(g_tilde)
    g_tilde_shift = np.append(g_tilde_shift, 0)
    return n, g_tilde_shift

if __name__ == '__main__':
    x = np.random.random(2**10)
    print(np.allclose(IFFT(x), np.fft.ifft(x)))
