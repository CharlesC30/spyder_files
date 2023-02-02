#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:21:19 2023

@author: charles
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, circulant


def step_func(t, loc):
    return np.where(t < loc, 0, 1)


def gaussian(x, amp, cen, sigma, bkg):
    """1-d gaussian: gaussian(x, amp, cen, sigma, bkg)"""
    return (bkg + amp * np.exp(-(x-cen)**2 / (2*sigma**2)))



def gaussian_matrix(t_in, t_out, sigma):
    # sigma = fwhm / 2.355
    ksi = (t_in[None, :] - t_out[:, None]) / sigma
    bla = np.exp( -0.5 * ksi**2)
    bla = bla / np.sum(bla, axis=1)[:, None] # !!!!!!!!!!!!!!!!!!!
    return bla


def trapz_normalized_gaussian(x, cen, sigma, bkg=0):
    return (0.3989 / (sigma)) * gaussian(x, 1, cen=cen, sigma=sigma, bkg=bkg)

def normalized_gaussian(x, cen, sigma, bkg=0):
    g = gaussian(x, 1, cen=cen, sigma=sigma, bkg=bkg)
    return g / g.sum()


def compare_convolution(t, sig1, sig2, conv_matrix, conv_function):
    res1 = conv_matrix @ sig2
    res2 = conv_function(sig1, sig2)
    plt.plot(t, res1, label="conv matrix")
    plt.plot(t, res2[len(t) // 2: 3*len(t) // 2], label="conv func")

    

def main():
    time = np.arange(-50, 50, 0.1)
    gauss_pulse = normalized_gaussian(time, cen=0, sigma=1)
    
    convolution_matrix = gaussian_matrix(time, time, 1)
    convolution_matrix = np.roll(circulant(gauss_pulse), gauss_pulse.shape[0] // 2)

    plt.plot(time, step_func(time, 0))
    compare_convolution(time, gauss_pulse, step_func(time, 0), convolution_matrix, np.convolve)
    
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()